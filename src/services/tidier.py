from __future__ import annotations

import asyncio
import logging
from time import perf_counter
from typing import TYPE_CHECKING
from uuid import UUID

from src.config import TidySettings
from src.models.tome import Tome

if TYPE_CHECKING:
    from src.services.ingestor import Ingestor
    from src.storage.tome_repository import TomeRepository


logger = logging.getLogger(__name__)


class Tidier:
    """A background agent that scans the library for duplicates and consolidates them."""

    def __init__(
        self,
        ingestor: Ingestor,
        tome_repo: TomeRepository,
        settings: TidySettings | None = None,
    ) -> None:
        self._ingestor = ingestor
        self._tome_repo = tome_repo
        self._settings = settings or TidySettings()

    async def run_cleanup(
        self,
        limit: int | None = None,
        threshold: float | None = None,
        skip_verify: bool | None = None,
    ) -> dict[str, int]:
        """Identify and consolidate similar tomes in bulk.

        Returns a report of total tomes scanned, groups identified, and consolidations performed.
        """
        started = perf_counter()
        effective_limit = limit if limit is not None else self._settings.limit_per_run
        effective_threshold = threshold if threshold is not None else self._settings.threshold
        effective_skip_verify = (
            skip_verify if skip_verify is not None else self._settings.skip_verify
        )

        logger.info("Scanning library for redundant data (threshold=%.2f)...", effective_threshold)
        scan_result = await self._tome_repo.find_all_near_duplicates(threshold=effective_threshold)
        groups_found = len(scan_result.groups)

        groups_to_process, skipped_groups = self._select_groups(
            scan_result.groups,
            effective_limit,
        )

        if not groups_to_process:
            logger.info("No redundant data found.")
            return {
                "scanned": scan_result.scanned,
                "groups_found": groups_found,
                "groups_consolidated": 0,
                "tomes_removed": 0,
                "failed_groups": 0,
                "skipped_groups": skipped_groups,
                "elapsed_ms": int((perf_counter() - started) * 1000),
            }

        logger.info(
            "Identified %d duplicate groups, processing %d.",
            groups_found,
            len(groups_to_process),
        )

        semaphore = asyncio.Semaphore(max(1, self._settings.group_concurrency))

        async def _run_group(group: list[Tome]) -> tuple[int, int]:
            async with semaphore:
                return await self._consolidate_group(group, effective_skip_verify)

        results = await asyncio.gather(
            *[_run_group(group) for group in groups_to_process],
            return_exceptions=True,
        )

        consolidated_count = 0
        removed_count = 0
        failed_groups = 0

        for result in results:
            if isinstance(result, BaseException):
                failed_groups += 1
                logger.error("Batch consolidation error: %s", result)
                continue
            consolidated_count += result[0]
            removed_count += result[1]

        return {
            "scanned": scan_result.scanned,
            "groups_found": groups_found,
            "groups_consolidated": consolidated_count,
            "tomes_removed": removed_count,
            "failed_groups": failed_groups,
            "skipped_groups": skipped_groups,
            "elapsed_ms": int((perf_counter() - started) * 1000),
        }

    def _select_groups(self, groups: list[list[Tome]], limit: int) -> tuple[list[list[Tome]], int]:
        selected: list[list[Tome]] = []
        used_ids: set[UUID] = set()
        skipped_groups = 0

        for group in groups:
            deduped_group = list({tome.id: tome for tome in group}.values())
            if len(deduped_group) < 2:
                skipped_groups += 1
                continue
            group_ids = {tome.id for tome in deduped_group}
            if group_ids & used_ids:
                skipped_groups += 1
                continue
            selected.append(deduped_group)
            used_ids.update(group_ids)

        if len(selected) > limit:
            skipped_groups += len(selected) - limit
            selected = selected[:limit]

        return selected, skipped_groups

    async def _consolidate_group(self, group: list[Tome], skip_verify: bool) -> tuple[int, int]:
        """Internal helper to consolidate a single group and return counts."""
        logger.info("Consolidating group of %d tomes.", len(group))
        try:
            replacements = await self._ingestor.consolidate(group, skip_verify=skip_verify)
            return 1, len(group) - len(replacements)
        except Exception as exc:
            logger.error("Failed to consolidate group: %s", exc)
            raise
