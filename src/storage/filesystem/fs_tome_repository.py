from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np
from numpy.typing import NDArray

from src.config import DatabaseSettings, TidySettings
from src.models.tome import Tome
from src.services.duplicate_detection import build_duplicate_groups
from src.services.embedding import EmbeddingService
from src.storage.errors import (
    BackendUnavailableError,
    StorageError,
)
from src.storage.filesystem.utils import resolve_base_path
from src.storage.tome_repository import DuplicateScanResult, TomeRepository


def _cosine_similarity(a: NDArray[np.floating[Any]], b: NDArray[np.floating[Any]]) -> float:
    """Cosine similarity in [-1, 1]. Returns 0.0 if shapes differ or either vector is zero."""
    if a.shape != b.shape:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _wrap_os(exc: OSError, context: str, path: Path | None = None) -> StorageError:
    """Translate a low-level filesystem error into the right StorageError subclass.

    Connectivity-style failures (missing root, permission denied on root)
    map to ``BackendUnavailableError``; everything else falls through to
    ``StorageError``. Callers must use ``raise ... from exc`` to preserve
    the original cause.
    """
    if isinstance(exc, FileNotFoundError | PermissionError):
        return BackendUnavailableError(
            f"Filesystem storage unavailable during {context}" + (f": {path}" if path else "")
        )
    return StorageError(f"Filesystem {context} failed: {exc.__class__.__name__}")


class FsTomeRepository(TomeRepository):
    """File-system implementation of the TomeRepository.

    Stores each Tome as a JSON file under ``<base_path>/tomes/``.
    Uses the injected :class:`~src.services.embedding.EmbeddingService` to
    embed search queries so cosine similarity scores are real rather than
    placeholder 1.0 values.
    """

    def __init__(
        self,
        settings: DatabaseSettings,
        embedding_service: EmbeddingService,
        tidy_settings: TidySettings | None = None,
    ) -> None:
        self._embedding_service = embedding_service
        self._tidy_settings = tidy_settings or TidySettings()
        self._tomes_dir = resolve_base_path(settings.uri) / settings.tomes_collection
        try:
            self._tomes_dir.mkdir(parents=True, exist_ok=True)
        except (FileNotFoundError, PermissionError) as exc:
            raise BackendUnavailableError(
                f"Filesystem storage root unreachable: {self._tomes_dir}"
            ) from exc
        except OSError as exc:
            raise StorageError(f"Filesystem storage root error: {self._tomes_dir}") from exc

    def _get_path(self, tome_id: UUID) -> Path:
        return self._tomes_dir / f"{tome_id}.json"

    def _read_all_tomes_sync(self) -> list[Tome]:
        """Synchronous helper to scan and read all Tomes from disk."""
        tomes: list[Tome] = []
        for path in self._tomes_dir.glob("*.json"):
            try:
                tome = Tome.model_validate_json(path.read_text())
                tomes.append(tome)
            except Exception:
                continue
        return tomes

    async def insert(self, tome: Tome) -> UUID:
        """Save a Tome as a JSON file.

        Note: the filesystem backend has no native unique constraint, so a
        same-id rewrite silently overwrites (matching the pre-taxonomy
        contract). ``DuplicateError`` is reserved for backends that enforce
        uniqueness server-side (e.g. Mongo's ``DuplicateKeyError``).
        """
        path = self._get_path(tome.id)
        try:
            await asyncio.to_thread(path.write_text, tome.model_dump_json(indent=2))
        except OSError as exc:
            raise _wrap_os(exc, "insert", path) from exc
        return tome.id

    async def delete(self, tome_id: UUID) -> bool:
        """Deletes a Tome by ID. Returns True if a file was removed."""
        path = self._get_path(tome_id)
        try:
            exists = await asyncio.to_thread(path.exists)
            if not exists:
                return False
            await asyncio.to_thread(path.unlink)
        except FileNotFoundError:
            # Race: another worker unlinked between exists() and unlink().
            return False
        except OSError as exc:
            raise _wrap_os(exc, "delete", path) from exc
        return True

    async def get_by_id(self, tome_id: UUID) -> Tome | None:
        """Read a Tome from its JSON file."""
        path = self._get_path(tome_id)
        try:
            exists = await asyncio.to_thread(path.exists)
        except OSError as exc:
            raise _wrap_os(exc, "get_by_id", path) from exc
        if not exists:
            return None
        try:
            content = await asyncio.to_thread(path.read_text)
        except FileNotFoundError:
            # Race: file removed between exists() and read.
            return None
        except OSError as exc:
            raise _wrap_os(exc, "get_by_id", path) from exc
        try:
            return Tome.model_validate_json(content)
        except Exception:
            # JSON parse / schema mismatch — preserve previous "treat as
            # absent" contract.
            return None

    def _matches_filters(
        self,
        tome: Tome,
        *,
        category: str | None,
        min_confidence: float,
        research_job_id: UUID | None,
    ) -> bool:
        if category is not None and tome.category != category:
            return False
        if tome.confidence < min_confidence:
            return False
        return not (research_job_id is not None and tome.research_job_id != research_job_id)

    async def list_all(
        self,
        limit: int = 100,
        offset: int = 0,
        *,
        category: str | None = None,
        min_confidence: float = 0.0,
        research_job_id: UUID | None = None,
    ) -> list[Tome]:
        """Return a filtered + paginated page of Tomes, newest first.

        Scans every JSON file under the tomes directory and filters in
        memory — acknowledged to be O(n) per call; acceptable for the
        filesystem backend's small-library use case. Tomes that fail to
        parse are silently skipped (matching the existing search-path
        behaviour).
        """
        try:
            all_tomes = await asyncio.to_thread(self._read_all_tomes_sync)
        except OSError as exc:
            raise _wrap_os(exc, "list_all", self._tomes_dir) from exc

        filtered = [
            t
            for t in all_tomes
            if self._matches_filters(
                t,
                category=category,
                min_confidence=min_confidence,
                research_job_id=research_job_id,
            )
        ]
        filtered.sort(key=lambda x: x.created_at, reverse=True)
        return filtered[offset : offset + limit]

    async def count(
        self,
        *,
        category: str | None = None,
        min_confidence: float = 0.0,
        research_job_id: UUID | None = None,
    ) -> int:
        """Count Tomes matching the same filter predicates as :meth:`list_all`."""
        try:
            all_tomes = await asyncio.to_thread(self._read_all_tomes_sync)
        except OSError as exc:
            raise _wrap_os(exc, "count", self._tomes_dir) from exc
        return sum(
            1
            for t in all_tomes
            if self._matches_filters(
                t,
                category=category,
                min_confidence=min_confidence,
                research_job_id=research_job_id,
            )
        )

    async def search(
        self,
        query: str,
        top_k: int = 5,
        min_confidence: float = 0.5,
        category: str | None = None,
    ) -> list[tuple[Tome, float]]:
        """Brute-force cosine-similarity scan over all stored Tomes.

        Embeds *query* with the injected :class:`EmbeddingService` and ranks
        results by cosine similarity to each Tome's stored embedding.
        Tomes without an embedding are still included but scored 0.0.
        """
        query_vec: NDArray[np.float64] | None = None
        try:
            raw = await self._embedding_service.embed(query)
            query_vec = np.array(raw, dtype=np.float64)
        except Exception:
            pass

        results: list[tuple[Tome, float]] = []
        try:
            all_tomes = await asyncio.to_thread(self._read_all_tomes_sync)
        except OSError as exc:
            raise _wrap_os(exc, "search", self._tomes_dir) from exc

        for tome in all_tomes:
            if tome.confidence < min_confidence:
                continue
            if category is not None and tome.category != category:
                continue

            score = 0.0
            if query_vec is not None and tome.embedding is not None:
                score = _cosine_similarity(query_vec, np.array(tome.embedding, dtype=np.float64))

            results.append((tome, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    async def find_near_duplicates(self, tome: Tome, threshold: float | None = None) -> list[Tome]:
        """Find existing Tomes with cosine similarity above the configured tidy threshold."""
        if tome.embedding is None:
            return []
        tome_vec = np.array(tome.embedding, dtype=np.float64)
        duplicates: list[Tome] = []

        effective_threshold = threshold if threshold is not None else self._tidy_settings.threshold
        try:
            all_tomes = await asyncio.to_thread(self._read_all_tomes_sync)
        except OSError as exc:
            raise _wrap_os(exc, "find_near_duplicates", self._tomes_dir) from exc

        for existing in all_tomes:
            if existing.id == tome.id or existing.embedding is None:
                continue
            sim = _cosine_similarity(tome_vec, np.array(existing.embedding, dtype=np.float64))
            if sim >= effective_threshold:
                duplicates.append(existing)
        return duplicates

    async def find_all_near_duplicates(self, threshold: float = 0.95) -> DuplicateScanResult:
        """Find duplicate groups for tidy-time consolidation."""
        try:
            all_tomes = await asyncio.to_thread(self._read_all_tomes_sync)
        except OSError as exc:
            raise _wrap_os(exc, "find_all_near_duplicates", self._tomes_dir) from exc
        return build_duplicate_groups(
            all_tomes,
            self._tidy_settings.model_copy(update={"threshold": threshold}),
        )

    def close(self) -> None:
        pass
