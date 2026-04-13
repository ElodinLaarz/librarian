from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from collections.abc import Awaitable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TypeVar
from uuid import UUID

import httpx

from src import constants
from src.config import LibrarianConfig
from src.models.enums import IngestStatus, SourceType
from src.models.tome import Tome
from src.models.tool_schemas import IngestOutput
from src.services.embedding import EmbeddingService
from src.services.verifier import Verifier
from src.storage.tome_repository import TomeRepository

T = TypeVar("T")


@dataclass
class IngestCallOptions:
    skip_verify: bool = False
    source_type: SourceType = SourceType.AGENT_INPUT
    source_url: str | None = None
    research_job_id: UUID | None = None
    category_hint: str | None = None
    tags_hint: list[str] | None = None


class ReshardError(Exception):
    """Raised when a reshard operation cannot be completed safely."""

    def __init__(self, message: str, tomes: list[Tome] | None = None) -> None:
        super().__init__(message)
        self.tomes = tomes or []


class Ingestor:
    """Receives raw knowledge, validates, reshards, embeds, and stores it as Tomes."""

    def __init__(
        self,
        config: LibrarianConfig,
        embedding_service: EmbeddingService,
        verifier: Verifier,
        tome_repo: TomeRepository,
    ) -> None:
        self._config = config
        self._embedding_service = embedding_service
        self._verifier = verifier
        self._tome_repo = tome_repo

    async def ingest(self, blob: str, options: IngestCallOptions | None = None) -> IngestOutput:
        """Convert unstructured text into one or more Tomes and save them."""
        opts = options or IngestCallOptions()
        if not blob.strip():
            return IngestOutput(
                tomes=[],
                status=IngestStatus.REJECTED,
                reject_reason="Content is empty",
            )

        shards = await self._reshard(blob)
        if not shards:
            return IngestOutput(
                tomes=[],
                status=IngestStatus.REJECTED,
                reject_reason=(
                    f"Unable to re-shard content "
                    f"({len(blob)} chars, shard_size={self._config.ingest.shard_size}, "
                    f"shard_overlap={self._config.ingest.shard_overlap})"
                ),
            )

        tomes = await asyncio.gather(
            *[self._process_text(s, opts) for s in shards],
            return_exceptions=True,
        )

        stored: list[Tome] = []
        any_rejected = False
        reject_reasons: list[str] = []

        for result in tomes:
            if isinstance(result, BaseException):
                any_rejected = True
                if isinstance(result, ReshardError):
                    stored.extend(result.tomes)
                    reject_reasons.append(str(result))
                else:
                    logging.error("Unhandled exception during ingest", exc_info=result)
                    reject_reasons.append(f"Unexpected error: {result}")
            elif not result:
                any_rejected = True
            else:
                stored.extend(result)

        if not stored:
            reason = "All shards failed verification"
            if reject_reasons:
                reasons_str = constants.JOIN_SEPARATOR.join(reject_reasons)
                reason = f"All shards failed verification (or had errors): {reasons_str}"
            return IngestOutput(
                tomes=[],
                status=IngestStatus.REJECTED,
                reject_reason=reason,
            )

        status = IngestStatus.PARTIAL if any_rejected else IngestStatus.STORED
        final_reason = constants.JOIN_SEPARATOR.join(reject_reasons) if reject_reasons else None
        return IngestOutput(tomes=stored, status=status, reject_reason=final_reason)

    async def _process_text(self, text: str, opts: IngestCallOptions) -> list[Tome] | None:
        """Verify, classify, embed, and dedup/store a single text.

        Returns None if verification is enabled and confidence is below the reject
        threshold.  When verification is disabled the text is stored unconditionally
        with a confidence of ingest.unverified_confidence.
        """
        tome = await self._build_tome(text, opts)
        if tome is None:
            return None
        return await self._dedup_and_store(tome, opts)

    async def _build_tome(self, text: str, opts: IngestCallOptions) -> Tome | None:
        """Verify, classify, and embed a text, returning a Tome ready for storage.

        Returns None if verification rejects the text.  Does NOT persist anything.
        """
        should_verify = self._config.verification.enabled and not opts.skip_verify
        if should_verify:
            verification_result = await self._verifier.verify(text)
            if verification_result.confidence < self._config.verification.reject_threshold:
                return None
            confidence = verification_result.confidence
        else:
            confidence = self._config.ingest.unverified_confidence

        (category, tags), (title, summary), embedding = await asyncio.gather(
            self._classify_and_tag(text, opts.category_hint, opts.tags_hint),
            self._generate_title_and_summary(text),
            self._embedding_service.embed(text),
        )

        tome = Tome(
            id=uuid.uuid4(),
            content=text,
            summary=summary,
            title=title,
            category=category,
            tags=tags,
            embedding=embedding,
            source_url=opts.source_url,
            source_type=opts.source_type,
            confidence=confidence,
            research_job_id=opts.research_job_id,
            created_at=datetime.now(UTC),
        )
        try:
            self._validate(tome)
        except ValueError:
            logging.error("Tome validation failed", exc_info=True)
            return None
        return tome

    async def consolidate(self, tomes: list[Tome], skip_verify: bool = False) -> list[Tome]:
        """Merge a list of existing tomes into a new set of resharded tomes.

        Useful for background 'garbage collection' or manual library tidying.
        """
        if not tomes:
            return []
        if len(tomes) == 1:
            return tomes

        # Build replacements from combined content.
        combined = constants.CONTENT_SEPARATOR.join([t.content for t in tomes])
        shards = await self._reshard(combined)

        # Deduplicate shards to avoid redundant tomes.
        unique_shards = list(dict.fromkeys([s.strip() for s in shards if s.strip()]))

        # Use the first tome's metadata as a hint for replacements.
        first = tomes[0]
        opts = IngestCallOptions(
            skip_verify=skip_verify,
            category_hint=first.category,
            tags_hint=first.tags,
            source_url=first.source_url,
            source_type=first.source_type,
            research_job_id=first.research_job_id,
        )

        replacements = await self._build_replacements(unique_shards, opts)

        if not replacements:
            return tomes  # Fallback to original if resharding failed/empty

        # Insert replacements.
        await self._run_in_batches(
            [self._tome_repo.insert(replacement) for replacement in replacements],
            self._config.ingest.write_batch_size,
        )

        # Delete old tomes.
        delete_results = await self._run_in_batches(
            [self._tome_repo.delete(tome.id) for tome in tomes],
            self._config.ingest.write_batch_size,
            return_exceptions=True,
        )
        delete_errors = []
        for tome, result in zip(tomes, delete_results, strict=True):
            if isinstance(result, Exception):
                logging.warning("Exception deleting %s during consolidate: %s", tome.id, result)
                delete_errors.append(str(tome.id))
            elif not result:
                delete_errors.append(str(tome.id))

        if delete_errors:
            failed_ids = constants.ID_SEPARATOR.join(delete_errors)
            msg = (
                "Failed to delete tomes during consolidate "
                f"(duplicate data may exist for: {failed_ids})"
            )
            raise ReshardError(msg, tomes=replacements)

        return replacements

    async def _dedup_and_store(self, tome: Tome, opts: IngestCallOptions) -> list[Tome]:
        """Insert tome, or reshard with any near-duplicates found in the repository.

        Reshard strategy (safe ordering):
        1. Build and verify all replacement tomes from combined content.
        2. Abort without touching existing tomes if no replacement survives verification.
        3. Insert all replacements first to ensure no data loss if the operation is interrupted.
        4. Delete old tomes only once all replacements are successfully persisted.
        """
        duplicates = await self._tome_repo.find_near_duplicates(tome)
        if not duplicates:
            await self._tome_repo.insert(tome)
            return [tome]

        # Step 1 — build replacements from combined content (nothing persisted yet).
        combined = constants.CONTENT_SEPARATOR.join(
            [d.content for d in duplicates] + [tome.content]
        )
        shards = await self._reshard(combined)

        # Deduplicate shards to avoid redundant tomes.
        unique_shards = list(dict.fromkeys([s.strip() for s in shards if s.strip()]))
        replacements = await self._build_replacements(unique_shards, opts)

        # Step 2 — abort if verification left us with nothing to store.
        if not replacements:
            return []

        # Step 3 - Insert replacements.
        await self._run_in_batches(
            [self._tome_repo.insert(replacement) for replacement in replacements],
            self._config.ingest.write_batch_size,
        )

        # Step 4 - Delete old tomes.
        # Technically if we fail here, we may end up with duplicate data in the
        # library, but that seems like a better choice (IMO) than aborting.
        delete_results = await self._run_in_batches(
            [self._tome_repo.delete(dup.id) for dup in duplicates],
            self._config.ingest.write_batch_size,
            return_exceptions=True,
        )
        delete_errors = []
        for dup, result in zip(duplicates, delete_results, strict=True):
            if isinstance(result, Exception):
                logging.warning("Exception deleting %s during reshard: %s", dup.id, result)
                delete_errors.append(str(dup.id))
            elif not result:
                delete_errors.append(str(dup.id))

        if delete_errors:
            failed_ids = constants.ID_SEPARATOR.join(delete_errors)
            msg = (
                "Failed to delete tomes during reshard "
                f"(duplicate data may exist for: {failed_ids})"
            )
            raise ReshardError(msg, tomes=replacements)

        return replacements

    async def _reshard(self, blob: str) -> list[str]:
        """Split blob into atomic, self-contained fact shards."""
        if self._config.ingest.use_llm_chunking:
            llm_shards = await self._reshard_llm(blob)
            if llm_shards:
                return llm_shards

        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._config.ingest.shard_size,
            chunk_overlap=self._config.ingest.shard_overlap,
        )
        return await asyncio.to_thread(splitter.split_text, blob)

    async def _build_replacements(self, shards: list[str], opts: IngestCallOptions) -> list[Tome]:
        replacement_results = await self._gather_limited(
            [self._build_tome(chunk, opts) for chunk in shards],
            self._config.ingest.build_concurrency,
        )
        return [tome for tome in replacement_results if tome is not None]

    async def _gather_limited(
        self,
        operations: list[Awaitable[T | None]],
        concurrency: int,
    ) -> list[T | None]:
        semaphore = asyncio.Semaphore(max(1, concurrency))

        async def _run(operation: Awaitable[T | None]) -> T | None:
            async with semaphore:
                return await operation

        return await asyncio.gather(*[_run(operation) for operation in operations])

    async def _run_in_batches(
        self,
        operations: list[Awaitable[T]],
        batch_size: int,
        *,
        return_exceptions: bool = False,
    ) -> list[T | BaseException]:
        results: list[T | BaseException] = []
        for start in range(0, len(operations), max(1, batch_size)):
            batch = operations[start : start + max(1, batch_size)]
            batch_results = await asyncio.gather(*batch, return_exceptions=return_exceptions)
            results.extend(batch_results)
        return results

    async def _reshard_llm(self, blob: str) -> list[str] | None:
        """Use an LLM agent to decompose text into atomic facts."""
        base = self._config.ingest.ollama_base_url.rstrip("/")
        model = self._config.ingest.extraction_model
        prompt = (
            "Decompose the following text into a list of atomic, self-contained factual "
            "statements or concepts. Each statement or concept must contain enough context "
            "to be fully understood on its own. "
            f"Do not exceed {self._config.ingest.shard_size} characters per fact. "
            'Output JSON only with the shape `{"facts": ["...", "..."]}`.\\n\\nTEXT:\\n'
            f"{blob[:8000]}"
        )
        url = f"{base}/v1/chat/completions"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
        }
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
        except (httpx.HTTPError, OSError, ValueError, KeyError) as exc:
            logging.debug("_reshard_llm HTTP/parse error: %s", exc)
            return None

        try:
            message = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            return None

        message = message.strip()
        if message.startswith("```"):
            message = re.sub(r"^```(?:json)?\\s*", "", message)
            message = re.sub(r"\\s*```$", "", message)

        try:
            parsed = json.loads(message)
            facts = parsed.get("facts") if isinstance(parsed, dict) else None
            if not isinstance(facts, list):
                return None
            out = [str(f).strip() for f in facts if str(f).strip()]
        except (json.JSONDecodeError, TypeError, ValueError):
            return None

        if not out:
            return None
        return out

    async def _classify_and_tag(
        self,
        text: str,
        category_hint: str | None = None,
        tags_hint: list[str] | None = None,
    ) -> tuple[str, list[str]]:
        """Auto-classify into a category and extract topic tags."""
        category = category_hint or self._config.ingest.default_category
        if tags_hint:
            merged = list(dict.fromkeys([*tags_hint, *self._config.ingest.default_tags]))
            return category, merged
        return category, list(self._config.ingest.default_tags)

    async def _generate_title_and_summary(self, text: str) -> tuple[str, str]:
        """Generate a short title and one-to-two sentence summary for a text."""
        clean_text = text.strip().replace("\n", " ")
        if len(clean_text) > self._config.ingest.title_length:
            suffix = constants.TRUNCATION_SUFFIX
            title = clean_text[: self._config.ingest.title_length - len(suffix)] + suffix
        else:
            title = clean_text
        summary = clean_text[: self._config.ingest.summary_length]
        return title, summary

    def _validate(self, tome: Tome) -> None:
        """Post-construction checks; raises on failure."""
        if not tome.content.strip():
            raise ValueError("Tome content cannot be empty")
        if len(tome.title) > self._config.ingest.title_length:
            raise ValueError(
                f"Tome title too long ({len(tome.title)} > {self._config.ingest.title_length})"
            )

        # In a real implementation we would enforce the embedding size.
        # Here we skip the dimension check if a dummy/string embedding is supplied.
        if (
            tome.embedding is not None
            and tome.embedding.shape[0] != self._config.embedding.dimensions
        ):
            raise ValueError(
                f"Tome embedding has dimension {tome.embedding.shape[0]}, "
                f"expected {self._config.embedding.dimensions}"
            )
