from __future__ import annotations

import asyncio
import uuid

from src.config import LibrarianConfig
from src.models.enums import IngestStatus, SourceType
from src.models.tome import Tome
from src.models.tool_schemas import IngestOutput
from src.services.embedding import EmbeddingService
from src.services.verifier import Verifier
from src.storage.tome_repository import TomeRepository


class ReshardError(Exception):
    """Raised when a reshard operation cannot be completed safely."""


class Ingestor:
    """Receives raw knowledge, validates, chunks, embeds, and stores it as Tomes."""

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

    async def ingest(self, blob: str) -> IngestOutput:
        """Convert unstructured text into one or more Tomes and save them."""
        if not blob.strip():
            return IngestOutput(
                tomes=[],
                status=IngestStatus.REJECTED,
                reject_reason="Content is empty",
            )

        chunks = await self._chunk(blob)
        if not chunks:
            return IngestOutput(
                tomes=[],
                status=IngestStatus.REJECTED,
                reject_reason="Content too short to chunk",
            )

        chunk_results = await asyncio.gather(*[self._process_chunk(c) for c in chunks])

        stored: list[Tome] = []
        any_rejected = False
        for result in chunk_results:
            if result is None:
                any_rejected = True
            else:
                stored.extend(result)

        if not stored:
            return IngestOutput(
                tomes=[],
                status=IngestStatus.REJECTED,
                reject_reason="All chunks failed verification",
            )

        status = IngestStatus.PARTIAL if any_rejected else IngestStatus.STORED
        return IngestOutput(tomes=stored, status=status)

    async def _process_chunk(self, chunk: str) -> list[Tome] | None:
        """Verify, classify, embed, and dedup/store a single chunk.

        Returns None if verification is enabled and confidence is below the reject
        threshold.  When verification is disabled the chunk is stored unconditionally
        with a confidence of 1.0.
        """
        tome = await self._build_tome(chunk)
        if tome is None:
            return None
        return await self._dedup_and_store(tome)

    async def _build_tome(self, chunk: str) -> Tome | None:
        """Verify, classify, and embed a chunk, returning a Tome ready for storage.

        Returns None if verification rejects the chunk.  Does NOT persist anything.
        """
        if self._config.verification.enabled:
            (
                verification_result,
                (category, tags),
                (title, summary),
                embedding,
            ) = await asyncio.gather(
                self._verifier.verify(chunk),
                self._classify_and_tag(chunk),
                self._generate_title_and_summary(chunk),
                self._embedding_service.embed(chunk),
            )
            if verification_result.confidence < self._config.verification.reject_threshold:
                return None
            confidence = verification_result.confidence
        else:
            (category, tags), (title, summary), embedding = await asyncio.gather(
                self._classify_and_tag(chunk),
                self._generate_title_and_summary(chunk),
                self._embedding_service.embed(chunk),
            )
            confidence = 1.0

        return Tome(
            id=uuid.uuid4(),
            content=chunk,
            summary=summary,
            title=title,
            category=category,
            tags=tags,
            embedding=embedding,
            source_url=None,
            source_type=SourceType.AGENT_INPUT,
            confidence=confidence,
        )

    async def _dedup_and_store(self, tome: Tome) -> list[Tome]:
        """Insert tome, or reshard with any near-duplicates found in the repository.

        Reshard strategy (safe ordering):
        1. Build and verify all replacement tomes from combined content.
        2. Abort without touching existing tomes if no replacement survives verification.
        3. Delete old tomes only once at least one replacement is ready — checking
           each delete return value and raising ReshardError on failure.
        4. Insert all replacements.
        """
        duplicates = await self._tome_repo.find_near_duplicates(tome)
        if not duplicates:
            await self._tome_repo.insert(tome)
            return [tome]

        # Step 1 — build replacements from combined content (nothing persisted yet).
        combined = "\n\n".join([d.content for d in duplicates] + [tome.content])
        new_chunks = await self._chunk(combined)
        replacement_results = await asyncio.gather(*[self._build_tome(c) for c in new_chunks])
        replacements = [t for t in replacement_results if t is not None]

        # Step 2 — abort if verification left us with nothing to store.
        if not replacements:
            return []

        # Step 3 — safe to delete now that we have confirmed replacements.
        for dup in duplicates:
            deleted = await self._tome_repo.delete(dup.id)
            if not deleted:
                raise ReshardError(
                    f"Failed to delete tome {dup.id!r} during reshard; "
                    "aborting to prevent duplicate data."
                )

        # Step 4 — insert all replacements.
        for replacement in replacements:
            await self._tome_repo.insert(replacement)

        return replacements

    async def _chunk(self, blob: str) -> list[str]:
        """Split blob into atomic, self-contained fact chunks.

        Subclasses override this with LLM-driven decomposition.
        """
        raise NotImplementedError

    async def _classify_and_tag(
        self, chunk: str, category_hint: str | None = None
    ) -> tuple[str, list[str]]:
        """Auto-classify into a category and extract topic tags."""
        raise NotImplementedError

    async def _generate_title_and_summary(self, chunk: str) -> tuple[str, str]:
        """Generate a short title and one-to-two sentence summary for a chunk."""
        raise NotImplementedError
