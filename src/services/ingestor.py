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

        Returns None if verification confidence is below the reject threshold.
        """
        verify_task = self._verifier.verify(chunk)
        classify_task = self._classify_and_tag(chunk)
        summarize_task = self._generate_title_and_summary(chunk)
        embed_task = self._embedding_service.embed(chunk)

        verification, (category, tags), (title, summary), embedding = (
            await asyncio.gather(verify_task, classify_task, summarize_task, embed_task)
        )

        if verification.confidence < self._config.verification.reject_threshold:
            return None

        tome = Tome(
            id=uuid.uuid4(),
            content=chunk,
            summary=summary,
            title=title,
            category=category,
            tags=tags,
            embedding=embedding,
            source_url=None,
            source_type=SourceType.AGENT_INPUT,
            confidence=verification.confidence,
        )
        return await self._dedup_and_store(tome)

    async def _dedup_and_store(self, tome: Tome) -> list[Tome]:
        """Insert tome, or reshard with any near-duplicates found in the repository.

        Reshard: combine the new tome's content with duplicate content, delete the
        old tomes, re-chunk, and re-process each new chunk through the full pipeline.
        Because the old tomes are deleted before re-processing, subsequent
        find_near_duplicates calls on the new chunks will not match them again.
        """
        duplicates = await self._tome_repo.find_near_duplicates(tome)
        if not duplicates:
            await self._tome_repo.insert(tome)
            return [tome]

        # Combine existing duplicate content with the incoming content.
        combined = "\n\n".join([d.content for d in duplicates] + [tome.content])

        # Delete old tomes before inserting resharded replacements.
        for dup in duplicates:
            await self._tome_repo.delete(dup.id)

        new_chunks = await self._chunk(combined)
        chunk_results = await asyncio.gather(
            *[self._process_chunk(c) for c in new_chunks]
        )
        results: list[Tome] = []
        for result in chunk_results:
            if result is not None:
                results.extend(result)
        return results

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
