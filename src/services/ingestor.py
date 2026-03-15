from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime

from src.config import LibrarianConfig
from src.models.enums import SourceType
from src.models.tome import Tome
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

    async def ingest(self, blob: str) -> list[Tome]:
        """Convert unstructured text into a structured knowledge Tome
        and save it in the collection."""

        # Run async methods concurrently
        classify_task = self._classify_and_tag(blob)
        summarize_task = self._generate_title_and_summary(blob)
        embed_task = self._embedding_service.embed(blob)

        # Wait for all three to finish
        (category, tags), (title, summary), embedding = await asyncio.gather(
            classify_task, summarize_task, embed_task
        )

        timestamp = datetime.now(tz=UTC)
        tome = Tome(
            id=uuid.uuid4(),
            content=blob,
            summary=summary,
            title=title,
            category=category,
            tags=tags,
            embedding=embedding,
            created_at=timestamp,
            source_url=None,
            source_type=SourceType.AGENT_INPUT,
            confidence=0.5,
        )

        # Validate (sync method)
        self._validate(tome)

        # Store async
        await self._tome_repo.insert(tome)

        return [tome]

    def _validate(self, params: Tome) -> None:
        """Pre-flight checks: input length, required fields, HTML sanitisation."""

    async def _classify_and_tag(
        self, chunk: str, category_hint: str | None = None
    ) -> tuple[str, list[str]]:
        """Auto-classify into a category and extract topic tags."""
        raise NotImplementedError

    async def _generate_title_and_summary(self, chunk: str) -> tuple[str, str]:
        """Generate a short title and one-to-two sentence summary for a chunk."""
        raise NotImplementedError

    async def _dedup_and_store(self, tome: Tome, allow_update: bool) -> str:
        """Check for near-duplicates; merge, skip, or insert accordingly. Returns the Tome ID."""
        raise NotImplementedError
