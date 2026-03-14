from __future__ import annotations

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
        """Run the full ingest pipeline: validate -> verify -> chunk -> embed -> dedup -> store."""
        category, tags = self._classify_and_tag(blob)
        title, summary = self._generate_title_and_summary(blob)
        embedding = await self._embedding_service.embed(blob)
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
        self._validate(tome)
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
