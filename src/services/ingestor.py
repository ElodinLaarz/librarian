from __future__ import annotations

from src.config import LibrarianConfig
from src.models.tome import Tome
from src.models.tool_schemas import IngestInput, IngestOutput
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

    async def ingest(self, params: IngestInput) -> IngestOutput:
        """Run the full ingest pipeline: validate -> verify -> chunk -> embed -> dedup -> store."""
        raise NotImplementedError

    def _validate(self, params: IngestInput) -> None:
        """Pre-flight checks: input length, required fields, HTML sanitisation."""
        raise NotImplementedError

    def _chunk(self, content: str) -> list[str]:
        """Split content into single-topic chunks of ~400 words
        using sentence-boundary splitting."""
        raise NotImplementedError

    async def _classify_and_tag(
        self, chunk: str, category_hint: str | None
    ) -> tuple[str, list[str]]:
        """Auto-classify into a category and extract topic tags."""
        raise NotImplementedError

    async def _generate_title_and_summary(
        self, chunk: str
    ) -> tuple[str, str]:
        """Generate a short title and one-to-two sentence summary for a chunk."""
        raise NotImplementedError

    async def _dedup_and_store(
        self, tome: Tome, allow_update: bool
    ) -> str:
        """Check for near-duplicates; merge, skip, or insert accordingly. Returns the Tome ID."""
        raise NotImplementedError
