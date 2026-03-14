from __future__ import annotations

from src.config import SearchSettings
from src.models.tool_schemas import SearchInput, SearchOutput
from src.services.embedding import EmbeddingService
from src.storage.tome_repository import TomeRepository


class SearchEngine:
    """Converts natural-language queries to vector embeddings and retrieves
    the most semantically relevant Tomes from the library."""

    def __init__(
        self,
        settings: SearchSettings,
        embedding_service: EmbeddingService,
        tome_repo: TomeRepository,
    ) -> None:
        self._settings = settings
        self._embedding_service = embedding_service
        self._tome_repo = tome_repo

    async def search(self, params: SearchInput) -> SearchOutput:
        """Execute the full search pipeline: embed -> vector search -> re-rank -> return."""
        ...

    async def _embed_query(self, query: str) -> tuple[list[float], bool]:
        """Embed the query string. Returns (vector, from_cache)."""
        ...

    def _rerank(
        self,
        results: list[tuple],
        recency_boost: float = 0.1,
    ) -> list[tuple]:
        """Re-rank results by blending similarity score with recency and confidence."""
        ...
