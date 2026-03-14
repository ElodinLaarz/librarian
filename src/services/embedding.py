from __future__ import annotations

from abc import ABC, abstractmethod

from src.config import EmbeddingSettings


class EmbeddingService(ABC):
    """Abstract interface for dense vector embedding generation.

    Concrete implementations (Ollama, sentence-transformers, OpenAI, etc.)
    must provide the methods below. An LRU cache keyed on SHA-256 of input
    text should be maintained by each implementation.
    """

    def __init__(self, settings: EmbeddingSettings) -> None:
        self._settings = settings

    @abstractmethod
    async def initialize(self) -> None:
        """Load the embedding model and warm up the provider connection."""
        ...

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Produce a dense vector embedding for a single text string.

        Returns from cache if the text has been embedded before.
        """
        ...

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in a single call for throughput."""
        ...

    def _cache_key(self, text: str) -> str:
        """Compute SHA-256 hash of text for cache lookup."""
        ...

    @property
    def dimensions(self) -> int:
        """Return the dimensionality of the active model's output vectors."""
        return self._settings.dimensions
