from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

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
    async def embed(self, text: str) -> np.ndarray | str | list[float]:
        """Produce a dense vector embedding for a single text string.

        Returns from cache if the text has been embedded before.
        """
        ...


class DummyEmbeddingService(EmbeddingService):
    """A placeholder embedding service that returns the text directly for local dev/testing."""

    async def initialize(self) -> None:
        pass

    async def embed(self, text: str) -> str:
        return text
