from __future__ import annotations

import asyncio
import hashlib
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import TYPE_CHECKING

import numpy as np

from src.config import EmbeddingSettings

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class EmbeddingService(ABC):
    """Abstract interface for dense vector embedding generation.

    Concrete implementations (Ollama, sentence-transformers, OpenAI, etc.)
    must provide the methods below. An LRU cache keyed on SHA-256 of input
    text should be maintained by each implementation.
    """

    def __init__(self, settings: EmbeddingSettings) -> None:
        self._settings = settings

    @property
    def dimensions(self) -> int:
        """Get the dimensions of the embedding vector."""
        return self._settings.dimensions

    @abstractmethod
    async def initialize(self) -> None:
        """Load the embedding model and warm up the provider connection."""
        ...

    @abstractmethod
    async def embed(self, text: str) -> np.ndarray:
        """Produce a dense vector embedding for a single text string.

        Returns from cache if the text has been embedded before.
        """
        ...


class DummyEmbeddingService(EmbeddingService):
    """A placeholder embedding service that returns zero vectors for local dev/testing."""

    async def initialize(self) -> None:
        pass

    async def embed(self, text: str) -> np.ndarray:
        return np.zeros(self._settings.dimensions, dtype=np.float32)


class SentenceTransformerEmbeddingService(EmbeddingService):
    """Real embedding service using sentence-transformers."""

    def __init__(self, settings: EmbeddingSettings) -> None:
        super().__init__(settings)
        self._model: SentenceTransformer | None = None
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Load the model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            # Loading model can be slow, run in thread
            self._model = await asyncio.to_thread(SentenceTransformer, self._settings.model_name)

    async def embed(self, text: str) -> np.ndarray:
        """Produce embedding with LRU cache."""
        if self._model is None:
            raise RuntimeError("Model not initialized. Call initialize() before embed().")

        # Compute SHA-256 hash of text for key
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()

        async with self._lock:
            if key in self._cache:
                # Move to end (MRU)
                self._cache.move_to_end(key)
                return self._cache[key]

            # Generate embedding in thread
            embedding = await asyncio.to_thread(self._model.encode, text, convert_to_numpy=True)

            # Ensure it's a 1-D numpy array of float32
            embedding = np.asarray(embedding, dtype=np.float32)
            if embedding.ndim == 2 and embedding.shape[0] == 1:
                embedding = embedding[0]
            else:
                embedding = np.squeeze(embedding)

            # Cache it
            self._cache[key] = embedding

            # Enforce cache size
            if len(self._cache) > self._settings.cache_size:
                self._cache.popitem(last=False)  # Pop oldest (LRU)

            return embedding
