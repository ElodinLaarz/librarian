from __future__ import annotations

import asyncio
import hashlib
import logging
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import TYPE_CHECKING

import httpx
import numpy as np

from src.config import EmbeddingSettings

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


logger = logging.getLogger(__name__)


async def build_embedding_service(settings: EmbeddingSettings) -> EmbeddingService:
    """Factory to instantiate the correct embedding service based on settings.

    In 'auto' mode, it attempts to load SentenceTransformers first, then Ollama,
    and finally falls back to Dummy embeddings if neither is available or
    fails to initialize.
    """
    provider = settings.provider
    service: EmbeddingService

    if provider == "dummy":
        service = DummyEmbeddingService(settings)
        await service.initialize()
        return service

    if provider == "ollama":
        service = OllamaEmbeddingService(settings)
        await service.initialize()
        return service

    if provider == "sentence-transformers":
        service = SentenceTransformerEmbeddingService(settings)
        await service.initialize()
        return service

    if provider == "auto":
        # 1. Try SentenceTransformers
        try:
            service = SentenceTransformerEmbeddingService(settings)
            await service.initialize()
            return service
        except Exception as exc:
            logger.warning(
                "SentenceTransformers initialization failed, falling back to Ollama: %s",
                exc,
            )

        # 2. Try Ollama
        try:
            service = OllamaEmbeddingService(settings)
            await service.initialize()
            return service
        except Exception as exc:
            logger.warning(
                "Ollama initialization failed, falling back to dummy: %s",
                exc,
            )

        # 3. Fallback to Dummy
        service = DummyEmbeddingService(settings)
        await service.initialize()
        return service

    raise ValueError(f"Unknown embedding provider: {provider}")


class EmbeddingService(ABC):
    """Abstract interface for dense vector embedding generation.

    Concrete implementations (Ollama, sentence-transformers, OpenAI, etc.)
    must provide the methods below. An LRU cache keyed on SHA-256 of input
    text should be maintained by each implementation.
    """

    def __init__(self, settings: EmbeddingSettings) -> None:
        self._settings = settings
        self._measured_dimensions: int | None = None

    @property
    def dimensions(self) -> int:
        """Get the dimensions of the embedding vector.

        Prefers the size measured from the underlying model when ``initialize()``
        has been able to probe it. Falls back to the configured value (used by
        services with no model, like ``DummyEmbeddingService``).
        """
        if self._measured_dimensions is not None:
            return self._measured_dimensions
        return self._settings.dimensions

    def _check_measured_dimensions(self, measured: int) -> None:
        """Validate that ``measured`` is consistent with explicitly configured dimensions.

        If the user explicitly set ``embedding.dimensions`` in their config and it
        disagrees with what the model reports, raise ``ValueError`` so the
        misconfiguration is loud rather than silently producing wrong-sized
        Atlas indexes / ingestion checks.
        """
        if (
            "dimensions" in self._settings.model_fields_set
            and self._settings.dimensions != measured
        ):
            raise ValueError(
                f"Configured dimensions={self._settings.dimensions} but model="
                f"{self._settings.model_name} produces {measured}. "
                "Update embedding.dimensions or remove the setting."
            )

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


class OllamaEmbeddingService(EmbeddingService):
    """Embedding service backed by a local Ollama instance."""

    def __init__(self, settings: EmbeddingSettings) -> None:
        super().__init__(settings)
        self._client = httpx.AsyncClient(base_url=settings.ollama_url, timeout=30.0)
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Verify Ollama is reachable, then probe the model for its real dimensions."""
        try:
            response = await self._client.get("/api/tags")
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self._settings.ollama_url}: {exc}"
            ) from exc

        # Probe the model with a tiny prompt so we discover its real embedding size.
        try:
            probe = await self._client.post(
                "/api/embeddings",
                json={"model": self._settings.model_name, "prompt": "x"},
            )
            probe.raise_for_status()
            vector = probe.json()["embedding"]
            measured = len(vector)
        except (httpx.HTTPError, KeyError, ValueError, TypeError) as exc:
            raise RuntimeError(
                f"Cannot probe Ollama model {self._settings.model_name} "
                f"for embedding dimensions: {exc}"
            ) from exc

        self._check_measured_dimensions(measured)
        self._measured_dimensions = measured

    async def embed(self, text: str) -> np.ndarray:
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()

        async with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]

        response = await self._client.post(
            "/api/embeddings",
            json={"model": self._settings.model_name, "prompt": text},
        )
        response.raise_for_status()
        vector = np.array(response.json()["embedding"], dtype=np.float32)

        async with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            self._cache[key] = vector
            if len(self._cache) > self._settings.cache_size:
                self._cache.popitem(last=False)
            return vector

    async def aclose(self) -> None:
        await self._client.aclose()


class SentenceTransformerEmbeddingService(EmbeddingService):
    """Real embedding service using sentence-transformers."""

    def __init__(self, settings: EmbeddingSettings) -> None:
        super().__init__(settings)
        self._model: SentenceTransformer | None = None
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Load the model and probe it for its true embedding dimension."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            # Loading model can be slow, run in thread
            self._model = await asyncio.to_thread(SentenceTransformer, self._settings.model_name)

        # Ask the model for its dimension; fall back to a probe embedding if needed.
        dim: int | None = None
        getter = getattr(self._model, "get_sentence_embedding_dimension", None)
        if callable(getter):
            try:
                value = await asyncio.to_thread(getter)
            except Exception as exc:  # noqa: BLE001 - any model error means we fall back
                logger.warning(
                    "get_sentence_embedding_dimension failed: %s. Falling back to probe.",
                    exc,
                )
                value = None
            if isinstance(value, int) and value > 0:
                dim = value

        if dim is None:
            probe = await asyncio.to_thread(self._model.encode, "x", convert_to_numpy=True)
            probe_arr = np.atleast_1d(np.squeeze(np.asarray(probe, dtype=np.float32)))
            dim = int(probe_arr.shape[0])

        self._check_measured_dimensions(dim)
        self._measured_dimensions = dim

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

        # Generate embedding in thread (outside lock to not block other lookups)
        embedding = await asyncio.to_thread(self._model.encode, text, convert_to_numpy=True)

        # Ensure it's a 1-D numpy array of float32
        embedding = np.atleast_1d(np.squeeze(np.asarray(embedding, dtype=np.float32)))

        async with self._lock:
            # Another task might have computed and cached it while we were yielding
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]

            # Cache it
            self._cache[key] = embedding

            # Enforce cache size
            if len(self._cache) > self._settings.cache_size:
                self._cache.popitem(last=False)  # Pop oldest (LRU)

            return embedding
