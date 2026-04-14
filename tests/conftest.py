"""Shared pytest fixtures for the Librarian test suite."""

from __future__ import annotations

import hashlib
import json
from collections.abc import AsyncGenerator

import httpx
import numpy as np
import pytest

from src.config import DatabaseSettings, EmbeddingSettings, LibrarianConfig
from src.services.embedding import OllamaEmbeddingService
from tests.stubs import StubEmbeddingService, StubIngestor, StubTomeRepository, StubVerifier

_TEST_DB_SETTINGS = DatabaseSettings(
    uri="mongodb://localhost:27017/?directConnection=true",
    tls=False,
)


def make_test_config(**overrides: object) -> LibrarianConfig:
    """Build a LibrarianConfig with dummy database credentials for tests."""
    overrides.setdefault("database", _TEST_DB_SETTINGS)
    return LibrarianConfig(**overrides)


@pytest.fixture
def config() -> LibrarianConfig:
    return make_test_config()


@pytest.fixture
def repo() -> StubTomeRepository:
    return StubTomeRepository()


@pytest.fixture
def verifier() -> StubVerifier:
    return StubVerifier(confidence=0.8)


@pytest.fixture
def embedding_service(config: LibrarianConfig) -> StubEmbeddingService:
    return StubEmbeddingService(dimensions=config.embedding.dimensions)


@pytest.fixture
def ingestor(
    config: LibrarianConfig,
    embedding_service: StubEmbeddingService,
    verifier: StubVerifier,
    repo: StubTomeRepository,
) -> StubIngestor:
    return StubIngestor(config, embedding_service, verifier, repo)


@pytest.fixture
async def ollama_embedding_service() -> AsyncGenerator[OllamaEmbeddingService, None]:
    """Provides a deterministic Ollama-compatible embedding service for tests."""
    settings = EmbeddingSettings(
        provider="ollama",
        model_name="nomic-embed-text",
        dimensions=768,
    )
    service = OllamaEmbeddingService(settings)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": settings.model_name}]})

        if request.url.path == "/api/embed":
            payload = json.loads(request.content.decode("utf-8"))
            text = str(payload.get("input") or "")
            base_vector = np.zeros(settings.dimensions, dtype=np.float32)
            lowered = text.lower()
            if any(token in lowered for token in ("dog", "puppy", "park", "field")):
                base_vector[0] = 1.0
                base_vector[1] = 0.25
            elif any(token in lowered for token in ("quantum", "physics")):
                base_vector[0] = 0.2
                base_vector[1] = 1.0
            else:
                base_vector[0] = 0.4
                base_vector[1] = 0.4

            # Add deterministic text-specific variation while preserving topic clusters.
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            for index, byte in enumerate(digest, start=2):
                if index >= settings.dimensions:
                    break
                base_vector[index] = byte / 255.0 / 50.0

            normalized = base_vector / np.linalg.norm(base_vector)
            return httpx.Response(
                200,
                json={"embeddings": [normalized.astype(float).tolist()]},
            )

        return httpx.Response(404, json={"error": "unexpected path"})

    await service._client.aclose()
    service._client = httpx.AsyncClient(
        base_url=settings.ollama_url,
        timeout=30.0,
        transport=httpx.MockTransport(handler),
    )
    await service.initialize()

    yield service
    await service.aclose()
