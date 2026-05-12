"""Shared pytest fixtures for the Librarian test suite."""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from typing import Any

import pytest

from src.config import DatabaseSettings, EmbeddingSettings, IngestSettings, LibrarianConfig

# Disable automatic .env loading during tests to prevent leakage from the repo root
os.environ["LIBRARIAN_SKIP_DOTENV"] = "1"
from src.services.embedding import OllamaEmbeddingService
from tests.stubs import StubEmbeddingService, StubIngestor, StubTomeRepository, StubVerifier

_TEST_DB_SETTINGS = DatabaseSettings(
    uri="mongodb://localhost:27017/?directConnection=true",
    tls=False,
)


def make_test_config(**overrides: Any) -> LibrarianConfig:
    """Build a LibrarianConfig with dummy database credentials for tests.

    Tests use min_shard_chars=0 to allow arbitrary content lengths without
    enforcing the 400-char production minimum.
    """
    overrides.setdefault("database", _TEST_DB_SETTINGS)
    overrides.setdefault("ingest", IngestSettings(min_shard_chars=0, min_shard_words=0))
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
    """Provides a real Ollama embedding service if available locally, else skips."""
    settings = EmbeddingSettings(
        provider="ollama",
        model_name="nomic-embed-text",
        dimensions=768,
    )
    service = OllamaEmbeddingService(settings)
    try:
        await service.initialize()
    except Exception as exc:
        pytest.skip(f"Ollama is unavailable locally {exc}")

    yield service
    await service.aclose()
