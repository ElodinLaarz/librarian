"""Shared pytest fixtures for the Librarian test suite."""

from __future__ import annotations

import pytest

from src.config import DatabaseSettings, LibrarianConfig
from tests.stubs import StubEmbeddingService, StubIngestor, StubTomeRepository, StubVerifier

_TEST_DB_SETTINGS = DatabaseSettings(uri="mongodb://localhost:27017", tls_cert_path="/dev/null")


def make_test_config(**overrides) -> LibrarianConfig:
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
def embedding_service() -> StubEmbeddingService:
    return StubEmbeddingService()


@pytest.fixture
def ingestor(
    config: LibrarianConfig,
    embedding_service: StubEmbeddingService,
    verifier: StubVerifier,
    repo: StubTomeRepository,
) -> StubIngestor:
    return StubIngestor(config, embedding_service, verifier, repo)
