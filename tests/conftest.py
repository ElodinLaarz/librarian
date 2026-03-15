"""Shared pytest fixtures for the Librarian test suite."""

from __future__ import annotations

import pytest

from src.config import LibrarianConfig
from tests.stubs import StubEmbeddingService, StubIngestor, StubTomeRepository, StubVerifier


@pytest.fixture
def config() -> LibrarianConfig:
    return LibrarianConfig()


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
    verifier: StubVerifier,
    repo: StubTomeRepository,
) -> StubIngestor:
    return StubIngestor(config, verifier, repo)
