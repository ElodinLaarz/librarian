"""Tests for FsTomeRepository — embedding persistence, category filter, cosine dedup."""

from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np
import pytest

from src.config import DatabaseSettings
from src.models.enums import SourceType
from src.models.tome import Tome
from src.storage.filesystem.fs_tome_repository import FsTomeRepository, _cosine_similarity

# ── helpers ──────────────────────────────────────────────────────────────────

RNG = np.random.default_rng(12345)


def _make_tome(
    *,
    category: str = "general",
    confidence: float = 0.8,
    embedding: np.ndarray | None = None,
) -> Tome:
    if embedding is None:
        embedding = RNG.random(8).astype(np.float32)
        embedding /= np.linalg.norm(embedding)
    return Tome(
        id=uuid.uuid4(),
        title="Test Tome",
        content="Some content.",
        summary="A summary.",
        category=category,
        tags=[],
        source_url=None,
        source_type=SourceType.AGENT_INPUT,
        confidence=confidence,
        embedding=embedding,
    )


@pytest.fixture
def repo(tmp_path: Path) -> FsTomeRepository:
    settings = DatabaseSettings(uri=str(tmp_path), tomes_collection="tomes", tls_cert_path="")
    return FsTomeRepository(settings)


# ── embedding round-trip ─────────────────────────────────────────────────────


async def test_embedding_survives_disk_roundtrip(repo: FsTomeRepository) -> None:
    original = _make_tome()
    await repo.insert(original)

    loaded = await repo.get_by_id(original.id)
    assert loaded is not None
    assert loaded.embedding is not None
    assert original.embedding is not None
    np.testing.assert_array_almost_equal(loaded.embedding, original.embedding, decimal=5)


async def test_embedding_in_search_results(repo: FsTomeRepository) -> None:
    original = _make_tome()
    await repo.insert(original)

    results = await repo.search(query="anything")
    assert len(results) == 1
    tome, score = results[0]
    assert tome.embedding is not None
    assert original.embedding is not None
    np.testing.assert_array_almost_equal(tome.embedding, original.embedding, decimal=5)
    assert score == 1.0


# ── category filter ───────────────────────────────────────────────────────────


async def test_search_returns_all_when_no_category_filter(repo: FsTomeRepository) -> None:
    await repo.insert(_make_tome(category="science"))
    await repo.insert(_make_tome(category="history"))

    results = await repo.search(query="anything")
    assert len(results) == 2


async def test_search_filters_by_category(repo: FsTomeRepository) -> None:
    await repo.insert(_make_tome(category="science"))
    await repo.insert(_make_tome(category="history"))

    results = await repo.search(query="anything", category="science")
    assert len(results) == 1
    tome, score = results[0]
    assert tome.category == "science"


async def test_search_category_no_match_returns_empty(repo: FsTomeRepository) -> None:
    await repo.insert(_make_tome(category="science"))

    results = await repo.search(query="anything", category="philosophy")
    assert results == []


# ── cosine similarity ─────────────────────────────────────────────────────────


def test_cosine_similarity_identical_vectors() -> None:
    v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    assert _cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors() -> None:
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    assert _cosine_similarity(a, b) == pytest.approx(0.0)


def test_cosine_similarity_zero_vector_returns_zero() -> None:
    zero = np.zeros(4, dtype=np.float32)
    v = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    assert _cosine_similarity(zero, v) == 0.0


async def test_find_near_duplicates_detects_similar_tomes(repo: FsTomeRepository) -> None:
    base = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    similar = np.array([0.999, 0.045, 0.0, 0.0], dtype=np.float32)
    similar /= np.linalg.norm(similar)

    existing = _make_tome(embedding=base)
    await repo.insert(existing)

    candidate = _make_tome(embedding=similar)
    duplicates = await repo.find_near_duplicates(candidate)

    assert len(duplicates) == 1
    assert duplicates[0].id == existing.id


async def test_find_near_duplicates_ignores_dissimilar_tomes(repo: FsTomeRepository) -> None:
    a = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

    existing = _make_tome(embedding=a)
    await repo.insert(existing)

    candidate = _make_tome(embedding=b)
    duplicates = await repo.find_near_duplicates(candidate)

    assert duplicates == []


async def test_find_near_duplicates_excludes_self(repo: FsTomeRepository) -> None:
    v = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    tome = _make_tome(embedding=v)
    await repo.insert(tome)

    duplicates = await repo.find_near_duplicates(tome)
    assert duplicates == []
