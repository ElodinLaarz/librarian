"""Tests for FsTomeRepository — embedding persistence, category filter, cosine dedup."""

from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from src.config import DatabaseSettings, EmbeddingSettings, TidySettings
from src.models.enums import SourceType
from src.models.tome import Tome
from src.services.embedding import DummyEmbeddingService, OllamaEmbeddingService
from src.storage.filesystem.fs_tome_repository import FsTomeRepository, _cosine_similarity

# ── helpers ──────────────────────────────────────────────────────────────────

RNG = np.random.default_rng(12345)


def _make_tome(
    *,
    category: str = "general",
    title: str = "Test Tome",
    confidence: float = 0.8,
    embedding: NDArray[np.float32] | None = None,
) -> Tome:
    if embedding is None:
        embedding = RNG.random(8).astype(np.float32)
        embedding /= np.linalg.norm(embedding)
    return Tome(
        id=uuid.uuid4(),
        title=title,
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
def embedding_service() -> DummyEmbeddingService:
    return DummyEmbeddingService(EmbeddingSettings(dimensions=8))


@pytest.fixture
def repo(tmp_path: Path, embedding_service: DummyEmbeddingService) -> FsTomeRepository:
    settings = DatabaseSettings(
        uri=str(tmp_path),
        tomes_collection="tomes",
        tls_cert_path="",
        tls=False,
    )
    tidy_settings = TidySettings(
        threshold=0.95,
        max_fact_frequency=3,
        min_shared_facts=2,
        min_fact_overlap=0.66,
    )
    return FsTomeRepository(settings, embedding_service, tidy_settings)


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

    results = await repo.search(query="anything", min_confidence=0.0)
    assert len(results) == 1
    tome, score = results[0]
    assert tome.embedding is not None
    assert original.embedding is not None
    np.testing.assert_array_almost_equal(tome.embedding, original.embedding, decimal=5)
    # DummyEmbeddingService returns zero vectors → cosine similarity is 0.0
    assert score == pytest.approx(0.0)


# ── category filter ───────────────────────────────────────────────────────────


async def test_search_returns_all_when_no_category_filter(repo: FsTomeRepository) -> None:
    await repo.insert(_make_tome(category="science"))
    await repo.insert(_make_tome(category="history"))

    results = await repo.search(query="anything", min_confidence=0.0)
    assert len(results) == 2


async def test_search_filters_by_category(repo: FsTomeRepository) -> None:
    await repo.insert(_make_tome(category="science"))
    await repo.insert(_make_tome(category="history"))

    results = await repo.search(query="anything", category="science", min_confidence=0.0)
    assert len(results) == 1
    tome, score = results[0]
    assert tome.category == "science"


async def test_search_category_no_match_returns_empty(repo: FsTomeRepository) -> None:
    await repo.insert(_make_tome(category="science"))

    results = await repo.search(query="anything", category="philosophy", min_confidence=0.0)
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


@pytest.fixture
def repo_ollama(
    tmp_path: Path, ollama_embedding_service: OllamaEmbeddingService
) -> FsTomeRepository:
    settings = DatabaseSettings(
        uri=str(tmp_path),
        tomes_collection="tomes",
        tls_cert_path="",
        tls=False,
    )
    return FsTomeRepository(settings, ollama_embedding_service, TidySettings())


async def test_search_real_embeddings(
    repo_ollama: FsTomeRepository, ollama_embedding_service: OllamaEmbeddingService
) -> None:
    embed_dog = await ollama_embedding_service.embed("The dog ran through the park")
    embed_physics = await ollama_embedding_service.embed("Quantum physics is hard")

    await repo_ollama.insert(_make_tome(title="Dog", embedding=embed_dog))
    await repo_ollama.insert(_make_tome(title="Physics", embedding=embed_physics))

    results = await repo_ollama.search(query="a puppy played outside", min_confidence=0.0)
    assert len(results) == 2

    tome0, score0 = results[0]
    tome1, score1 = results[1]

    # Dog should be more similar to "puppy played outside" than Physics
    assert tome0.title == "Dog"
    assert tome1.title == "Physics"
    assert score0 > score1
    assert score0 > 0.0


async def test_find_all_near_duplicates_detects_exact_content_groups(
    repo: FsTomeRepository,
) -> None:
    t1 = _make_tome(embedding=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    t1.content = "Shared fact"
    t2 = _make_tome(embedding=np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32))
    t2.content = "Shared fact"
    await repo.insert(t1)
    await repo.insert(t2)

    result = await repo.find_all_near_duplicates()

    assert result.scanned == 2
    assert len(result.groups) == 1
    assert {tome.id for tome in result.groups[0]} == {t1.id, t2.id}


async def test_find_all_near_duplicates_detects_fact_overlap_groups(repo: FsTomeRepository) -> None:
    content_a = "Alpha fact\n\nBeta fact\n\nGamma fact"
    content_b = "Alpha fact\n\nBeta fact\n\nDelta fact"
    t1 = _make_tome(embedding=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    t2 = _make_tome(embedding=np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32))
    t1.content = content_a
    t2.content = content_b
    await repo.insert(t1)
    await repo.insert(t2)

    result = await repo.find_all_near_duplicates()

    assert len(result.groups) == 1
    assert {tome.id for tome in result.groups[0]} == {t1.id, t2.id}


async def test_find_all_near_duplicates_ignores_boilerplate_facts(repo: FsTomeRepository) -> None:
    boilerplate = "Common disclaimer"
    embeddings = [
        np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    ]
    for index in range(4):
        tome = _make_tome(embedding=embeddings[index])
        tome.content = f"{boilerplate}\n\nUnique fact {index}"
        await repo.insert(tome)

    result = await repo.find_all_near_duplicates()

    assert result.groups == []
    assert result.ignored_high_frequency_facts == 1


async def test_find_all_near_duplicates_detects_semantic_only_groups(
    repo: FsTomeRepository,
) -> None:
    a = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([0.999, 0.045, 0.0, 0.0], dtype=np.float32)
    b /= np.linalg.norm(b)
    t1 = _make_tome(embedding=a)
    t2 = _make_tome(embedding=b)
    t1.content = "Cats chase lasers."
    t2.content = "Felines sprint after lights."
    await repo.insert(t1)
    await repo.insert(t2)

    result = await repo.find_all_near_duplicates()

    assert len(result.groups) == 1
    assert {tome.id for tome in result.groups[0]} == {t1.id, t2.id}
