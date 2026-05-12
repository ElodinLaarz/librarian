"""RED test for recency weighting in search results.

This test suite validates that search results can be ranked by recency
using exponential decay formula: recency_score = exp(-ln(2) * age_days / half_life_days)
and blended with RRF scores using: final = rrf * (1 - w) + w * recency
"""

import asyncio
import logging
import os
import uuid
from collections.abc import AsyncIterator
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from pymongo import MongoClient
from pymongo.errors import PyMongoError

from src.config import DatabaseSettings
from src.models.enums import SourceType
from src.models.tome import Tome
from src.storage.mongo.mongo_tome_repository import MongoTomeRepository
from tests.stubs import StubEmbeddingService

DEFAULT_TEST_MONGO_URI = "mongodb://localhost:27017/?directConnection=true"
TEST_MONGO_URI = os.environ.get("LIBRARIAN_TEST_MONGO_URI", DEFAULT_TEST_MONGO_URI)

logger = logging.getLogger(__name__)


def _mongo_available(uri: str) -> bool:
    """Ping the configured Mongo URI with a short timeout."""
    client: MongoClient | None = None
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=500)
        client.admin.command("ping")
        return True
    except (PyMongoError, OSError):
        return False
    finally:
        if client is not None:
            client.close()


requires_live_mongo = pytest.mark.skipif(
    not _mongo_available(TEST_MONGO_URI),
    reason=(
        f"No live MongoDB reachable at {TEST_MONGO_URI!r}. "
        "Set LIBRARIAN_TEST_MONGO_URI to point at a running instance to enable "
        "the live-Mongo tests in tests/test_search_recency.py."
    ),
)


@pytest.fixture
async def mongo_repo() -> AsyncIterator[MongoTomeRepository]:
    settings = DatabaseSettings(
        uri=TEST_MONGO_URI,
        tls=False,
        tls_cert_path="/dev/null",
        database="test_library_recency",
    )
    embedding_service = StubEmbeddingService()
    repo = MongoTomeRepository(settings, embedding_service)

    await repo._collection.delete_many({})

    # Check existing indexes BEFORE ensure_indexes
    existing_search_indexes = []
    try:
        async for index in repo._collection.aggregate([{"$listSearchIndexes": {}}]):
            existing_search_indexes.append(index["name"])
    except Exception:
        pass

    await repo.ensure_indexes()

    if "vectors" not in existing_search_indexes or "default" not in existing_search_indexes:
        logger.info("Indexes being created, waiting 20s...")
        await asyncio.sleep(20)
    else:
        logger.info("Indexes already existed, skipping long wait.")
        await asyncio.sleep(2)

    yield repo

    await repo._collection.delete_many({})
    repo.close()


@requires_live_mongo
@pytest.mark.asyncio
async def test_newer_tome_ranks_higher_with_recency_weight(
    mongo_repo: MongoTomeRepository,
) -> None:
    """
    RED TEST: Verify recency weighting works.

    Create two tomes with identical embeddings and text match (cosine 0.95 equivalent).
    Tome A: created 2 days ago (newer, should rank higher with recency_weight > 0)
    Tome B: created 730 days ago (older, should rank lower with recency_weight > 0)

    With recency_weight=0.5, recency_half_life_days=90, Tome A should rank first.
    """
    now_utc = datetime.now(timezone.utc)
    two_days_ago = now_utc - timedelta(days=2)
    seven_hundred_days_ago = now_utc - timedelta(days=730)

    # Create two tomes with identical embeddings (stub service returns zeros)
    # and identical content to ensure cosine similarity is equal
    search_content = "REST API design patterns best practices"

    tome_a = Tome(
        id=uuid.uuid4(),
        title="REST API Design Guide",
        content=search_content,
        summary="Modern REST API design patterns",
        category="Technology",
        source_type=SourceType.MANUAL,
        tags=["api", "design"],
        embedding=np.array([0.1] * 768, dtype=np.float32),
        confidence=0.95,
        created_at=two_days_ago,
    )

    tome_b = Tome(
        id=uuid.uuid4(),
        title="REST API Design Guide",
        content=search_content,
        summary="Modern REST API design patterns",
        category="Technology",
        source_type=SourceType.MANUAL,
        tags=["api", "design"],
        embedding=np.array([0.1] * 768, dtype=np.float32),
        confidence=0.95,
        created_at=seven_hundred_days_ago,
    )

    # Insert tomes
    await mongo_repo.insert(tome_a)
    await mongo_repo.insert(tome_b)

    # Wait for indexing
    await asyncio.sleep(10)

    # Search WITHOUT recency weighting (should be equal or based on RRF only)
    results_no_recency = await mongo_repo.search(
        query="REST API",
        top_k=2,
        min_confidence=0.5,
    )
    logger.info(
        f"Results without recency weighting: "
        f"[{results_no_recency[0][0].title} ({results_no_recency[0][1]:.4f}), "
        f"{results_no_recency[1][0].title} ({results_no_recency[1][1]:.4f})]"
    )

    # Search WITH recency weighting (newer should rank higher)
    results_with_recency = await mongo_repo.search(
        query="REST API",
        top_k=2,
        min_confidence=0.5,
        recency_weight=0.5,
        recency_half_life_days=90,
    )

    assert len(results_with_recency) >= 2, "Expected at least 2 results"

    # Newer tome (A) should rank higher than older tome (B)
    # (when recency_weight > 0)
    top_result_id = results_with_recency[0][0].id
    top_result_score = results_with_recency[0][1]
    second_result_id = results_with_recency[1][0].id
    second_result_score = results_with_recency[1][1]

    logger.info(
        f"Results with recency_weight=0.5: "
        f"[{results_with_recency[0][0].title} ({top_result_score:.4f}), "
        f"{results_with_recency[1][0].title} ({second_result_score:.4f})]"
    )

    # Assert: newer tome (A) ranks first
    assert top_result_id == tome_a.id, (
        f"Expected newer tome A (2 days old) to rank first with recency weighting, "
        f"but got {results_with_recency[0][0].title} "
        f"(created_at={results_with_recency[0][0].created_at})"
    )

    # Assert: older tome (B) ranks second
    assert second_result_id == tome_b.id, (
        f"Expected older tome B (730 days old) to rank second, "
        f"but got {results_with_recency[1][0].title}"
    )

    # Assert: newer ranks higher (higher score)
    assert top_result_score > second_result_score, (
        f"Expected newer tome to have higher score ({top_result_score:.4f}) "
        f"than older tome ({second_result_score:.4f})"
    )

    logger.info(
        f"✓ Recency test passed: A ({top_result_score:.4f}) > B ({second_result_score:.4f})"
    )
