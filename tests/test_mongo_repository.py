import asyncio
import os
import uuid
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
import pymongo.errors

from src.config import DatabaseSettings
from src.models.enums import SourceType
from src.models.tome import Tome
from src.storage.mongo.mongo_tome_repository import MongoTomeRepository
from tests.stubs import StubEmbeddingService

# Skip all MongoDB tests if no live Mongo is reachable
_MONGO_URI = os.environ.get(
    "LIBRARIAN_TEST_MONGO_URI",
    "mongodb://localhost:27017/?directConnection=true",
)


def _mongo_is_reachable() -> bool:
    """Check if a MongoDB instance is reachable with a short timeout."""
    try:
        from pymongo import MongoClient

        client = MongoClient(_MONGO_URI, serverSelectionTimeoutMS=500)
        client.admin.command("ping")
        client.close()
        return True
    except (pymongo.errors.ServerSelectionTimeoutError, pymongo.errors.ConnectionFailure):
        return False
    except Exception:
        return False


mongo_available = _mongo_is_reachable()
pytestmark = pytest.mark.skipif(
    not mongo_available,
    reason=f"MongoDB not reachable at {_MONGO_URI}",
)


@pytest.fixture
async def mongo_repo() -> AsyncIterator[MongoTomeRepository]:
    settings = DatabaseSettings(
        uri=_MONGO_URI,
        database="test_librarian",
        tomes_collection="test_tomes",
    )
    embedding_service = StubEmbeddingService(dimensions=768)
    repo = MongoTomeRepository(settings, embedding_service)
    # Clean up any leftover test data, then drop collection after tests
    try:
        await repo._collection.drop()
    except Exception:
        pass
    yield repo
    try:
        await repo._collection.drop()
    except Exception:
        pass


@pytest.mark.asyncio
async def test_insert_and_get(mongo_repo: MongoTomeRepository) -> None:
    tome = Tome(
        id=uuid.uuid4(),
        title="Test Tome",
        content="Test Content",
        summary="Test Summary",
        category="test",
        source_type=SourceType.MANUAL,
        tags=["test"],
        embedding=np.array([0.1] * 768, dtype=np.float32),
        confidence=0.9,
    )
    inserted_id = await mongo_repo.insert(tome)
    assert inserted_id == tome.id

    retrieved = await mongo_repo.get_by_id(tome.id)
    assert retrieved is not None
    assert retrieved.id == tome.id
    assert retrieved.title == tome.title


@pytest.mark.asyncio
async def test_find_near_duplicates(mongo_repo: MongoTomeRepository) -> None:
    random_embedding = np.random.rand(768).astype(np.float32)
    tome1 = Tome(
        id=uuid.uuid4(),
        title="Doc 1",
        content="Content 1",
        summary="Summary 1",
        category="test",
        source_type=SourceType.MANUAL,
        tags=["tag"],
        embedding=random_embedding,
        confidence=0.9,
    )
    tome2 = Tome(
        id=uuid.uuid4(),
        title="Doc 2",
        content="Content 2",
        summary="Summary 2",
        category="test",
        source_type=SourceType.MANUAL,
        tags=["tag"],
        embedding=random_embedding,  # Identical embedding
        confidence=0.9,
    )

    await mongo_repo.insert(tome1)
    await mongo_repo.insert(tome2)

    # Wait for indexing
    await asyncio.sleep(10)

    duplicates = await mongo_repo.find_near_duplicates(tome1, threshold=0.9)
    assert len(duplicates) == 1
    assert duplicates[0].id == tome2.id


@pytest.mark.asyncio
async def test_search(mongo_repo: MongoTomeRepository) -> None:
    tome = Tome(
        id=uuid.uuid4(),
        title="Unique Title",
        content="This is unique content about quantum ducks.",
        summary="Summary",
        category="test",
        source_type=SourceType.MANUAL,
        tags=["quantum"],
        embedding=np.array([0.2] * 768, dtype=np.float32),
        confidence=0.9,
    )
    await mongo_repo.insert(tome)

    # Mock embedding service to return matching vector for query
    with patch.object(
        mongo_repo._embedding_service,
        "embed",
        AsyncMock(return_value=np.array([0.2] * 768, dtype=np.float32)),
    ):
        # Wait for indexing
        await asyncio.sleep(10)

        results = await mongo_repo.search("quantum ducks", top_k=5)
        assert len(results) > 0
        assert results[0][0].id == tome.id