import asyncio
from collections.abc import AsyncIterator
import uuid
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from src.config import DatabaseSettings
from src.models.enums import SourceType
from src.models.tome import Tome
from src.storage.mongo.mongo_tome_repository import MongoTomeRepository
from tests.stubs import StubEmbeddingService


@pytest.fixture
async def mongo_repo() -> AsyncIterator[MongoTomeRepository]:
    settings = DatabaseSettings(
        uri="mongodb://localhost:27017/?directConnection=true",
        tls=False,
        tls_cert_path="/dev/null",
        database="test_library",
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
        print("DEBUG: Indexes being created, waiting 20s...")
        await asyncio.sleep(20)
    else:
        print("DEBUG: Indexes already existed, skipping long wait.")
        await asyncio.sleep(2)

    yield repo

    await repo._collection.delete_many({})
    repo.close()


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
