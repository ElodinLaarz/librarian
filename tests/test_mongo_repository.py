import asyncio
import logging
import os
import uuid
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from pymongo import MongoClient
from pymongo.errors import OperationFailure, PyMongoError

from src.config import DatabaseSettings
from src.models.enums import SourceType
from src.models.tome import Tome
from src.storage.errors import StorageError
from src.storage.mongo.client import build_motor_client
from src.storage.mongo.mongo_research_job_repository import MongoResearchJobRepository
from src.storage.mongo.mongo_tome_repository import MongoTomeRepository
from tests.stubs import StubEmbeddingService

DEFAULT_TEST_MONGO_URI = "mongodb://localhost:27017/?directConnection=true"
TEST_MONGO_URI = os.environ.get("LIBRARIAN_TEST_MONGO_URI", DEFAULT_TEST_MONGO_URI)


def _mongo_available(uri: str) -> bool:
    """Ping the configured Mongo URI with a short timeout.

    Returns True iff a `ping` command succeeds. Any pymongo / network failure
    is interpreted as "no live Mongo" and triggers a clean skip of the tests
    decorated with `requires_live_mongo`.
    """
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
        "the live-Mongo tests in tests/test_mongo_repository.py."
    ),
)


@pytest.fixture
async def mongo_repo() -> AsyncIterator[MongoTomeRepository]:
    settings = DatabaseSettings(
        uri=TEST_MONGO_URI,
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


@requires_live_mongo
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


@requires_live_mongo
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


@requires_live_mongo
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


# ---------------------------------------------------------------------------
# ensure_indexes failure surfacing
# ---------------------------------------------------------------------------


def _build_repo_with_mocked_collection(
    aggregate_side_effect: object | None = None,
    aggregate_documents: list[dict] | None = None,
    create_search_index_side_effect: object | None = None,
) -> MongoTomeRepository:
    """Construct a MongoTomeRepository whose collection is fully mocked.

    Avoids any network round trip so the test can run without a live mongod.
    """
    settings = DatabaseSettings(
        uri=TEST_MONGO_URI,
        tls=False,
        tls_cert_path="/dev/null",
        database="test_library",
    )
    embedding_service = StubEmbeddingService()
    repo = MongoTomeRepository(settings, embedding_service)

    collection = MagicMock()
    collection.name = "tomes"
    collection.database = MagicMock()
    collection.database.create_collection = AsyncMock(return_value=None)
    collection.create_search_index = AsyncMock(return_value=None)
    if create_search_index_side_effect is not None:
        collection.create_search_index.side_effect = create_search_index_side_effect

    if aggregate_side_effect is not None:

        def _aggregate(_pipeline: object) -> AsyncIterator[dict]:
            raise aggregate_side_effect

        collection.aggregate = _aggregate
    else:
        docs = list(aggregate_documents or [])

        class _AsyncIter:
            def __init__(self, items: list[dict]) -> None:
                self._items = list(items)

            def __aiter__(self) -> "_AsyncIter":
                return self

            async def __anext__(self) -> dict:
                if not self._items:
                    raise StopAsyncIteration
                return self._items.pop(0)

        def _aggregate(_pipeline: object) -> "_AsyncIter":
            return _AsyncIter(docs)

        collection.aggregate = _aggregate

    repo._collection = collection
    return repo


@pytest.mark.asyncio
async def test_ensure_indexes_propagates_operation_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A transient Atlas failure on $listSearchIndexes must surface, not be swallowed."""
    repo = _build_repo_with_mocked_collection(
        aggregate_side_effect=OperationFailure(
            "transient atlas error",
            code=111,
            details={"codeName": "InternalError"},
        ),
    )

    with (
        caplog.at_level(logging.ERROR, logger="src.storage.mongo.mongo_tome_repository"),
        pytest.raises(StorageError) as ei,
    ):
        await repo.ensure_indexes()

    # The original driver error is preserved as the StorageError's cause so
    # operators retain the underlying details when debugging.
    assert isinstance(ei.value.__cause__, OperationFailure)
    assert any(
        record.levelno >= logging.ERROR and "search index" in record.getMessage().lower()
        for record in caplog.records
    ), f"expected ERROR log about search indexes, got: {[r.getMessage() for r in caplog.records]}"


@pytest.mark.asyncio
async def test_ensure_indexes_skips_when_atlas_search_unsupported(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """CommandNotFound on $listSearchIndexes must skip silently (local-mongod path)."""
    repo = _build_repo_with_mocked_collection(
        aggregate_side_effect=OperationFailure(
            "Unrecognized pipeline stage name: '$listSearchIndexes'",
            code=40324,
            details={"codeName": "CommandNotFound"},
        ),
    )

    with caplog.at_level(logging.WARNING, logger="src.storage.mongo.mongo_tome_repository"):
        # Must NOT raise — preserves local-mongod CI fixture path.
        await repo.ensure_indexes()

    assert any(record.levelno == logging.WARNING for record in caplog.records), (
        f"expected WARNING log, got: {[(r.levelname, r.getMessage()) for r in caplog.records]}"
    )
    # And create_search_index must not have been invoked.
    assert repo._collection.create_search_index.await_count == 0


# ---------------------------------------------------------------------------
# Shared Motor client across Mongo repositories (issue #25)
# ---------------------------------------------------------------------------


def test_repos_share_injected_motor_client() -> None:
    """A single Motor client passed into both repos must be reused, not duplicated.

    Network-free: we never call any method on the client, just check identity
    against ``_client``. This is the regression guard for issue #25 — before
    the refactor each repo built its own ``AsyncIOMotorClient`` in ``__init__``.
    """
    settings = DatabaseSettings(
        uri=TEST_MONGO_URI,
        tls=False,
        tls_cert_path="/dev/null",
        database="test_library",
    )
    embedding_service = StubEmbeddingService()
    shared_client = build_motor_client(settings)
    try:
        tome_repo = MongoTomeRepository(settings, embedding_service, client=shared_client)
        job_repo = MongoResearchJobRepository(settings, client=shared_client)
        assert tome_repo._client is shared_client
        assert job_repo._client is shared_client
        assert tome_repo._client is job_repo._client
        # Repos must not own the lifespan-owned client; closing one must NOT
        # tear down the underlying connection pool used by the other.
        assert tome_repo._owns_client is False
        assert job_repo._owns_client is False
    finally:
        shared_client.close()


def test_repo_close_is_noop_for_injected_client() -> None:
    """Closing a repo built with an injected client must NOT close that client.

    The lifespan owns the shared client; if one repo's ``close()`` tore it
    down, the sibling repo would suddenly see a dead connection pool.
    """
    embedding_service = StubEmbeddingService()
    shared_client = MagicMock()
    tome_repo = MongoTomeRepository.__new__(MongoTomeRepository)
    tome_repo._client = shared_client  # type: ignore[attr-defined]
    tome_repo._owns_client = False  # type: ignore[attr-defined]
    tome_repo._embedding_service = embedding_service  # type: ignore[attr-defined]
    tome_repo._collection = MagicMock()  # type: ignore[attr-defined]
    tome_repo.close()
    assert shared_client.close.call_count == 0

    # And the opposite case: a repo that built its own client DOES close it.
    owned_client = MagicMock()
    tome_repo_owned = MongoTomeRepository.__new__(MongoTomeRepository)
    tome_repo_owned._client = owned_client  # type: ignore[attr-defined]
    tome_repo_owned._owns_client = True  # type: ignore[attr-defined]
    tome_repo_owned._embedding_service = embedding_service  # type: ignore[attr-defined]
    tome_repo_owned._collection = MagicMock()  # type: ignore[attr-defined]
    tome_repo_owned.close()
    assert owned_client.close.call_count == 1


def test_build_motor_client_sets_timeout_defaults() -> None:
    """The shared factory must apply the project's connection-timeout defaults.

    These exist so a misconfigured / unreachable Mongo surfaces quickly instead
    of hanging server startup behind the driver's 30-second default.
    """
    settings = DatabaseSettings(
        uri=TEST_MONGO_URI,
        tls=False,
        tls_cert_path="/dev/null",
        database="test_library",
    )
    client = build_motor_client(settings)
    try:
        opts = client.options
        # Motor / pymongo converts ms kwargs into seconds on the ClientOptions
        # and PoolOptions snapshots.
        assert opts.server_selection_timeout == pytest.approx(5.0)
        assert opts.pool_options.connect_timeout == pytest.approx(5.0)
        assert opts.pool_options.socket_timeout == pytest.approx(30.0)
    finally:
        client.close()


@pytest.mark.asyncio
async def test_ensure_indexes_propagates_create_search_index_failure() -> None:
    """A failure inside create_search_index must surface up to the lifespan."""
    repo = _build_repo_with_mocked_collection(
        aggregate_documents=[],  # listSearchIndexes returns empty -> both indexes get created
        create_search_index_side_effect=OperationFailure(
            "permission denied",
            code=13,
            details={"codeName": "Unauthorized"},
        ),
    )

    with pytest.raises(StorageError) as ei:
        await repo.ensure_indexes()
    assert isinstance(ei.value.__cause__, OperationFailure)
