"""Storage exception taxonomy — driver errors are wrapped in StorageError.

Validates that every repository method translates backend-specific failures
(``pymongo.errors.*``, ``OSError``) into the public :mod:`src.storage.errors`
hierarchy so service-layer code can stay backend-agnostic.

Also structurally inspects ``Ingestor`` and ``Researcher`` to confirm they
catch ``StorageError`` rather than bare ``Exception`` around storage calls.
"""

from __future__ import annotations

import inspect
import uuid
from pathlib import Path
from typing import Any
from unittest import mock
from unittest.mock import AsyncMock

import numpy as np
import pytest
from numpy.typing import NDArray
from pydantic import ValidationError
from pymongo.errors import (
    ConnectionFailure,
    DuplicateKeyError,
    ServerSelectionTimeoutError,
)

from src.config import DatabaseSettings, EmbeddingSettings, TidySettings
from src.models.enums import SourceType
from src.models.research_job import ResearchJob
from src.models.tome import Tome
from src.services.embedding import DummyEmbeddingService
from src.services.ingestor import Ingestor
from src.services.researcher import Researcher
from src.storage.errors import (
    BackendUnavailableError,
    DuplicateError,
    NotFoundError,
    StorageError,
)
from src.storage.filesystem.fs_research_job_repository import FsResearchJobRepository
from src.storage.filesystem.fs_tome_repository import FsTomeRepository
from src.storage.mongo.mongo_research_job_repository import MongoResearchJobRepository
from src.storage.mongo.mongo_tome_repository import MongoTomeRepository

# ── helpers ──────────────────────────────────────────────────────────────────


def _make_tome(embedding: NDArray[np.floating[Any]] | None = None) -> Tome:
    if embedding is None:
        embedding = np.zeros(8, dtype=np.float32)
    return Tome(
        id=uuid.uuid4(),
        title="t",
        content="c",
        summary="s",
        category="general",
        tags=[],
        source_url=None,
        source_type=SourceType.AGENT_INPUT,
        confidence=0.8,
        embedding=embedding,
    )


@pytest.fixture
def fs_tome_repo(tmp_path: Path) -> FsTomeRepository:
    settings = DatabaseSettings(uri=str(tmp_path), tomes_collection="tomes")
    return FsTomeRepository(
        settings,
        DummyEmbeddingService(EmbeddingSettings(dimensions=8)),
        TidySettings(),
    )


@pytest.fixture
def fs_job_repo(tmp_path: Path) -> FsResearchJobRepository:
    return FsResearchJobRepository(DatabaseSettings(uri=str(tmp_path)))


@pytest.fixture
def mongo_tome_repo() -> MongoTomeRepository:
    """A MongoTomeRepository with the live driver replaced by an in-test mock.

    We bypass ``__init__`` entirely so we never touch the real Mongo client;
    only the attributes the methods read are set up.
    """
    repo = MongoTomeRepository.__new__(MongoTomeRepository)
    repo._collection = mock.MagicMock()  # type: ignore[attr-defined]
    repo._embedding_service = DummyEmbeddingService(EmbeddingSettings(dimensions=8))  # type: ignore[attr-defined]
    repo._tidy_settings = TidySettings()  # type: ignore[attr-defined]
    repo._client = mock.MagicMock()  # type: ignore[attr-defined]
    return repo


@pytest.fixture
def mongo_job_repo() -> MongoResearchJobRepository:
    repo = MongoResearchJobRepository.__new__(MongoResearchJobRepository)
    repo._collection = mock.MagicMock()  # type: ignore[attr-defined]
    repo._client = mock.MagicMock()  # type: ignore[attr-defined]
    return repo


# ── error hierarchy ──────────────────────────────────────────────────────────


def test_error_hierarchy_subclasses_storage_error() -> None:
    assert issubclass(NotFoundError, StorageError)
    assert issubclass(DuplicateError, StorageError)
    assert issubclass(BackendUnavailableError, StorageError)


# ── Mongo tome repository ────────────────────────────────────────────────────


async def test_mongo_insert_duplicate_raises_duplicate_error(
    mongo_tome_repo: MongoTomeRepository,
) -> None:
    mongo_tome_repo._collection.insert_one = AsyncMock(side_effect=DuplicateKeyError("dup"))
    with pytest.raises(DuplicateError) as ei:
        await mongo_tome_repo.insert(_make_tome())
    assert isinstance(ei.value.__cause__, DuplicateKeyError)


async def test_mongo_insert_timeout_raises_backend_unavailable(
    mongo_tome_repo: MongoTomeRepository,
) -> None:
    mongo_tome_repo._collection.insert_one = AsyncMock(
        side_effect=ServerSelectionTimeoutError("timeout")
    )
    with pytest.raises(BackendUnavailableError):
        await mongo_tome_repo.insert(_make_tome())


async def test_mongo_insert_connection_failure_raises_backend_unavailable(
    mongo_tome_repo: MongoTomeRepository,
) -> None:
    mongo_tome_repo._collection.insert_one = AsyncMock(side_effect=ConnectionFailure("down"))
    with pytest.raises(BackendUnavailableError):
        await mongo_tome_repo.insert(_make_tome())


async def test_mongo_delete_pymongo_error_raises_storage_error(
    mongo_tome_repo: MongoTomeRepository,
) -> None:
    mongo_tome_repo._collection.delete_one = AsyncMock(
        side_effect=ServerSelectionTimeoutError("down")
    )
    with pytest.raises(BackendUnavailableError):
        await mongo_tome_repo.delete(uuid.uuid4())


async def test_mongo_get_by_id_pymongo_error_raises_storage_error(
    mongo_tome_repo: MongoTomeRepository,
) -> None:
    mongo_tome_repo._collection.find_one = AsyncMock(side_effect=ConnectionFailure("oops"))
    with pytest.raises(BackendUnavailableError):
        await mongo_tome_repo.get_by_id(uuid.uuid4())


class _RaisingAsyncIter:
    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    def __aiter__(self) -> _RaisingAsyncIter:
        return self

    async def __anext__(self) -> Any:
        raise self._exc


async def test_mongo_search_pymongo_error_raises_storage_error(
    mongo_tome_repo: MongoTomeRepository,
) -> None:
    mongo_tome_repo._collection.aggregate = mock.MagicMock(
        return_value=_RaisingAsyncIter(ConnectionFailure("down"))
    )
    with pytest.raises(BackendUnavailableError):
        await mongo_tome_repo.search("anything")


async def test_mongo_find_near_duplicates_pymongo_error_raises_storage_error(
    mongo_tome_repo: MongoTomeRepository,
) -> None:
    mongo_tome_repo._collection.aggregate = mock.MagicMock(
        return_value=_RaisingAsyncIter(ServerSelectionTimeoutError("down"))
    )
    tome = _make_tome(embedding=np.array([1.0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32))
    with pytest.raises(BackendUnavailableError):
        await mongo_tome_repo.find_near_duplicates(tome)


async def test_mongo_ensure_indexes_connection_failure_raises_backend_unavailable(
    mongo_tome_repo: MongoTomeRepository,
) -> None:
    db = mock.MagicMock()
    db.create_collection = AsyncMock(side_effect=ServerSelectionTimeoutError("down"))
    mongo_tome_repo._collection.database = db
    mongo_tome_repo._collection.name = "tomes"
    with pytest.raises(BackendUnavailableError):
        await mongo_tome_repo.ensure_indexes()


# ── Mongo research-job repository ────────────────────────────────────────────


async def test_mongo_job_insert_duplicate_raises_duplicate_error(
    mongo_job_repo: MongoResearchJobRepository,
) -> None:
    mongo_job_repo._collection.insert_one = AsyncMock(side_effect=DuplicateKeyError("dup"))
    with pytest.raises(DuplicateError):
        await mongo_job_repo.insert(ResearchJob(id=uuid.uuid4(), topic="t"))


async def test_mongo_job_update_missing_raises_not_found(
    mongo_job_repo: MongoResearchJobRepository,
) -> None:
    result = mock.MagicMock()
    result.matched_count = 0
    mongo_job_repo._collection.update_one = AsyncMock(return_value=result)
    with pytest.raises(NotFoundError):
        await mongo_job_repo.update(ResearchJob(id=uuid.uuid4(), topic="t"))


async def test_mongo_job_get_by_id_connection_failure_raises_backend_unavailable(
    mongo_job_repo: MongoResearchJobRepository,
) -> None:
    mongo_job_repo._collection.find_one = AsyncMock(side_effect=ServerSelectionTimeoutError("down"))
    with pytest.raises(BackendUnavailableError):
        await mongo_job_repo.get_by_id(uuid.uuid4())


# ── filesystem tome repository ───────────────────────────────────────────────


async def test_fs_insert_oserror_raises_storage_error(
    fs_tome_repo: FsTomeRepository,
) -> None:
    with (
        mock.patch.object(Path, "write_text", side_effect=OSError("ENOSPC")),
        pytest.raises(StorageError),
    ):
        await fs_tome_repo.insert(_make_tome())


async def test_fs_insert_filenotfound_raises_backend_unavailable(
    fs_tome_repo: FsTomeRepository,
) -> None:
    with (
        mock.patch.object(Path, "write_text", side_effect=FileNotFoundError("missing root")),
        pytest.raises(BackendUnavailableError),
    ):
        await fs_tome_repo.insert(_make_tome())


async def test_fs_delete_oserror_raises_storage_error(
    fs_tome_repo: FsTomeRepository,
) -> None:
    tome = _make_tome()
    await fs_tome_repo.insert(tome)
    with (
        mock.patch.object(Path, "unlink", side_effect=PermissionError("denied")),
        pytest.raises(BackendUnavailableError),
    ):
        await fs_tome_repo.delete(tome.id)


async def test_fs_get_by_id_permission_error_raises_storage_error(
    fs_tome_repo: FsTomeRepository,
) -> None:
    tome = _make_tome()
    await fs_tome_repo.insert(tome)
    target = fs_tome_repo._get_path(tome.id)

    real_read_text = Path.read_text

    def selective(self: Path, *args: Any, **kwargs: Any) -> str:
        if self.resolve() == target.resolve():
            raise PermissionError("denied")
        return real_read_text(self, *args, **kwargs)

    with mock.patch.object(Path, "read_text", selective), pytest.raises(StorageError):
        await fs_tome_repo.get_by_id(tome.id)


async def test_fs_search_oserror_raises_storage_error(
    fs_tome_repo: FsTomeRepository,
) -> None:
    with (
        mock.patch.object(
            FsTomeRepository,
            "_read_all_tomes_sync",
            side_effect=PermissionError("denied"),
        ),
        pytest.raises(BackendUnavailableError),
    ):
        await fs_tome_repo.search("anything")


async def test_fs_find_near_duplicates_oserror_raises_storage_error(
    fs_tome_repo: FsTomeRepository,
) -> None:
    tome = _make_tome(embedding=np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32))
    with (
        mock.patch.object(
            FsTomeRepository,
            "_read_all_tomes_sync",
            side_effect=OSError("EIO"),
        ),
        pytest.raises(StorageError),
    ):
        await fs_tome_repo.find_near_duplicates(tome)


def test_fs_repo_init_missing_root_raises_backend_unavailable(tmp_path: Path) -> None:
    settings = DatabaseSettings(uri=str(tmp_path), tomes_collection="tomes")
    with (
        mock.patch.object(Path, "mkdir", side_effect=FileNotFoundError("missing")),
        pytest.raises(BackendUnavailableError),
    ):
        FsTomeRepository(
            settings,
            DummyEmbeddingService(EmbeddingSettings(dimensions=8)),
            TidySettings(),
        )


# ── filesystem research-job repository ───────────────────────────────────────


async def test_fs_job_update_missing_raises_not_found(
    fs_job_repo: FsResearchJobRepository,
) -> None:
    with pytest.raises(NotFoundError):
        await fs_job_repo.update(ResearchJob(id=uuid.uuid4(), topic="t"))


async def test_fs_job_insert_filenotfound_raises_backend_unavailable(
    fs_job_repo: FsResearchJobRepository,
) -> None:
    job = ResearchJob(id=uuid.uuid4(), topic="t")
    with (
        mock.patch.object(Path, "write_text", side_effect=FileNotFoundError("missing")),
        pytest.raises(BackendUnavailableError),
    ):
        await fs_job_repo.insert(job)


async def test_fs_job_insert_oserror_raises_storage_error(
    fs_job_repo: FsResearchJobRepository,
) -> None:
    job = ResearchJob(id=uuid.uuid4(), topic="t")
    with (
        mock.patch.object(Path, "write_text", side_effect=OSError("EIO")),
        pytest.raises(StorageError),
    ):
        await fs_job_repo.insert(job)


# ── query-embedding failures surface as StorageError ─────────────────────────


class _ExplodingEmbeddingService(DummyEmbeddingService):
    """Embedding service whose embed() always fails (provider outage)."""

    async def embed(self, text: str) -> NDArray[np.float32]:
        raise RuntimeError("embedding provider down")


async def test_fs_search_embed_failure_raises_storage_error(tmp_path: Path) -> None:
    """FS search must not silently degrade to 0.0 scores when embedding fails."""
    settings = DatabaseSettings(uri=str(tmp_path), tomes_collection="tomes")
    repo = FsTomeRepository(
        settings,
        _ExplodingEmbeddingService(EmbeddingSettings(dimensions=8)),
        TidySettings(),
    )
    await repo.insert(_make_tome())
    with pytest.raises(StorageError) as ei:
        await repo.search("anything", min_confidence=0.0)
    assert isinstance(ei.value.__cause__, RuntimeError)


async def test_mongo_search_embed_failure_raises_storage_error(
    mongo_tome_repo: MongoTomeRepository,
) -> None:
    """Mongo search must wrap embedding failures instead of leaking them raw."""
    mongo_tome_repo._embedding_service = _ExplodingEmbeddingService(EmbeddingSettings(dimensions=8))
    with pytest.raises(StorageError) as ei:
        await mongo_tome_repo.search("anything")
    assert isinstance(ei.value.__cause__, RuntimeError)


# ── corrupt documents surface as StorageError, not ValidationError ───────────


async def test_mongo_get_by_id_corrupt_document_raises_storage_error(
    mongo_tome_repo: MongoTomeRepository,
) -> None:
    """A schema-invalid Mongo document must not leak pydantic.ValidationError."""
    mongo_tome_repo._collection.find_one = AsyncMock(
        return_value={"_id": "not-a-uuid", "title": 42}
    )
    with pytest.raises(StorageError) as ei:
        await mongo_tome_repo.get_by_id(uuid.uuid4())
    assert isinstance(ei.value.__cause__, ValidationError)


async def test_mongo_job_get_by_id_corrupt_document_raises_storage_error(
    mongo_job_repo: MongoResearchJobRepository,
) -> None:
    mongo_job_repo._collection.find_one = AsyncMock(return_value={"_id": "garbage"})
    with pytest.raises(StorageError) as ei:
        await mongo_job_repo.get_by_id(uuid.uuid4())
    assert isinstance(ei.value.__cause__, ValidationError)


# ── service-layer structural assertions ──────────────────────────────────────


def test_ingestor_catches_storage_error() -> None:
    """Service code must catch StorageError (not bare Exception) around storage."""
    source = inspect.getsource(Ingestor)
    assert "except StorageError" in source, "Ingestor must catch StorageError around storage calls"


def test_researcher_catches_storage_error() -> None:
    source = inspect.getsource(Researcher)
    assert "except StorageError" in source, (
        "Researcher must catch StorageError around storage calls"
    )


def test_no_pymongo_in_services() -> None:
    """Services must not import or reference pymongo or OSError as storage handlers."""
    import src.services.ingestor as ingestor_mod
    import src.services.researcher as researcher_mod

    for mod in (ingestor_mod, researcher_mod):
        src = inspect.getsource(mod)
        assert "pymongo" not in src, f"{mod.__name__} leaks pymongo"
        assert "OSError" not in src, f"{mod.__name__} leaks OSError"


# ── backfill: update / mark_superseded / list_all / count error paths ─────────


async def test_mongo_update_returns_updated_tome(
    mongo_tome_repo: MongoTomeRepository,
) -> None:
    """Happy path for the previously-untested Mongo update()."""
    from src.storage.mongo.mongo_tome import MongoTome

    tome = _make_tome()
    doc = MongoTome.from_tome(tome).model_dump(by_alias=True)
    doc["category"] = "updated-category"
    mongo_tome_repo._collection.find_one_and_update = AsyncMock(return_value=doc)

    updated = await mongo_tome_repo.update(tome.id, category="updated-category")

    assert updated is not None
    assert updated.category == "updated-category"
    call = mongo_tome_repo._collection.find_one_and_update.await_args
    assert call.args[0] == {"_id": tome.id}
    assert call.args[1] == {"$set": {"category": "updated-category"}}


async def test_mongo_update_unknown_id_returns_none(
    mongo_tome_repo: MongoTomeRepository,
) -> None:
    mongo_tome_repo._collection.find_one_and_update = AsyncMock(return_value=None)
    assert await mongo_tome_repo.update(uuid.uuid4(), category="x") is None


async def test_mongo_update_pymongo_error_raises_storage_error(
    mongo_tome_repo: MongoTomeRepository,
) -> None:
    mongo_tome_repo._collection.find_one_and_update = AsyncMock(
        side_effect=ConnectionFailure("down")
    )
    with pytest.raises(BackendUnavailableError):
        await mongo_tome_repo.update(uuid.uuid4(), category="x")


async def test_mongo_mark_superseded_pymongo_error_raises_storage_error(
    mongo_tome_repo: MongoTomeRepository,
) -> None:
    mongo_tome_repo._collection.update_one = AsyncMock(
        side_effect=ServerSelectionTimeoutError("down")
    )
    with pytest.raises(BackendUnavailableError):
        await mongo_tome_repo.mark_superseded(uuid.uuid4(), uuid.uuid4())


async def test_mongo_list_all_pymongo_error_raises_storage_error(
    mongo_tome_repo: MongoTomeRepository,
) -> None:
    cursor = mock.MagicMock()
    cursor.sort.return_value.skip.return_value.limit.return_value = _RaisingAsyncIter(
        ConnectionFailure("down")
    )
    mongo_tome_repo._collection.find = mock.MagicMock(return_value=cursor)
    with pytest.raises(BackendUnavailableError):
        await mongo_tome_repo.list_all()


async def test_mongo_count_pymongo_error_raises_storage_error(
    mongo_tome_repo: MongoTomeRepository,
) -> None:
    mongo_tome_repo._collection.count_documents = AsyncMock(
        side_effect=ServerSelectionTimeoutError("down")
    )
    with pytest.raises(BackendUnavailableError):
        await mongo_tome_repo.count()


async def test_fs_update_write_failure_raises_storage_error(
    fs_tome_repo: FsTomeRepository,
) -> None:
    """A generic OSError exercises ``_wrap_os``'s fallback branch: plain
    ``StorageError``, not ``BackendUnavailableError`` (reserved for
    connectivity-style failures like the PermissionError in the
    mark_superseded test below)."""
    tome = _make_tome()
    await fs_tome_repo.insert(tome)
    with (
        mock.patch.object(Path, "write_text", side_effect=OSError("EIO")),
        pytest.raises(StorageError) as excinfo,
    ):
        await fs_tome_repo.update(tome.id, category="x")
    assert not isinstance(excinfo.value, BackendUnavailableError)


async def test_fs_mark_superseded_write_failure_raises_storage_error(
    fs_tome_repo: FsTomeRepository,
) -> None:
    tome = _make_tome()
    await fs_tome_repo.insert(tome)
    with (
        mock.patch.object(Path, "write_text", side_effect=PermissionError("denied")),
        pytest.raises(BackendUnavailableError),
    ):
        await fs_tome_repo.mark_superseded(tome.id, uuid.uuid4())


async def test_fs_list_all_scan_failure_raises_storage_error(
    fs_tome_repo: FsTomeRepository,
) -> None:
    """A directory-level failure (not per-file) must surface, not become []."""
    with (
        mock.patch.object(Path, "glob", side_effect=PermissionError("denied")),
        pytest.raises(BackendUnavailableError),
    ):
        await fs_tome_repo.list_all()


async def test_fs_count_scan_failure_raises_storage_error(
    fs_tome_repo: FsTomeRepository,
) -> None:
    with (
        mock.patch.object(Path, "glob", side_effect=PermissionError("denied")),
        pytest.raises(BackendUnavailableError),
    ):
        await fs_tome_repo.count()
