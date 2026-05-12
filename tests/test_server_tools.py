"""Smoke tests for the MCP server tool registration."""

from __future__ import annotations

import importlib
import uuid
from uuid import uuid4

import numpy as np
import pytest

from src.models.enums import SourceType
from src.models.tome import Tome
from tests.conftest import make_test_config
from tests.stubs import StubTomeRepository


def _build_mcp():
    """Construct a LibrarianServer with test config and return its mcp instance."""
    from src.server import LibrarianServer

    config = make_test_config()
    return LibrarianServer(config).mcp


def test_import_with_no_env_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Importing src.server must not load config or fail on missing env vars."""
    monkeypatch.delenv("LIBRARIAN_CONFIG", raising=False)
    monkeypatch.delenv("LIBRARIAN_DATABASE_URI", raising=False)

    import src.server

    importlib.reload(src.server)

    # Module exposes the class but does not eagerly construct a server/config.
    assert hasattr(src.server, "LibrarianServer")
    assert hasattr(src.server, "load_config")


def test_load_config_with_bad_path_raises_controlled_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """load_config raises a normal exception (not ImportError) on a bad path.

    Acceptance for issue #22: errors only occur when ``load_config()`` runs,
    not at import time. The specific exception type is whatever
    ``LibrarianConfig.from_yaml`` raises (FileNotFoundError, ValueError, etc.).
    """
    monkeypatch.setenv("LIBRARIAN_CONFIG", str(tmp_path / "does-not-exist.yml"))
    monkeypatch.delenv("LIBRARIAN_DATABASE_URI", raising=False)

    from src.server import load_config

    with pytest.raises(Exception) as exc_info:
        load_config()

    assert not isinstance(exc_info.value, ImportError)


def test_expected_tools_are_registered() -> None:
    """All MCP tools must be registered on the server."""
    mcp = _build_mcp()

    tool_names = {t.name for t in mcp._tool_manager.list_tools()}
    assert "library_search" in tool_names
    assert "library_ingest" in tool_names
    assert "library_research" in tool_names
    assert "library_get" in tool_names
    assert "library_delete" in tool_names


def test_no_unexpected_tools_registered() -> None:
    """Guard against accidental extra tool registrations."""
    mcp = _build_mcp()

    tool_names = {t.name for t in mcp._tool_manager.list_tools()}
    assert tool_names == {
        "library_search",
        "library_ingest",
        "library_research",
        "library_tidy",
        "library_get",
        "library_delete",
        "library_list",
    }


def _make_tome(
    *,
    title: str,
    category: str = "general",
    confidence: float = 0.8,
    research_job_id: uuid.UUID | None = None,
) -> Tome:
    return Tome(
        id=uuid.uuid4(),
        title=title,
        content=f"Body of {title}",
        summary=f"Summary of {title}",
        category=category,
        tags=["t"],
        source_url=None,
        source_type=SourceType.AGENT_INPUT,
        confidence=confidence,
        research_job_id=research_job_id,
        embedding=np.zeros(8, dtype=np.float32),
    )


def _build_server_with_repo(repo: StubTomeRepository):
    """Construct a LibrarianServer and inject a pre-populated stub repo.

    Bypasses lifespan() so unit tests don't need MongoDB or Ollama running.
    """
    from src.server import LibrarianServer

    server = LibrarianServer(make_test_config())
    server.tome_repo = repo
    return server


def _get_tool(server, name: str):
    return next(t.fn for t in server.mcp._tool_manager.list_tools() if t.name == name)


async def test_library_list_paginates_first_page() -> None:
    """library_list returns the requested page and signals more results remain."""
    from src.models.tool_schemas import ListInput

    repo = StubTomeRepository()
    for i in range(5):
        await repo.insert(_make_tome(title=f"Tome {i}"))

    server = _build_server_with_repo(repo)
    library_list = _get_tool(server, "library_list")

    out = await library_list(ListInput(limit=2, offset=0))

    assert len(out.tomes) == 2
    assert out.total == 5
    assert out.has_more is True


async def test_library_list_offset_returns_remainder_no_more() -> None:
    """library_list with offset returns the tail and reports no more pages."""
    from src.models.tool_schemas import ListInput

    repo = StubTomeRepository()
    for i in range(5):
        await repo.insert(_make_tome(title=f"Tome {i}"))

    server = _build_server_with_repo(repo)
    library_list = _get_tool(server, "library_list")

    out = await library_list(ListInput(limit=10, offset=3))

    assert len(out.tomes) == 2
    assert out.total == 5
    assert out.has_more is False


async def test_library_list_filters_by_category() -> None:
    """library_list applies category filter before pagination."""
    from src.models.tool_schemas import ListInput

    repo = StubTomeRepository()
    await repo.insert(_make_tome(title="A", category="science"))
    await repo.insert(_make_tome(title="B", category="history"))
    await repo.insert(_make_tome(title="C", category="science"))

    server = _build_server_with_repo(repo)
    library_list = _get_tool(server, "library_list")

    out = await library_list(ListInput(category="science"))

    assert out.total == 2
    assert {t.title for t in out.tomes} == {"A", "C"}
    assert out.has_more is False


async def test_library_list_filters_by_research_job_id() -> None:
    """library_list filters by research_job_id when provided."""
    from src.models.tool_schemas import ListInput

    job_id = uuid.uuid4()
    repo = StubTomeRepository()
    await repo.insert(_make_tome(title="J1", research_job_id=job_id))
    await repo.insert(_make_tome(title="J2", research_job_id=job_id))
    await repo.insert(_make_tome(title="Other", research_job_id=uuid.uuid4()))

    server = _build_server_with_repo(repo)
    library_list = _get_tool(server, "library_list")

    out = await library_list(ListInput(research_job_id=str(job_id)))

    assert out.total == 2
    assert {t.title for t in out.tomes} == {"J1", "J2"}


async def test_library_list_strips_embeddings() -> None:
    """Embeddings should be stripped from list output to keep payload small."""
    from src.models.tool_schemas import ListInput

    repo = StubTomeRepository()
    await repo.insert(_make_tome(title="Tome"))

    server = _build_server_with_repo(repo)
    library_list = _get_tool(server, "library_list")

    out = await library_list(ListInput())

    assert len(out.tomes) == 1
    assert out.tomes[0].embedding is None


async def test_library_list_min_confidence_filter() -> None:
    """library_list respects min_confidence filter."""
    from src.models.tool_schemas import ListInput

    repo = StubTomeRepository()
    await repo.insert(_make_tome(title="Hi", confidence=0.9))
    await repo.insert(_make_tome(title="Lo", confidence=0.2))

    server = _build_server_with_repo(repo)
    library_list = _get_tool(server, "library_list")

    out = await library_list(ListInput(min_confidence=0.5))

    assert out.total == 1
    assert out.tomes[0].title == "Hi"


async def test_library_search_outside_lifespan_raises_runtime_error() -> None:
    """Tool handlers must raise RuntimeError (not AssertionError or AttributeError)
    when invoked before the lifespan has initialised the server's repositories."""
    from src.models.tool_schemas import SearchInput
    from src.server import LibrarianServer

    server = LibrarianServer(make_test_config())
    # Sanity: repositories are not initialised because lifespan was not entered.
    assert server.tome_repo is None

    library_search = next(
        t.fn for t in server.mcp._tool_manager.list_tools() if t.name == "library_search"
    )

    with pytest.raises(RuntimeError, match="LibrarianServer not initialised"):
        await library_search(SearchInput(query="anything"))


def _make_tome(*, tome_id: uuid.UUID | None = None, title: str = "Test Tome") -> Tome:
    return Tome(
        id=tome_id or uuid.uuid4(),
        title=title,
        content="Some content.",
        summary="A summary.",
        category="general",
        tags=[],
        source_url=None,
        source_type=SourceType.AGENT_INPUT,
        confidence=0.8,
        embedding=None,
    )


# ── library_delete behavioural tests (issue #39) ────────────────────


def _make_test_tome(tome_id=None):
    """Build a Tome instance suitable for inserting into a stub repo."""
    from uuid import uuid4 as _uuid4

    from src.models.enums import SourceType
    from src.models.tome import Tome

    return Tome(
        id=tome_id or _uuid4(),
        title="t",
        content="c",
        summary="s",
        category="general",
        tags=["x"],
        source_type=SourceType.MANUAL,
        confidence=0.9,
        embedding=None,
    )


async def test_library_get_returns_tome_when_present() -> None:
    """library_get(tome_id) returns the Tome when it exists in the repository."""
    from src.models.tool_schemas import GetInput
    from src.server import LibrarianServer

    server = LibrarianServer(make_test_config())
    repo = StubTomeRepository()
    tome = _make_tome(title="The Hobbit")
    await repo.insert(tome)
    server.tome_repo = repo

    library_get = next(
        t.fn for t in server.mcp._tool_manager.list_tools() if t.name == "library_get"
    )

    result = await library_get(GetInput(tome_id=str(tome.id)))

    assert result.tome is not None
    assert result.tome.id == tome.id
    assert result.tome.title == "The Hobbit"
    assert result.status == "found"
    assert result.error is None


async def test_library_get_accepts_canonical_hyphenated_uuid() -> None:
    """library_get must accept the canonical hyphenated UUID form (the default
    str(uuid) representation also used by Tome.id serialisation)."""
    from src.models.tool_schemas import GetInput
    from src.server import LibrarianServer

    server = LibrarianServer(make_test_config())
    repo = StubTomeRepository()
    tome = _make_tome()
    await repo.insert(tome)
    server.tome_repo = repo

    library_get = next(
        t.fn for t in server.mcp._tool_manager.list_tools() if t.name == "library_get"
    )

    canonical = str(tome.id)
    assert "-" in canonical
    result = await library_get(GetInput(tome_id=canonical))
    assert result.status == "found"
    assert result.tome is not None
    assert result.tome.id == tome.id


async def test_library_get_returns_not_found_for_unknown_id() -> None:
    """library_get returns a not_found status for an unknown but valid UUID."""
    from src.models.tool_schemas import GetInput
    from src.server import LibrarianServer

    server = LibrarianServer(make_test_config())
    server.tome_repo = StubTomeRepository()

    library_get = next(
        t.fn for t in server.mcp._tool_manager.list_tools() if t.name == "library_get"
    )

    missing_id = uuid.uuid4()
    result = await library_get(GetInput(tome_id=str(missing_id)))

    assert result.tome is None
    assert result.status == "not_found"
    assert result.error is not None


async def test_library_get_returns_invalid_for_bad_uuid() -> None:
    """library_get returns invalid_tome_id when the tome_id is not a UUID."""
    from src.models.tool_schemas import GetInput
    from src.server import LibrarianServer

    server = LibrarianServer(make_test_config())
    server.tome_repo = StubTomeRepository()

    library_get = next(
        t.fn for t in server.mcp._tool_manager.list_tools() if t.name == "library_get"
    )

    result = await library_get(GetInput(tome_id="not-a-uuid"))

    assert result.tome is None
    assert result.status == "invalid_tome_id"
    assert result.error is not None


async def test_library_get_strips_embedding() -> None:
    """library_get returns the Tome without its embedding payload."""
    import numpy as np

    from src.models.tool_schemas import GetInput
    from src.server import LibrarianServer

    server = LibrarianServer(make_test_config())
    repo = StubTomeRepository()
    tome = _make_tome()
    tome = tome.model_copy(update={"embedding": np.zeros(8, dtype=np.float32)})
    await repo.insert(tome)
    server.tome_repo = repo

    library_get = next(
        t.fn for t in server.mcp._tool_manager.list_tools() if t.name == "library_get"
    )

    result = await library_get(GetInput(tome_id=str(tome.id)))
    assert result.tome is not None
    assert result.tome.embedding is None


async def test_library_get_outside_lifespan_raises_runtime_error() -> None:
    """library_get must raise RuntimeError if lifespan never initialised the repository."""
    from src.models.tool_schemas import GetInput
    from src.server import LibrarianServer

    server = LibrarianServer(make_test_config())
    assert server.tome_repo is None

    library_get = next(
        t.fn for t in server.mcp._tool_manager.list_tools() if t.name == "library_get"
    )

    with pytest.raises(RuntimeError, match="LibrarianServer not initialised"):
        await library_get(GetInput(tome_id=str(uuid.uuid4())))


async def test_library_delete_existing_tome_returns_deleted_true() -> None:
    """Deleting an existing tome returns deleted=True and removes it from the repo."""
    from src.models.tool_schemas import DeleteInput
    from src.server import LibrarianServer

    server = LibrarianServer(make_test_config())
    repo = StubTomeRepository()
    tome = _make_test_tome()
    await repo.insert(tome)
    server.tome_repo = repo

    library_delete = next(
        t.fn for t in server.mcp._tool_manager.list_tools() if t.name == "library_delete"
    )

    result = await library_delete(DeleteInput(tome_id=tome.id.hex))
    assert result.deleted is True
    assert result.error is None
    assert await repo.get_by_id(tome.id) is None


async def test_library_delete_missing_tome_returns_not_found() -> None:
    """Deleting a non-existent UUID returns deleted=False with error='not_found'."""
    from src.models.tool_schemas import DeleteInput
    from src.server import LibrarianServer

    server = LibrarianServer(make_test_config())
    server.tome_repo = StubTomeRepository()

    library_delete = next(
        t.fn for t in server.mcp._tool_manager.list_tools() if t.name == "library_delete"
    )

    result = await library_delete(DeleteInput(tome_id=uuid4().hex))
    assert result.deleted is False
    assert result.error == "not_found"


async def test_library_delete_accepts_hyphenated_uuid() -> None:
    """The handler accepts both hex and standard hyphenated UUID strings."""
    from src.models.tool_schemas import DeleteInput
    from src.server import LibrarianServer
    from tests.stubs import StubTomeRepository

    server = LibrarianServer(make_test_config())
    repo = StubTomeRepository()
    tome = _make_test_tome()
    await repo.insert(tome)
    server.tome_repo = repo

    library_delete = next(
        t.fn for t in server.mcp._tool_manager.list_tools() if t.name == "library_delete"
    )

    # Hyphenated form (36 chars) — the canonical str(UUID) representation.
    result = await library_delete(DeleteInput(tome_id=str(tome.id)))
    assert result.deleted is True
    assert result.error is None


async def test_library_delete_invalid_id_returns_invalid_id_error() -> None:
    """Malformed tome_id is reported as an error, never raised."""
    from src.models.tool_schemas import DeleteInput
    from src.server import LibrarianServer

    server = LibrarianServer(make_test_config())
    server.tome_repo = StubTomeRepository()

    library_delete = next(
        t.fn for t in server.mcp._tool_manager.list_tools() if t.name == "library_delete"
    )

    result = await library_delete(DeleteInput(tome_id="not-a-uuid"))
    assert result.deleted is False
    assert result.error == "invalid_id"
