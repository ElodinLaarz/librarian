"""Smoke tests for the MCP server tool registration."""

from __future__ import annotations

import importlib
import uuid

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
    }


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
