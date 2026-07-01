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
    assert "library_update" in tool_names


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
        "library_update",
        "library_list",
    }


def _make_tome(
    content: str | None = None,
    *,
    tome_id: uuid.UUID | None = None,
    title: str = "Test Tome",
    category: str = "general",
    confidence: float = 0.8,
    research_job_id: uuid.UUID | None = None,
    tags: list[str] | None = None,
) -> Tome:
    return Tome(
        id=tome_id or uuid.uuid4(),
        title=title,
        content=content if content is not None else f"Body of {title}",
        summary=f"Summary of {title}",
        category=category,
        tags=tags or ["t"],
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


# ── library_search payload capping (issue #48) ──────────────────────


def _get_library_search(server):
    return next(t.fn for t in server.mcp._tool_manager.list_tools() if t.name == "library_search")


def _attach_repo(server, repo):
    """Bypass the lifespan and bind a pre-seeded repository for handler-level tests."""
    server.tome_repo = repo


async def test_library_search_default_returns_full_content_backcompat() -> None:
    """Default invocation preserves backward compatibility: full content returned."""
    from src.models.tool_schemas import SearchInput
    from src.server import LibrarianServer
    from tests.stubs import StubTomeRepository

    repo = StubTomeRepository()
    long_content = "abcdefghij" * 600  # 6000 chars
    await repo.insert(_make_tome(long_content))

    server = LibrarianServer(make_test_config())
    _attach_repo(server, repo)
    library_search = _get_library_search(server)

    out = await library_search(SearchInput(query="anything"))
    assert len(out.tomes) == 1
    assert out.tomes[0].content == long_content
    assert out.is_snippet == [False]
    assert out.tome_ids == [str(out.tomes[0].id)]


async def test_library_search_include_content_false_drops_content() -> None:
    """include_content=False replaces content with empty string and flags the hit."""
    from src.models.tool_schemas import SearchInput
    from src.server import LibrarianServer
    from tests.stubs import StubTomeRepository

    repo = StubTomeRepository()
    tome = _make_tome("a" * 5000)
    await repo.insert(tome)

    server = LibrarianServer(make_test_config())
    _attach_repo(server, repo)
    library_search = _get_library_search(server)

    out = await library_search(SearchInput(query="anything", include_content=False))
    assert out.tomes[0].content == ""
    assert out.is_snippet == [True]
    assert out.tome_ids == [str(tome.id)]


async def test_library_search_content_max_chars_truncates_with_ellipsis() -> None:
    """content_max_chars caps content length and appends an ellipsis when cut."""
    from src.models.tool_schemas import SearchInput
    from src.server import LibrarianServer
    from tests.stubs import StubTomeRepository

    repo = StubTomeRepository()
    await repo.insert(_make_tome("x" * 5000))

    server = LibrarianServer(make_test_config())
    _attach_repo(server, repo)
    library_search = _get_library_search(server)

    out = await library_search(SearchInput(query="x", content_max_chars=200))
    body = out.tomes[0].content
    assert body.endswith("...")
    assert len(body) <= 200
    assert body.startswith("x")
    assert out.is_snippet == [True]


async def test_library_search_content_max_chars_no_truncation_when_short() -> None:
    """When content already fits the cap, leave it untouched and do not flag it."""
    from src.models.tool_schemas import SearchInput
    from src.server import LibrarianServer
    from tests.stubs import StubTomeRepository

    repo = StubTomeRepository()
    await repo.insert(_make_tome("short content"))

    server = LibrarianServer(make_test_config())
    _attach_repo(server, repo)
    library_search = _get_library_search(server)

    out = await library_search(SearchInput(query="short", content_max_chars=500))
    assert out.tomes[0].content == "short content"
    assert out.is_snippet == [False]


async def test_library_search_snippet_window_around_match() -> None:
    """snippet_chars>0 returns a query-centred window around the first match."""
    from src.models.tool_schemas import SearchInput
    from src.server import LibrarianServer
    from tests.stubs import StubTomeRepository

    # Build a long blob with a unique marker buried deep inside.
    prefix = "lorem ipsum " * 200  # ~2400 chars
    marker = "UNIQUEMARKER"
    suffix = " dolor sit amet" * 200
    content = prefix + marker + suffix

    repo = StubTomeRepository()
    await repo.insert(_make_tome(content))

    server = LibrarianServer(make_test_config())
    _attach_repo(server, repo)
    library_search = _get_library_search(server)

    out = await library_search(SearchInput(query="UNIQUEMARKER", snippet_chars=200))
    body = out.tomes[0].content
    assert marker in body
    # snippet length is roughly the window + ellipsis padding
    assert len(body) <= 200 + 10
    assert out.is_snippet == [True]


async def test_library_search_snippet_no_match_returns_leading_window() -> None:
    """If no query term is found, snippet returns the leading window of the content."""
    from src.models.tool_schemas import SearchInput
    from src.server import LibrarianServer
    from tests.stubs import StubTomeRepository

    content = "abcdef" * 1000
    repo = StubTomeRepository()
    await repo.insert(_make_tome(content))

    server = LibrarianServer(make_test_config())
    _attach_repo(server, repo)
    library_search = _get_library_search(server)

    out = await library_search(SearchInput(query="ZZZ_not_present", snippet_chars=120))
    body = out.tomes[0].content
    assert body.startswith("abcdef")
    assert len(body) <= 120 + 10
    assert out.is_snippet == [True]


async def test_library_search_snippet_plus_max_chars_caps_snippet() -> None:
    """When both snippet_chars and content_max_chars are set, the cap also applies."""
    from src.models.tool_schemas import SearchInput
    from src.server import LibrarianServer
    from tests.stubs import StubTomeRepository

    content = "Q" * 50 + "needle" + "Q" * 1000
    repo = StubTomeRepository()
    await repo.insert(_make_tome(content))

    server = LibrarianServer(make_test_config())
    _attach_repo(server, repo)
    library_search = _get_library_search(server)

    out = await library_search(SearchInput(query="needle", snippet_chars=400, content_max_chars=80))
    body = out.tomes[0].content
    assert len(body) <= 80
    assert out.is_snippet == [True]


async def test_library_search_default_cap_from_config() -> None:
    """search.default_content_max_chars in config applies when caller does not override."""
    from src.config import SearchSettings
    from src.models.tool_schemas import SearchInput
    from src.server import LibrarianServer
    from tests.stubs import StubTomeRepository

    repo = StubTomeRepository()
    await repo.insert(_make_tome("p" * 4000))

    cfg = make_test_config(search=SearchSettings(default_content_max_chars=300))
    server = LibrarianServer(cfg)
    _attach_repo(server, repo)
    library_search = _get_library_search(server)

    out = await library_search(SearchInput(query="anything"))
    body = out.tomes[0].content
    assert body.endswith("...")
    assert len(body) <= 300
    assert out.is_snippet == [True]


async def test_library_search_output_exposes_tome_ids() -> None:
    """tome_ids is always populated 1:1 with tomes so callers can fetch full content."""
    from src.models.tool_schemas import SearchInput
    from src.server import LibrarianServer
    from tests.stubs import StubTomeRepository

    repo = StubTomeRepository()
    t1 = _make_tome("alpha", title="A")
    t2 = _make_tome("beta", title="B")
    await repo.insert(t1)
    await repo.insert(t2)

    server = LibrarianServer(make_test_config())
    _attach_repo(server, repo)
    library_search = _get_library_search(server)

    out = await library_search(SearchInput(query="x", top_k=5))
    assert out.tome_ids == [str(t.id) for t in out.tomes]


# ────────────────────────────────────────────────────────────────────────────
# Tests for library_update (Issue #43)
# ────────────────────────────────────────────────────────────────────────────


async def test_library_update_basic() -> None:
    """Test basic update of content and tags."""
    from src.models.tool_schemas import UpdateInput

    repo = StubTomeRepository()
    tome = _make_tome("original", tags=["old"])
    await repo.insert(tome)
    tome_id = str(tome.id)

    server = _build_server_with_repo(repo)
    library_update = _get_tool(server, "library_update")

    result = await library_update(
        UpdateInput(tome_id=tome_id, content="updated content", tags=["new"])
    )

    assert result.status == "updated"
    assert result.tome is not None
    assert result.tome.content == "updated content"
    assert result.tome.tags == ["new"]


async def test_library_update_partial() -> None:
    """Test that only specified fields update, others unchanged."""
    from src.models.tool_schemas import UpdateInput

    repo = StubTomeRepository()
    tome = _make_tome("original", category="math", tags=["t1", "t2"])
    await repo.insert(tome)
    tome_id = str(tome.id)

    server = _build_server_with_repo(repo)
    library_update = _get_tool(server, "library_update")

    result = await library_update(UpdateInput(tome_id=tome_id, tags=["new_tag"]))

    assert result.status == "updated"
    assert result.tome is not None
    assert result.tome.category == "math"  # unchanged
    assert result.tome.tags == ["new_tag"]  # updated
    assert result.tome.content == "original"  # unchanged


async def test_library_update_invalid_tome_id() -> None:
    """Test invalid UUID format returns invalid_id status."""
    from src.models.tool_schemas import UpdateInput

    server = _build_server_with_repo(StubTomeRepository())
    library_update = _get_tool(server, "library_update")

    result = await library_update(UpdateInput(tome_id="not-a-uuid", content="new content"))

    assert result.status == "invalid_id"
    assert result.tome is None


async def test_library_update_not_found() -> None:
    """Test valid UUID but non-existent tome returns not_found status."""
    from src.models.tool_schemas import UpdateInput

    server = _build_server_with_repo(StubTomeRepository())
    library_update = _get_tool(server, "library_update")

    result = await library_update(UpdateInput(tome_id=str(uuid.uuid4()), content="new content"))

    assert result.status == "not_found"
    assert result.tome is None


async def test_library_update_persists_via_get() -> None:
    """Test that updated tome can be retrieved via library_get."""
    from src.models.tool_schemas import GetInput, UpdateInput

    repo = StubTomeRepository()
    tome = _make_tome("original")
    await repo.insert(tome)
    tome_id = str(tome.id)

    server = _build_server_with_repo(repo)
    library_update = _get_tool(server, "library_update")
    library_get = _get_tool(server, "library_get")

    # Update the tome
    update_result = await library_update(
        UpdateInput(tome_id=tome_id, content="updated content", category="physics")
    )
    assert update_result.status == "updated"

    # Retrieve via library_get
    get_result = await library_get(GetInput(tome_id=tome_id))
    assert get_result.status == "found"
    assert get_result.tome is not None
    assert get_result.tome.content == "updated content"
    assert get_result.tome.category == "physics"


# ---------------------------------------------------------------------------
# Backfill: previously-untested handler branches (coverage workstream)
# ---------------------------------------------------------------------------


async def test_library_tidy_handler_returns_numeric_report() -> None:
    """The library_tidy MCP handler was never invoked by any test.

    Guards the ``TidyOutput(**report)`` unpacking: if the Tidier report dict
    shape drifts from the TidyOutput schema, this fails.
    """
    from src.models.tool_schemas import TidyInput
    from src.services.tidier import Tidier
    from tests.stubs import StubEmbeddingService, StubIngestor, StubVerifier

    config = make_test_config()
    repo = StubTomeRepository()
    await repo.insert(_make_tome(title="Solo"))
    ingestor = StubIngestor(
        config, StubEmbeddingService(dimensions=config.embedding.dimensions), StubVerifier(), repo
    )

    server = _build_server_with_repo(repo)
    server.tidier = Tidier(ingestor, repo, config.tidy)
    library_tidy = _get_tool(server, "library_tidy")

    out = await library_tidy(TidyInput())

    assert out.scanned >= 0
    assert out.groups_found >= 0
    assert out.tomes_removed >= 0
    assert out.elapsed_ms >= 0


async def test_library_list_invalid_research_job_id_raises_value_error() -> None:
    """A malformed research_job_id must raise, not silently return everything."""
    from src.models.tool_schemas import ListInput

    server = _build_server_with_repo(StubTomeRepository())
    library_list = _get_tool(server, "library_list")

    with pytest.raises(ValueError, match="research_job_id"):
        await library_list(ListInput(research_job_id="not-a-uuid"))


async def test_library_list_include_summary_false_blanks_summaries() -> None:
    from src.models.tool_schemas import ListInput

    repo = StubTomeRepository()
    await repo.insert(_make_tome(title="A"))

    server = _build_server_with_repo(repo)
    library_list = _get_tool(server, "library_list")

    out = await library_list(ListInput(include_summary=False))

    assert out.tomes
    assert all(t.summary == "" for t in out.tomes)


async def test_library_search_include_summary_false_blanks_summaries() -> None:
    from src.models.tool_schemas import SearchInput

    repo = StubTomeRepository()
    await repo.insert(_make_tome(title="A"))

    server = _build_server_with_repo(repo)
    library_search = _get_tool(server, "library_search")

    out = await library_search(SearchInput(query="a", min_confidence=0.0, include_summary=False))

    assert out.tomes
    assert all(t.summary == "" for t in out.tomes)


async def test_library_search_passes_filters_through_to_repo() -> None:
    """The handler must forward top_k / min_confidence / category /
    include_superseded to the repository rather than re-filtering itself."""
    from src.models.tool_schemas import SearchInput

    repo = StubTomeRepository()
    await repo.insert(_make_tome(title="A"))
    server = _build_server_with_repo(repo)
    library_search = _get_tool(server, "library_search")

    captured: dict[str, object] = {}
    original_search = repo.search

    async def capturing_search(query, top_k=5, min_confidence=0.5, **kwargs):
        captured.update({"query": query, "top_k": top_k, "min_confidence": min_confidence})
        captured.update(kwargs)
        return await original_search(query, top_k=top_k, min_confidence=min_confidence, **kwargs)

    repo.search = capturing_search  # type: ignore[method-assign]

    await library_search(
        SearchInput(
            query="q",
            top_k=7,
            min_confidence=0.25,
            category="science",
            include_superseded=True,
        )
    )

    assert captured["top_k"] == 7
    assert captured["min_confidence"] == 0.25
    assert captured["category"] == "science"
    assert captured["include_superseded"] is True


async def test_library_research_blank_topic_returns_invalid_input() -> None:
    """No job_id and a whitespace topic must return invalid_input in-band."""
    from unittest.mock import MagicMock

    from src.models.tool_schemas import ResearchInput

    server = _build_server_with_repo(StubTomeRepository())
    # The handler requires these to be non-None, but the invalid-input branch
    # returns before either is used.
    server.job_repo = MagicMock()
    server.researcher = MagicMock()
    library_research = _get_tool(server, "library_research")

    out = await library_research(ResearchInput(topic="   "))

    assert out.status == "invalid_input"
    assert out.error is not None


async def test_library_research_sync_path_runs_job_to_completion() -> None:
    """async_=False must run the job inline and return its final state."""
    from unittest.mock import AsyncMock, MagicMock

    from src.models.enums import ResearchJobStatus
    from src.models.research_job import ResearchJob
    from src.models.tool_schemas import ResearchInput

    server = _build_server_with_repo(StubTomeRepository())

    inserted: list[ResearchJob] = []

    async def capture_insert(job: ResearchJob):
        inserted.append(job)
        return job.id

    job_repo = MagicMock()
    job_repo.insert = AsyncMock(side_effect=capture_insert)
    server.job_repo = job_repo
    researcher = MagicMock()
    researcher.run_job = AsyncMock()
    server.researcher = researcher
    library_research = _get_tool(server, "library_research")

    def make_completed(job_id):
        return ResearchJob(
            id=job_id,
            topic="quantum ducks",
            status=ResearchJobStatus.COMPLETED,
        )

    job_repo.get_by_id = AsyncMock(side_effect=lambda jid: make_completed(jid))

    out = await library_research(ResearchInput(topic="quantum ducks", async_=False))

    assert len(inserted) == 1
    researcher.run_job.assert_awaited_once_with(inserted[0].id)
    assert out.status == ResearchJobStatus.COMPLETED.value
    assert server._bg_tasks == set(), "sync path must not schedule a background task"
