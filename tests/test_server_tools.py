"""Smoke tests for the MCP server tool registration."""

from __future__ import annotations

import importlib

import pytest

from tests.conftest import make_test_config


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


def test_no_unexpected_tools_registered() -> None:
    """Guard against accidental extra tool registrations."""
    mcp = _build_mcp()

    tool_names = {t.name for t in mcp._tool_manager.list_tools()}
    assert tool_names == {
        "library_search",
        "library_ingest",
        "library_research",
        "library_tidy",
    }
