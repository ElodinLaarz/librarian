"""Smoke tests for the MCP server tool registration."""

from __future__ import annotations

import os

# database.uri is required; set before src.server is first imported.
os.environ.setdefault("LIBRARIAN_DATABASE_URI", "mongodb://localhost:27017")


def test_expected_tools_are_registered() -> None:
    """All MCP tools must be registered on the server at import time."""
    from src.server import mcp

    tool_names = {t.name for t in mcp._tool_manager.list_tools()}
    assert "library_search" in tool_names
    assert "library_ingest" in tool_names
    assert "library_research" in tool_names


def test_no_unexpected_tools_registered() -> None:
    """Guard against accidental extra tool registrations."""
    from src.server import mcp

    tool_names = {t.name for t in mcp._tool_manager.list_tools()}
    assert tool_names == {
        "library_search",
        "library_ingest",
        "library_research",
        "library_tidy",
    }
