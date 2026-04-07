"""Shared helpers for filesystem-backed repository implementations."""

from __future__ import annotations

from pathlib import Path


def resolve_base_path(uri: str) -> Path:
    """Translate a ``database.uri`` value into an absolute filesystem path.

    Accepted forms:
    - ``""`` or ``"localhost"``  → ``~/.librarian_mcp``  (default dev location)
    - ``"file:///abs/path"``     → ``/abs/path``
    - any other string           → treated as a raw path (``expanduser`` applied)
    """
    if uri in ("localhost", ""):
        return Path.home() / ".librarian_mcp"
    if uri.startswith("file://"):
        return Path(uri[7:]).expanduser()
    return Path(uri).expanduser()
