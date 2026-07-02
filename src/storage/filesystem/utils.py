"""Shared helpers for filesystem-backed repository implementations."""

from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4


def atomic_write_text(path: Path, data: str) -> None:
    """Write ``data`` to ``path`` via a temp file + ``os.replace``.

    A plain ``Path.write_text`` truncates then writes, so a concurrent reader
    (e.g. a research-job poll racing the background job's status update) can
    observe empty or partial JSON and misread a live record as corrupt or
    absent. ``os.replace`` is atomic on both POSIX and Windows, so readers
    always see either the old or the new complete document.

    The temp file lives in the same directory (required for an atomic rename)
    with a ``.tmp`` suffix so ``glob("*.json")`` scans never pick it up.
    """
    tmp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        tmp.write_text(data)
        os.replace(tmp, path)
    except OSError:
        tmp.unlink(missing_ok=True)
        raise


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
        path_str = uri[7:]
        if path_str.startswith("/~"):
            path_str = path_str[1:]
        return Path(path_str).expanduser()
    return Path(uri).expanduser()
