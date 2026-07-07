#!/usr/bin/env python3
"""Claude Code PostToolUse hook: mirror auto-memory files into the librarian.

Register on the ``PostToolUse`` event with matcher ``Write|Edit`` (async
recommended). See docs/claude-code-hooks.md for settings.json snippets.

Reads the hook JSON from stdin. When the written file is a Claude Code
auto-memory fact file (``.../memory/*.md``, excluding the ``MEMORY.md``
index), mirrors its content into the library so per-project file memories
become searchable from anywhere.

Semantics — replace-by-source, not similarity dedup: the memory FILE is
canonical, so existing tomes with the same ``source_url`` are deleted and the
current content re-ingested. Similarity-based deduplication is the wrong tool
for documents with a stable identity: when edits land close together, racing
merges interleave stale and fresh fragments. Concurrent runs for the same
file serialize on a lock, and the file is read AFTER the lock is acquired so
the last writer's content wins.

The YAML frontmatter is stripped; the memory's name and description are kept
as a header line so the generated title/summary (naive truncation without an
LLM) starts with something meaningful.

Tomes are stored at ``MEMORY_CONFIDENCE`` (via the ingest confidence
override) so they clear downstream ``min_confidence`` retrieval filters —
``skip_verify`` alone would store at the unverified default, which the
companion prompt-search hook filters out.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

# This script lives in <repo>/scripts/; LIBRARIAN_REPO overrides.
_DEFAULT_REPO = Path(__file__).resolve().parents[1]
REPO = Path(os.environ.get("LIBRARIAN_REPO") or _DEFAULT_REPO)
sys.path.insert(0, str(REPO))

_CONFIG_CANDIDATES = [
    os.environ.get("LIBRARIAN_CONFIG", ""),
    str(REPO / "librarian.config.yaml"),
    str(REPO / "config/local-fs.yaml"),
]

CATEGORY = "agent-memory"
MEMORY_CONFIDENCE = 0.75
LOCK_STALE_SECONDS = 600
# Keep below the hook timeout configured in settings.json so a queued run
# waiting on a long sync isn't killed mid-wait.
LOCK_WAIT_SECONDS = 150
MIN_BODY_CHARS = 40

_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _resolve_config() -> Path | None:
    for candidate in _CONFIG_CANDIDATES:
        if candidate:
            p = Path(candidate)
            if p.exists():
                return p
    return None


def _target_from_stdin() -> Path | None:
    """Extract the written file path from hook JSON; gate to memory fact files."""
    try:
        data = json.loads(sys.stdin.read())
    except Exception:
        return None
    tool_input = data.get("tool_input") or {}
    tool_response = data.get("tool_response") or {}
    raw = tool_input.get("file_path") or tool_response.get("filePath") or ""
    if not raw:
        return None
    normalized = raw.replace("\\", "/")
    if "/memory/" not in normalized or not normalized.endswith(".md"):
        return None
    target = Path(raw)
    if target.name == "MEMORY.md" or not target.is_file():
        return None
    return target


def _project_slug(target: Path) -> str:
    """Name of the directory that owns the memory/ dir (the munged project path)."""
    parts = list(target.parts)
    for i, part in enumerate(parts):
        if part.lower() == "memory" and i > 0:
            return parts[i - 1]
    return "unknown-project"


def _split_frontmatter(raw: str) -> tuple[str, str, str]:
    """Return (name, description, body). Falls back gracefully on parse failure."""
    match = _FRONTMATTER_RE.match(raw)
    if not match:
        return "", "", raw.strip()
    body = raw[match.end() :].strip()
    name, description = "", ""
    try:
        import yaml

        meta = yaml.safe_load(match.group(1)) or {}
        if isinstance(meta, dict):
            name = str(meta.get("name") or "")
            description = str(meta.get("description") or "")
    except Exception:
        pass
    return name, description, body


class _FileLock:
    """Cross-process lock via O_CREAT|O_EXCL; stale locks are broken."""

    def __init__(self, key: str) -> None:
        lock_dir = Path(tempfile.gettempdir()) / "librarian-memory-sync-locks"
        lock_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
        self._path = lock_dir / f"{digest}.lock"
        self._acquired = False

    def acquire(self) -> bool:
        deadline = time.monotonic() + LOCK_WAIT_SECONDS
        while time.monotonic() < deadline:
            try:
                fd = os.open(self._path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
                self._acquired = True
                return True
            except FileExistsError:
                try:
                    if time.time() - self._path.stat().st_mtime > LOCK_STALE_SECONDS:
                        self._path.unlink(missing_ok=True)
                        continue
                except OSError:
                    pass
                time.sleep(2)
        return False

    def release(self) -> None:
        if self._acquired:
            self._path.unlink(missing_ok=True)


async def _sync(target: Path) -> None:
    # Read AFTER acquiring the lock (caller) so the newest content wins.
    raw = target.read_text(encoding="utf-8", errors="replace")
    name, description, body = _split_frontmatter(raw)
    if len(body) < MIN_BODY_CHARS:
        return
    header = f"Memory '{name or target.stem}': {description}".strip()
    blob = f"{header}\n\n{body}".strip()
    source_url = target.as_uri()

    config_path = _resolve_config()
    if config_path is None:
        print("claude-memory-sync-hook: no librarian config found", file=sys.stderr)
        return

    from src.config import LibrarianConfig
    from src.models.enums import IngestStatus
    from src.services.embedding import build_embedding_service
    from src.services.ingestor import IngestCallOptions, Ingestor
    from src.services.verifier import Verifier
    from src.services.web_search import build_web_search_client
    from src.storage.tome_repository import TomeRepository

    config = LibrarianConfig.from_yaml(config_path)
    embedding_service = await build_embedding_service(config.embedding)
    repo: TomeRepository
    if config.database.uri.startswith("mongodb"):
        from src.storage.mongo.mongo_tome_repository import MongoTomeRepository

        repo = MongoTomeRepository(config.database, embedding_service)
    else:
        from src.storage.filesystem.fs_tome_repository import FsTomeRepository

        repo = FsTomeRepository(config.database, embedding_service)

    try:
        # Replace-by-source: drop every tome previously mirrored from this
        # file. Delete-first (not insert-first) so the fresh ingest can't hit
        # the similarity-dedup merge path against its own predecessors. If
        # the ingest then fails, the mirror is missing until the next edit —
        # acceptable because the FILE is canonical, and the failure is logged.
        existing = await repo.list_all(limit=500, category=CATEGORY)
        stale = [t for t in existing if t.source_url == source_url]
        for tome in stale:
            await repo.delete(tome.id)

        ingestor = Ingestor(
            config,
            embedding_service,
            Verifier(config, build_web_search_client(config)),
            repo,
        )
        opts = IngestCallOptions(
            skip_verify=True,
            category_hint=CATEGORY,
            tags_hint=["memory-sync", _project_slug(target)],
            force_format="text",
            source_url=source_url,
            confidence=MEMORY_CONFIDENCE,
        )
        result = await ingestor.ingest(blob, opts)
        if result.status in (IngestStatus.STORED, IngestStatus.PARTIAL):
            print(
                f"claude-memory-sync-hook: {target.name}: replaced {len(stale)} "
                f"-> {len(result.tomes)} tome(s)",
                file=sys.stderr,
            )
        else:
            print(
                f"claude-memory-sync-hook: {target.name} rejected "
                f"({result.reject_reason}); {len(stale)} prior tome(s) were already deleted",
                file=sys.stderr,
            )
    finally:
        repo.close()


def main() -> int:
    target = _target_from_stdin()
    if target is None:
        return 0
    lock = _FileLock(str(target).lower())
    if not lock.acquire():
        print(f"claude-memory-sync-hook: lock timeout for {target.name}", file=sys.stderr)
        return 0
    try:
        if target.is_file():
            asyncio.run(_sync(target))
    finally:
        lock.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
