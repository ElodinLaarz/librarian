#!/usr/bin/env python3
"""
Librarian UserPromptSubmit hook for Claude Code.

Reads the user's prompt from stdin (Claude Code hook JSON format),
searches the library for relevant context, and writes results to stdout
for injection into the conversation before Claude responds.

Called automatically by Claude Code; never blocks the prompt on error.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

# ── Config resolution ────────────────────────────────────────────────────────

REPO = Path(os.environ.get("LIBRARIAN_REPO", Path.home() / "github/librarian"))
sys.path.insert(0, str(REPO))

_CONFIG_CANDIDATES = [
    os.environ.get("LIBRARIAN_CONFIG", ""),
    str(REPO / "librarian.config.yaml"),
    str(REPO / "config/local-fs.yaml"),
]


def _resolve_config() -> Path | None:
    for candidate in _CONFIG_CANDIDATES:
        if candidate:
            p = Path(candidate)
            if p.exists():
                return p
    return None


# ── Search ───────────────────────────────────────────────────────────────────

MAX_RESULTS = 3
MIN_CONFIDENCE = 0.55
MAX_CONTENT_CHARS = 600


async def _search(query: str) -> list[tuple[str, str, float]]:
    """Return (summary, content, score) triples; empty list on any failure."""
    config_path = _resolve_config()
    if config_path is None:
        return []

    try:
        from src.config import LibrarianConfig
        from src.services.embedding import build_embedding_service

        config = LibrarianConfig.from_yaml(config_path)
        embedding_service = await build_embedding_service(config.embedding)

        if config.database.uri.startswith("mongodb"):
            from src.storage.mongo.mongo_tome_repository import MongoTomeRepository

            repo: object = MongoTomeRepository(config.database, embedding_service)
            await repo.ensure_indexes()  # type: ignore[attr-defined]
        else:
            from src.storage.filesystem.fs_tome_repository import FsTomeRepository

            repo = FsTomeRepository(config.database, embedding_service)

        results = await repo.search(  # type: ignore[union-attr]
            query, top_k=MAX_RESULTS, min_confidence=MIN_CONFIDENCE
        )
        return [(t.summary, t.content, s) for t, s in results]

    except Exception as exc:  # noqa: BLE001
        print(f"librarian-hook: {exc}", file=sys.stderr)
        return []


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    try:
        data = json.load(sys.stdin)
        prompt = (data.get("prompt") or "").strip()
    except Exception:  # noqa: BLE001
        return

    # Skip trivially short prompts — unlikely to benefit from library context.
    if len(prompt) < 20:
        return

    results = asyncio.run(_search(prompt))
    if not results:
        return

    lines = [
        "<librarian_context>",
        "Relevant knowledge retrieved from your personal library:",
    ]
    for i, (summary, content, score) in enumerate(results, 1):
        snippet = content.strip()[:MAX_CONTENT_CHARS]
        if len(content.strip()) > MAX_CONTENT_CHARS:
            snippet += "…"
        lines.append(f"\n[{i}] {summary}  (score: {score:.2f})")
        lines.append(snippet)
    lines.append("</librarian_context>")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
