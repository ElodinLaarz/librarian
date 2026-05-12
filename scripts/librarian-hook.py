#!/usr/bin/env python3
"""
Librarian hook for AI agents (Claude Code, Gemini CLI).

Handles:
1. UserPromptSubmit (Claude Code): Pre-prompt search & context injection.
2. AfterAgent (Gemini CLI): Post-response auto-ingest of new knowledge.
3. AfterTool (Gemini CLI): Post-tool auto-ingest (e.g. from save_memory).

Reads from stdin (JSON format), optionally searches/ingests, and writes
JSON to stdout (Gemini CLI) or text (Claude Code).
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


# ── Services ─────────────────────────────────────────────────────────────────

MAX_RESULTS = 3
MIN_CONFIDENCE = 0.55
MAX_CONTENT_CHARS = 600


async def _get_repo_and_ingestor():
    config_path = _resolve_config()
    if config_path is None:
        return None, None

    try:
        from src.config import LibrarianConfig
        from src.services.embedding import build_embedding_service
        from src.services.ingestor import Ingestor
        from src.services.verifier import Verifier
        from src.services.web_search import build_web_search_client
        from src.storage.tome_repository import TomeRepository

        config = LibrarianConfig.from_yaml(config_path)
        embedding_service = await build_embedding_service(config.embedding)

        if config.database.uri.startswith("mongodb"):
            from src.storage.mongo.mongo_tome_repository import MongoTomeRepository

            repo: TomeRepository = MongoTomeRepository(config.database, embedding_service)
        else:
            from src.storage.filesystem.fs_tome_repository import FsTomeRepository

            repo = FsTomeRepository(config.database, embedding_service)

        web_client = build_web_search_client(config)
        verifier = Verifier(config, web_client)
        ingestor = Ingestor(config, embedding_service, verifier, repo)

        return repo, ingestor
    except Exception as exc:
        print(f"librarian-hook: Service init failed: {exc}", file=sys.stderr)
        return None, None


async def _search(query: str) -> list[tuple[str, str, float]]:
    """Return (summary, content, score) triples; empty list on any failure."""
    repo, _ = await _get_repo_and_ingestor()
    if repo is None:
        return []

    try:
        results = await repo.search(query, top_k=MAX_RESULTS, min_confidence=MIN_CONFIDENCE)
        return [(t.summary, t.content, s) for t, s in results]
    except Exception as exc:  # noqa: BLE001
        print(f"librarian-hook search: {exc}", file=sys.stderr)
        return []


async def _ingest(content: str, source_url: str | None = None) -> bool:
    """Ingest content into the library; returns True on success."""
    _, ingestor = await _get_repo_and_ingestor()
    if ingestor is None:
        return False

    try:
        from src.models.enums import IngestStatus
        from src.services.ingestor import IngestCallOptions

        # We skip verification for auto-ingest hooks to avoid side-effects/latency
        # unless the content is very long or specifically requested.
        opts = IngestCallOptions(skip_verify=True, source_url=source_url)
        result = await ingestor.ingest(content, opts)

        if result.status in (IngestStatus.STORED, IngestStatus.PARTIAL):
            print(f"librarian-hook: Auto-ingested {len(result.tomes)} tomes.", file=sys.stderr)
            return True
        return False
    except Exception as exc:
        print(f"librarian-hook ingest: {exc}", file=sys.stderr)
        return False


# ── Main ─────────────────────────────────────────────────────────────────────


def handle_claude_code(data: dict) -> None:
    prompt = (data.get("prompt") or "").strip()
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


def handle_gemini_cli(data: dict, event: str) -> None:
    # Always output JSON to stdout for Gemini CLI
    output = {}

    try:
        if event == "AfterAgent":
            # Auto-ingest the agent's response if it seems informative
            response = (data.get("prompt_response") or "").strip()
            if len(response) > 100:
                asyncio.run(_ingest(response))

        elif event == "AfterTool":
            # Auto-ingest when memory-related tools are called
            tool_name = data.get("tool_name")
            tool_input = data.get("tool_input", {})

            if tool_name == "save_memory":
                fact = tool_input.get("fact")
                if fact:
                    asyncio.run(_ingest(fact))
            elif tool_name in ("library_ingest", "mcp_librarian_library_ingest"):
                # Already handled by the tool itself, but could log or post-process here.
                pass

    except Exception as exc:
        print(f"librarian-hook gemini: {exc}", file=sys.stderr)

    print(json.dumps(output))


def main() -> None:
    try:
        input_str = sys.stdin.read()
        if not input_str:
            return
        data = json.loads(input_str)
    except Exception:
        return

    gemini_event = os.environ.get("GEMINI_HOOK_EVENT")
    if gemini_event:
        handle_gemini_cli(data, gemini_event)
    else:
        # Default to Claude Code logic (UserPromptSubmit)
        handle_claude_code(data)


if __name__ == "__main__":
    main()
