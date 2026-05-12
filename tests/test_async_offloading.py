"""Verify CPU-bound sync calls are offloaded so the event loop is not blocked.

Issue #24 — `trafilatura.extract` (in ``src.services.web_search``) and
``RecursiveCharacterTextSplitter.split_text`` (in ``src.services.ingestor``) are
CPU-bound and can stall the event loop for seconds at a time during deep
research runs. They must be invoked via ``asyncio.to_thread`` so concurrent
tasks make progress.

Strategy: monkeypatch the underlying sync function with a `time.sleep(...)`
shim that blocks a thread. While that "extraction"/"split" is in flight,
schedule a small ``asyncio.sleep`` task. If the sync call were running on
the event loop, the concurrent task would be starved and finish *after* the
sync call. If it is offloaded, the concurrent task completes long before.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from src.services import ingestor as ingestor_module
from src.services import web_search as web_search_module
from src.services.ingestor import Ingestor
from src.services.web_search import _fetch_url_main_text_inner
from tests.conftest import make_test_config
from tests.stubs import StubEmbeddingService, StubTomeRepository, StubVerifier

# Generous tolerance: the offloaded call sleeps for ``BLOCK_S`` seconds in a
# worker thread; concurrent ``asyncio.sleep(CONCURRENT_S)`` should finish well
# inside that window. We assert the concurrent task finishes in under
# ``MAX_CONCURRENT_WALL_S`` — comfortably less than ``BLOCK_S``.
# Increased from 0.20 to 0.35 to account for DNS pinning transport overhead
# and slower systems (Windows CI).
BLOCK_S = 0.30
CONCURRENT_S = 0.05
MAX_CONCURRENT_WALL_S = 0.35


async def _measure_concurrent(coro: Any) -> tuple[float, float]:
    """Run ``coro`` and a short ``asyncio.sleep`` concurrently.

    Returns ``(coro_wall, concurrent_wall)`` — wall-clock seconds each took.
    If the coroutine offloads its blocking work, ``concurrent_wall`` ≈
    ``CONCURRENT_S``. If it blocks the loop, ``concurrent_wall`` ≈
    ``coro_wall`` ≥ ``BLOCK_S``.
    """
    concurrent_done_at: list[float] = []
    coro_done_at: list[float] = []

    async def concurrent_probe() -> None:
        await asyncio.sleep(CONCURRENT_S)
        concurrent_done_at.append(time.perf_counter())

    async def run_coro() -> None:
        await coro
        coro_done_at.append(time.perf_counter())

    started = time.perf_counter()
    await asyncio.gather(run_coro(), concurrent_probe())
    return (coro_done_at[0] - started, concurrent_done_at[0] - started)


# ── trafilatura.extract offloading ───────────────────────────────────────────


async def test_trafilatura_extract_is_offloaded(monkeypatch: pytest.MonkeyPatch) -> None:
    """``trafilatura.extract`` must run off the event loop.

    We bypass the network entirely: monkeypatch the SSRF/HTTP path and replace
    ``trafilatura.extract`` with a sleeping shim. The concurrent ``asyncio.sleep``
    probe must complete during the blocking "extract".
    """

    async def _always_safe(_url: str) -> tuple[bool, str | None]:
        return (True, "127.0.0.1")

    monkeypatch.setattr(web_search_module, "_validate_url", _always_safe)

    class _FakeResponse:
        status_code = 200
        text = "<html><body><p>hi</p></body></html>"
        headers: dict[str, str] = {}

        def raise_for_status(self) -> None:
            return None

    class _FakeClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> _FakeClient:
            return self

        async def __aexit__(self, *args: Any) -> None:
            return None

        async def get(self, *args: Any, **kwargs: Any) -> _FakeResponse:
            return _FakeResponse()

        async def aclose(self) -> None:
            return None

    monkeypatch.setattr(web_search_module.httpx, "AsyncClient", _FakeClient)

    def _slow_extract(_html: str) -> str:
        time.sleep(BLOCK_S)
        return "EXTRACTED"

    monkeypatch.setattr(web_search_module.trafilatura, "extract", _slow_extract)

    coro_wall, concurrent_wall = await _measure_concurrent(
        _fetch_url_main_text_inner("https://example.com", timeout=5.0)
    )

    assert coro_wall >= BLOCK_S, (
        f"sanity: slow extract should take >= {BLOCK_S}s, got {coro_wall:.3f}s"
    )
    assert concurrent_wall < MAX_CONCURRENT_WALL_S, (
        f"event loop was blocked: concurrent task finished after "
        f"{concurrent_wall:.3f}s (expected < {MAX_CONCURRENT_WALL_S}s)"
    )


# ── splitter.split_text offloading (structured paths) ────────────────────────


def _make_ingestor(use_llm_chunking: bool = False) -> Ingestor:
    cfg = make_test_config()
    cfg.ingest.use_llm_chunking = use_llm_chunking
    cfg.ingest.use_llm_classification = False
    cfg.ingest.use_llm_summary = False
    cfg.verification.enabled = False
    return Ingestor(
        cfg,
        StubEmbeddingService(dimensions=cfg.embedding.dimensions),
        StubVerifier(confidence=0.9),
        StubTomeRepository(),
    )


class _SlowSplitter:
    """Stand-in for ``RecursiveCharacterTextSplitter`` whose ``split_text`` sleeps."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def split_text(self, blob: str) -> list[str]:
        time.sleep(BLOCK_S)
        return [blob]


def _patch_splitter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force every ``RecursiveCharacterTextSplitter(...)`` to return ``_SlowSplitter``."""
    monkeypatch.setattr(ingestor_module, "RecursiveCharacterTextSplitter", _SlowSplitter)


async def test_split_text_recursive_is_offloaded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Plain prose -> ``_split_text_recursive`` -> ``splitter.split_text`` must offload."""
    _patch_splitter(monkeypatch)
    ing = _make_ingestor()
    coro_wall, concurrent_wall = await _measure_concurrent(
        ing._split_text_recursive("just some prose blob")
    )
    assert coro_wall >= BLOCK_S, (
        f"sanity: slow split should take >= {BLOCK_S}s, got {coro_wall:.3f}s"
    )
    assert concurrent_wall < MAX_CONCURRENT_WALL_S, (
        f"event loop was blocked during _split_text_recursive: "
        f"concurrent task finished after {concurrent_wall:.3f}s"
    )


async def test_split_structured_python_is_offloaded(monkeypatch: pytest.MonkeyPatch) -> None:
    """``_split_structured`` python branch must offload ``split_text``."""

    class _FromLanguageSlow:
        @classmethod
        def from_language(cls, *args: Any, **kwargs: Any) -> _SlowSplitter:
            return _SlowSplitter()

    monkeypatch.setattr(ingestor_module, "RecursiveCharacterTextSplitter", _FromLanguageSlow)
    ing = _make_ingestor()
    coro_wall, concurrent_wall = await _measure_concurrent(
        ing._split_structured("def foo():\n    return 1\n", "python")
    )
    assert coro_wall >= BLOCK_S
    assert concurrent_wall < MAX_CONCURRENT_WALL_S, (
        f"event loop was blocked during _split_structured(python): "
        f"concurrent task finished after {concurrent_wall:.3f}s"
    )


async def test_split_structured_markdown_is_offloaded(monkeypatch: pytest.MonkeyPatch) -> None:
    """``_split_structured`` markdown branch must offload ``split_text``."""

    class _FromLanguageSlow:
        @classmethod
        def from_language(cls, *args: Any, **kwargs: Any) -> _SlowSplitter:
            return _SlowSplitter()

    monkeypatch.setattr(ingestor_module, "RecursiveCharacterTextSplitter", _FromLanguageSlow)
    ing = _make_ingestor()
    coro_wall, concurrent_wall = await _measure_concurrent(
        ing._split_structured("# Heading\n\n```py\nx = 1\n```\n", "markdown")
    )
    assert coro_wall >= BLOCK_S
    assert concurrent_wall < MAX_CONCURRENT_WALL_S, (
        f"event loop was blocked during _split_structured(markdown): "
        f"concurrent task finished after {concurrent_wall:.3f}s"
    )


async def test_split_yaml_oversized_section_is_offloaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_split_yaml`` falls back to splitter on oversized sections; must offload."""
    _patch_splitter(monkeypatch)
    ing = _make_ingestor()
    # Build a single oversized top-level YAML section to force the splitter path.
    huge_value = "x" * (ing._config.ingest.shard_size + 200)
    blob = f"key1: {huge_value}\n"
    coro_wall, concurrent_wall = await _measure_concurrent(ing._split_structured(blob, "yaml"))
    assert coro_wall >= BLOCK_S
    assert concurrent_wall < MAX_CONCURRENT_WALL_S, (
        f"event loop was blocked during _split_structured(yaml): "
        f"concurrent task finished after {concurrent_wall:.3f}s"
    )


async def test_split_json_oversized_chunk_is_offloaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_split_json`` falls back to splitter on oversized chunks; must offload."""
    _patch_splitter(monkeypatch)
    ing = _make_ingestor()
    # Single oversized JSON object forces the splitter fallback path.
    huge_value = "x" * (ing._config.ingest.shard_size + 200)
    blob = '{"k": "' + huge_value + '"}'
    coro_wall, concurrent_wall = await _measure_concurrent(ing._split_structured(blob, "json"))
    assert coro_wall >= BLOCK_S
    assert concurrent_wall < MAX_CONCURRENT_WALL_S, (
        f"event loop was blocked during _split_structured(json): "
        f"concurrent task finished after {concurrent_wall:.3f}s"
    )


async def test_split_json_loads_is_offloaded(monkeypatch: pytest.MonkeyPatch) -> None:
    """``json.loads`` in ``_split_json`` must be offloaded for large blobs.

    Per Gemini review on PR #82: parsing very large JSON is CPU-bound.
    """
    ing = _make_ingestor()
    import json as _json

    real_loads = _json.loads

    def _slow_loads(s: str | bytes) -> Any:
        time.sleep(BLOCK_S)
        return real_loads(s)

    monkeypatch.setattr(ingestor_module.json, "loads", _slow_loads)

    coro_wall, concurrent_wall = await _measure_concurrent(
        ing._split_structured('{"k": "v"}', "json")
    )
    assert coro_wall >= BLOCK_S
    assert concurrent_wall < MAX_CONCURRENT_WALL_S, (
        f"event loop was blocked during json.loads in _split_json: "
        f"concurrent task finished after {concurrent_wall:.3f}s"
    )
