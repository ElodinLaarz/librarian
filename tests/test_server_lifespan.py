"""Tests for the LibrarianServer lifespan shutdown semantics.

Regression coverage for issue #23: background tasks scheduled via
``_track_background_task`` must be awaited (not just cancelled) when the
lifespan tears down, so in-flight work has a chance to finish or unwind
cleanly instead of being dropped mid-flight.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import pytest

from src.config import (
    DatabaseSettings,
    EmbeddingSettings,
    TidySettings,
    VerificationSettings,
)
from src.server import LibrarianServer
from tests.conftest import make_test_config


async def test_shutdown_awaits_background_tasks_to_completion() -> None:
    """A background task that handles CancelledError must run to completion
    before the shutdown helper returns. This guards against the previous bug
    where ``t.cancel()`` was called and the set cleared without ``await``,
    leaving in-flight work (e.g. half-written ResearchJob records) orphaned.
    """
    server = LibrarianServer(make_test_config())

    cleaned_up = asyncio.Event()

    async def slow_task_that_cleans_up_on_cancel() -> None:
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            # Simulate a brief shutdown unwind — closing handles, flushing
            # state. The shutdown helper must wait for us to get here before
            # returning.
            await asyncio.sleep(0.05)
            cleaned_up.set()
            raise

    task = asyncio.create_task(slow_task_that_cleans_up_on_cancel())
    server._track_background_task(task)

    # Give the task one tick so it is actually suspended inside the sleep
    # before shutdown cancels it.
    await asyncio.sleep(0)

    await server._shutdown_background_tasks(timeout=5.0)

    assert task.done(), "Background task must be awaited before shutdown returns"
    assert cleaned_up.is_set(), (
        "Shutdown returned before background task finished its cancellation "
        "cleanup — in-flight work would be dropped (regression of issue #23)"
    )
    assert server._bg_tasks == set(), "Tracked task set must be empty after shutdown"


async def test_shutdown_with_no_background_tasks_is_noop() -> None:
    """Shutdown must not raise when no tasks were ever started."""
    server = LibrarianServer(make_test_config())
    await server._shutdown_background_tasks(timeout=1.0)
    assert server._bg_tasks == set()


async def test_shutdown_swallows_task_exceptions(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Exceptions raised by a background task during shutdown must be logged,
    not propagated — otherwise one misbehaving task would abort the rest of
    the shutdown sequence (closing repos, etc.)."""
    server = LibrarianServer(make_test_config())

    async def task_that_raises() -> None:
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            raise RuntimeError("boom during shutdown") from None

    task = asyncio.create_task(task_that_raises())
    server._track_background_task(task)
    await asyncio.sleep(0)

    with caplog.at_level("ERROR"):
        await server._shutdown_background_tasks(timeout=5.0)

    assert task.done()
    assert any(
        "boom during shutdown" in rec.getMessage() or "background" in rec.getMessage().lower()
        for rec in caplog.records
    ), "Expected an error log mentioning the failing background task"


def _offline_fs_config(tmp_path: Path) -> object:
    """A config the lifespan can fully start with zero external services."""
    return make_test_config(
        database=DatabaseSettings(uri=str(tmp_path), tls=False),
        embedding=EmbeddingSettings(provider="dummy", dimensions=8),
        verification=VerificationSettings(enabled=False),
        tidy=TidySettings(enabled=False),
    )


async def test_lifespan_tears_down_when_body_raises(tmp_path: Path) -> None:
    """An exception thrown into the running server must still release resources.

    Regression guard: teardown used to sit after a bare ``yield`` with no
    try/finally, so any exception skipped it entirely — leaking the Motor
    client, the ingestor's httpx client, and background tasks.
    """
    server = LibrarianServer(_offline_fs_config(tmp_path))  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="boom"):
        async with server.lifespan(server.mcp):
            assert server.tome_repo is not None
            assert server.ingestor is not None
            raise RuntimeError("boom")

    assert server.tome_repo is None
    assert server.job_repo is None
    assert server.ingestor is None
    assert server.tidier is None
    assert server.researcher is None
    assert server._embedding_service is None
    assert server._bg_tasks == set()


async def test_lifespan_logs_backend_and_embedding_provider(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Startup must announce which storage backend and embedding provider are live.

    An operator reading the logs has to be able to tell whether tomes are
    going to Mongo or to a filesystem directory (e.g. after a typo'd URI
    silently selected the fallback), and which embedding model is active.
    """
    server = LibrarianServer(_offline_fs_config(tmp_path))  # type: ignore[arg-type]

    with caplog.at_level(logging.INFO):
        async with server.lifespan(server.mcp):
            pass

    messages = [rec.getMessage() for rec in caplog.records]
    assert any("Storage backend: filesystem" in m for m in messages), messages
    assert any(
        "Embedding provider: DummyEmbeddingService" in m and "dimensions=8" in m for m in messages
    ), messages
