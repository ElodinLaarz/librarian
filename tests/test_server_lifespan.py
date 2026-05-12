"""Tests for the LibrarianServer lifespan shutdown semantics.

Regression coverage for issue #23: background tasks scheduled via
``_track_background_task`` must be awaited (not just cancelled) when the
lifespan tears down, so in-flight work has a chance to finish or unwind
cleanly instead of being dropped mid-flight.
"""

from __future__ import annotations

import asyncio

import pytest

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
