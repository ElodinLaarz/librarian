"""Unit tests for the live-Mongo skip guard used by `tests/test_mongo_repository.py`.

These tests verify the skip mechanism itself — independent of whether a real
Mongo is running — so they always execute in CI.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys

import pytest

from tests import test_mongo_repository as mongo_module


def test_mongo_available_returns_false_for_unreachable_uri() -> None:
    """A known-bad URI must yield a clean False (not raise)."""
    # Port 1 is reserved/unused — connection will fail fast.
    assert mongo_module._mongo_available("mongodb://127.0.0.1:1/?directConnection=true") is False


def test_mongo_available_returns_false_for_garbage_uri() -> None:
    """A malformed/unresolvable host must also yield False, not raise."""
    assert (
        mongo_module._mongo_available(
            "mongodb://this-host-does-not-exist.invalid:27017/?directConnection=true"
        )
        is False
    )


def test_module_skips_when_mongo_unreachable() -> None:
    """Setting LIBRARIAN_TEST_MONGO_URI to a bad URI must cause the live-mongo tests
    to skip cleanly — no `ServerSelectionTimeoutError`, no fixture crash.

    Mock-only tests in the same module must still execute, because they don't need
    a live mongo.
    """
    bad_uri = "mongodb://127.0.0.1:1/?directConnection=true"
    env = {**os.environ, "LIBRARIAN_TEST_MONGO_URI": bad_uri}
    result = subprocess.run(  # noqa: S603 - controlled args, no shell
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_mongo_repository.py",
            "-q",
            "--no-header",
            "-p",
            "no:cacheprovider",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    combined = result.stdout + result.stderr
    assert "skipped" in combined.lower(), (
        f"expected pytest to report skips for live-mongo tests, got:\n{combined}"
    )
    assert "ServerSelectionTimeoutError" not in combined, (
        f"live-mongo tests did not skip cleanly — connection attempted:\n{combined}"
    )
    # Pytest exit codes: 0 = all passed, 5 = no tests collected.
    # With our setup we expect 0 (3 mock tests pass + 3 live tests skip).
    assert result.returncode == 0, (
        f"pytest exited non-zero ({result.returncode}); expected clean pass+skip:\n{combined}"
    )


def test_default_uri_matches_documented_value() -> None:
    """The DEFAULT_TEST_MONGO_URI constant must stay aligned with what's documented."""
    assert mongo_module.DEFAULT_TEST_MONGO_URI == (
        "mongodb://localhost:27017/?directConnection=true"
    )


def test_env_var_overrides_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Setting LIBRARIAN_TEST_MONGO_URI must change the URI used by the module on reload."""
    monkeypatch.setenv("LIBRARIAN_TEST_MONGO_URI", "mongodb://example.invalid:1234/")
    reloaded = importlib.reload(mongo_module)
    try:
        assert reloaded.TEST_MONGO_URI == "mongodb://example.invalid:1234/"
    finally:
        # Restore for any other tests in the same session.
        monkeypatch.delenv("LIBRARIAN_TEST_MONGO_URI", raising=False)
        importlib.reload(mongo_module)
