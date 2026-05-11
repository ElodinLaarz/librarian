"""Issue #18 — default LIBRARIAN_CONFIG path matches docs.

Guards:
- The default config path string is ``librarian.config.yaml`` (matches README).
- When no config is found and no env var is set, the error mentions the
  ``librarian.config.example.yaml`` so users have a clear next step.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


def _reload_server_module(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Reimport src.server with a clean cwd / env so module-level load runs again."""
    monkeypatch.chdir(tmp_path)
    # Ensure DB uri is available so the YAML-not-found code path is the only failure mode.
    monkeypatch.setenv("LIBRARIAN_DATABASE_URI", "mongodb://localhost:27017")
    for mod in ("src.server", "src.__main__"):
        sys.modules.pop(mod, None)
    return importlib.import_module("src.server")


def test_default_config_path_matches_docs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The default config path used when ``LIBRARIAN_CONFIG`` is unset must be
    ``librarian.config.yaml`` so the README's instructions actually work."""
    monkeypatch.delenv("LIBRARIAN_CONFIG", raising=False)
    server = _reload_server_module(monkeypatch, tmp_path)
    expected = Path("librarian.config.yaml")
    assert expected == server.DEFAULT_CONFIG_PATH


def test_friendly_error_when_no_config_found(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When no config exists and required fields are missing, the error must
    name ``librarian.config.example.yaml`` so the user has an obvious fix."""
    monkeypatch.delenv("LIBRARIAN_CONFIG", raising=False)
    monkeypatch.delenv("LIBRARIAN_DATABASE_URI", raising=False)
    monkeypatch.setenv("LIBRARIAN_SKIP_DOTENV", "1")
    monkeypatch.chdir(tmp_path)
    for mod in ("src.server", "src.__main__"):
        sys.modules.pop(mod, None)
    with pytest.raises(ValueError, match=r"librarian\.config\.example\.yaml"):
        importlib.import_module("src.server")
