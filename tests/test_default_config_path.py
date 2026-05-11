"""Issue #18 — default LIBRARIAN_CONFIG path matches docs.

Guards:
- The default config path string is ``librarian.config.yaml`` (matches README).
- When no config is found and no env var is set, the error mentions the
  ``librarian.config.example.yaml`` so users have a clear next step.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def test_default_config_path_matches_docs() -> None:
    """The default config path used when ``LIBRARIAN_CONFIG`` is unset must be
    ``librarian.config.yaml`` so the README's instructions actually work."""
    from src.server import DEFAULT_CONFIG_PATH

    assert Path("librarian.config.yaml") == DEFAULT_CONFIG_PATH


def test_friendly_error_when_no_config_found(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When no config exists and required fields are missing, the error must
    name ``librarian.config.example.yaml`` so the user has an obvious fix."""
    monkeypatch.delenv("LIBRARIAN_CONFIG", raising=False)
    monkeypatch.delenv("LIBRARIAN_DATABASE_URI", raising=False)
    monkeypatch.setenv("LIBRARIAN_SKIP_DOTENV", "1")
    monkeypatch.chdir(tmp_path)

    from src.server import load_config

    with pytest.raises(ValueError, match=r"librarian\.config\.example\.yaml"):
        load_config()
