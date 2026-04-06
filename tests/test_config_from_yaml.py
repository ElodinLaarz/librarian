"""LibrarianConfig.from_yaml — YAML + environment interaction."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import LibrarianConfig


def test_from_yaml_missing_file_uses_database_uri_from_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Empty/missing YAML must not block required fields supplied only via env."""
    missing = tmp_path / "no-such-config.yml"
    assert not missing.exists()

    monkeypatch.setenv("LIBRARIAN_DATABASE_URI", "mongodb://localhost:27017/?env-only=1")
    cfg = LibrarianConfig.from_yaml(missing)
    assert cfg.database.uri == "mongodb://localhost:27017/?env-only=1"


def test_from_yaml_partial_database_merges_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """YAML partial section: missing required uri comes from environment."""
    path = tmp_path / "partial.yml"
    path.write_text("database:\n  tls: false\n")

    monkeypatch.setenv("LIBRARIAN_DATABASE_URI", "mongodb://merge-test:27017")
    cfg = LibrarianConfig.from_yaml(path)
    assert cfg.database.uri == "mongodb://merge-test:27017"
    assert cfg.database.tls is False


def test_from_yaml_rejects_non_mapping_section(tmp_path: Path) -> None:
    path = tmp_path / "bad.yml"
    path.write_text("database: not-a-mapping\n")

    with pytest.raises(ValueError, match="expected a mapping"):
        LibrarianConfig.from_yaml(path)
