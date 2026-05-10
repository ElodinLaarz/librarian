"""LibrarianConfig.from_yaml — YAML + environment interaction."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from src import constants
from src.config import DatabaseSettings, LibrarianConfig


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


def test_database_settings_tls_requires_non_empty_cert_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LIBRARIAN_DATABASE_TLS_CERT_PATH", "")
    with pytest.raises(ValidationError, match="tls_cert_path must be non-empty"):
        DatabaseSettings(uri="mongodb://localhost:27017", tls=True, tls_cert_path="")


def test_database_settings_tls_false_allows_empty_cert_path() -> None:
    s = DatabaseSettings(uri="mongodb://localhost:27017", tls=False, tls_cert_path="")
    assert s.tls_cert_path == ""


def test_dotenv_supplies_database_when_yaml_section_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`Path.cwd()/.env` is merged (via setdefault) before nested sections are built."""
    for key in (
        "LIBRARIAN_DATABASE_URI",
        "LIBRARIAN_DATABASE_TLS",
        "LIBRARIAN_DATABASE_TLS_CERT_PATH",
    ):
        monkeypatch.delenv(key, raising=False)
    (tmp_path / ".env").write_text(
        "LIBRARIAN_DATABASE_URI=mongodb://envfile-test:27017\nLIBRARIAN_DATABASE_TLS=false\n"
    )
    (tmp_path / "cfg.yml").write_text("database: {}\n")
    monkeypatch.chdir(tmp_path)
    # Re-enable for this specific test
    monkeypatch.delenv("LIBRARIAN_SKIP_DOTENV", raising=False)
    cfg = LibrarianConfig.from_yaml(tmp_path / "cfg.yml")
    assert cfg.database.uri == "mongodb://envfile-test:27017"
    assert cfg.database.tls is False


def test_from_yaml_tls_true_without_cert_path_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "tls-no-cert.yml"
    path.write_text("database:\n  uri: mongodb://localhost:27017\n  tls: true\n")
    monkeypatch.setenv("LIBRARIAN_DATABASE_TLS_CERT_PATH", "")

    with pytest.raises(ValueError, match="tls_cert_path must be non-empty"):
        LibrarianConfig.from_yaml(path)


def test_from_yaml_rejects_non_mapping_section(tmp_path: Path) -> None:
    path = tmp_path / "bad.yml"
    path.write_text("database: not-a-mapping\n")

    with pytest.raises(ValueError, match="expected a mapping"):
        LibrarianConfig.from_yaml(path)


def test_from_yaml_reads_tidy_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LIBRARIAN_DATABASE_URI", "mongodb://localhost:27017")
    path = tmp_path / "tidy.yml"
    path.write_text("tidy:\n  threshold: 0.91\n  group_concurrency: 2\n  max_fact_frequency: 12\n")

    cfg = LibrarianConfig.from_yaml(path)

    assert cfg.tidy.threshold == pytest.approx(0.91)
    assert cfg.tidy.group_concurrency == 2
    assert cfg.tidy.max_fact_frequency == 12


def test_ingest_settings_use_canonical_shard_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LIBRARIAN_DATABASE_URI", "mongodb://localhost:27017")

    cfg = LibrarianConfig()

    assert cfg.ingest.shard_size == constants.DEFAULT_SHARD_SIZE
    assert cfg.ingest.shard_overlap == constants.DEFAULT_SHARD_OVERLAP


def test_from_yaml_interpolates_env_vars(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TEST_URI", "mongodb://interp-test:27017")
    path = tmp_path / "interp.yml"
    path.write_text("database:\n  uri: ${TEST_URI}\n")

    cfg = LibrarianConfig.from_yaml(path)
    assert cfg.database.uri == "mongodb://interp-test:27017"


def test_from_yaml_interpolates_env_vars_with_defaults(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("TEST_URI", raising=False)
    path = tmp_path / "interp_default.yml"
    path.write_text("database:\n  uri: ${TEST_URI:-mongodb://default-uri:27017}\n")

    cfg = LibrarianConfig.from_yaml(path)
    assert cfg.database.uri == "mongodb://default-uri:27017"


def test_from_yaml_raises_on_missing_env_var_without_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("MISSING_VAR", raising=False)
    path = tmp_path / "missing_var.yml"
    path.write_text("database:\n  uri: ${MISSING_VAR}\n")

    with pytest.raises(
        ValueError, match="missing environment variables without defaults: MISSING_VAR"
    ):
        LibrarianConfig.from_yaml(path)
