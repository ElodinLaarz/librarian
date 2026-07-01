"""Regression tests for issue #19: default LLM model name and startup log line.

The previous default ``gemma4:e2b`` is not a real Ollama model — every default
install silently fell back to heuristics. The defaults must be a real small
Ollama model, and a startup log line must surface the configured model so the
user can tell whether LLM claim extraction is actually wired up.
"""

from __future__ import annotations

import logging
import os

import pytest

# database.uri is required for the server module-level config load.
os.environ.setdefault("LIBRARIAN_DATABASE_URI", "mongodb://localhost:27017")

from src.config import IngestSettings, VerificationSettings


def test_verification_claim_model_default_is_real_ollama_model() -> None:
    """``claim_model`` default must be a real Ollama model, not the typo placeholder."""
    settings = VerificationSettings()
    assert settings.claim_model == "gemma2:2b", (
        f"Expected real Ollama model 'gemma2:2b', got {settings.claim_model!r}. See issue #19."
    )


def test_ingest_extraction_model_default_is_real_ollama_model() -> None:
    """``extraction_model`` default must be a real Ollama model, not the typo placeholder."""
    settings = IngestSettings()
    assert settings.extraction_model == "gemma2:2b", (
        f"Expected real Ollama model 'gemma2:2b', got {settings.extraction_model!r}. See issue #19."
    )


@pytest.mark.asyncio
async def test_lifespan_logs_llm_claim_extraction_enabled(
    caplog: pytest.LogCaptureFixture, tmp_path: object
) -> None:
    """The lifespan startup must emit a log line naming the LLM model in use."""
    # Build a fresh server bound to filesystem storage with explicit dummy
    # embeddings so the lifespan initialises without requiring a live MongoDB,
    # sentence-transformers, or Ollama ('auto' now fails fast when neither
    # real provider is available).
    from src.config import DatabaseSettings, EmbeddingSettings, LibrarianConfig
    from src.server import LibrarianServer

    cfg = LibrarianConfig(
        database=DatabaseSettings(uri=f"file://{tmp_path}"),
        embedding=EmbeddingSettings(provider="dummy", dimensions=8),
    )
    server = LibrarianServer(cfg)

    caplog.set_level(logging.INFO)

    async with server.lifespan(server.mcp):
        pass

    expected_model = cfg.verification.claim_model
    matching = [
        r
        for r in caplog.records
        if "LLM claim extraction enabled" in r.getMessage() and expected_model in r.getMessage()
    ]
    assert matching, (
        "Expected an INFO log 'LLM claim extraction enabled, model=...' on startup. "
        f"Got records: {[r.getMessage() for r in caplog.records]}"
    )


@pytest.mark.asyncio
async def test_lifespan_logs_verification_disabled(
    caplog: pytest.LogCaptureFixture, tmp_path: object
) -> None:
    """When verification is disabled, the startup log must say so explicitly."""
    from src.config import (
        DatabaseSettings,
        EmbeddingSettings,
        LibrarianConfig,
        VerificationSettings,
    )
    from src.server import LibrarianServer

    cfg = LibrarianConfig(
        database=DatabaseSettings(uri=f"file://{tmp_path}"),
        embedding=EmbeddingSettings(provider="dummy", dimensions=8),
        verification=VerificationSettings(enabled=False),
    )
    server = LibrarianServer(cfg)

    caplog.set_level(logging.INFO)

    async with server.lifespan(server.mcp):
        pass

    assert any("Verification disabled" in r.getMessage() for r in caplog.records), (
        f"Expected 'Verification disabled' log. Got: {[r.getMessage() for r in caplog.records]}"
    )


@pytest.mark.asyncio
async def test_lifespan_logs_llm_claim_extraction_disabled(
    caplog: pytest.LogCaptureFixture, tmp_path: object
) -> None:
    """When verification is on but use_llm_claims=False, the heuristic note must log."""
    from src.config import (
        DatabaseSettings,
        EmbeddingSettings,
        LibrarianConfig,
        VerificationSettings,
    )
    from src.server import LibrarianServer

    cfg = LibrarianConfig(
        database=DatabaseSettings(uri=f"file://{tmp_path}"),
        embedding=EmbeddingSettings(provider="dummy", dimensions=8),
        verification=VerificationSettings(enabled=True, use_llm_claims=False),
    )
    server = LibrarianServer(cfg)

    caplog.set_level(logging.INFO)

    async with server.lifespan(server.mcp):
        pass

    assert any("LLM claim extraction disabled" in r.getMessage() for r in caplog.records), (
        "Expected 'LLM claim extraction disabled' log when use_llm_claims=False. "
        f"Got: {[r.getMessage() for r in caplog.records]}"
    )
