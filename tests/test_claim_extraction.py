"""Tests for claim_extraction.py — heuristic and LLM paths."""

from __future__ import annotations

import json

import pytest

from src import constants
from src.config import VerificationSettings
from src.services.claim_extraction import _heuristic_claims, extract_claims
from tests.conftest import make_test_config

# ── _heuristic_claims ────────────────────────────────────────────────────────


def test_heuristic_splits_on_sentence_boundaries() -> None:
    # Each sentence must be > 20 chars to survive the length filter
    text = (
        "Photosynthesis converts sunlight into glucose. "
        "Water molecules are split during light reactions. "
        "Carbon dioxide is fixed during the Calvin cycle."
    )
    claims = _heuristic_claims(text)
    assert len(claims) == 3


def test_heuristic_filters_short_fragments() -> None:
    # Fragments under 20 chars are dropped; the long one is kept
    claims = _heuristic_claims(
        "Hi. This is a sufficiently long sentence that passes the minimum length filter."
    )
    assert all(len(c) > 20 for c in claims)
    assert len(claims) == 1


def test_heuristic_empty_returns_empty() -> None:
    assert _heuristic_claims("") == []
    assert _heuristic_claims("   ") == []


def test_heuristic_respects_max_claims() -> None:
    # Build 20 long sentences — output should be capped at MAX_CLAIMS
    sentences = [
        f"This is a long enough sentence number {i} about a very interesting topic."
        for i in range(20)
    ]
    claims = _heuristic_claims(" ".join(sentences))
    assert len(claims) <= constants.MAX_CLAIMS


def test_heuristic_returns_single_chunk_for_unpunctuated_text() -> None:
    # Text with no sentence-ending punctuation and length > 20 → one chunk
    text = "a word " * 30  # 210 chars, no . ! ?
    claims = _heuristic_claims(text)
    assert len(claims) == 1


def test_heuristic_falls_back_to_blob_when_all_fragments_too_short() -> None:
    # All split fragments < 20 chars → fallback returns [text[:500]]
    text = "Hi. Ok. Yes. No."  # all splits are tiny
    claims = _heuristic_claims(text)
    # Falls back to [text[:500]] because no fragment survives the filter
    assert len(claims) == 1
    assert claims[0] == text[:500]


# ── extract_claims — LLM disabled ────────────────────────────────────────────


async def test_extract_claims_uses_heuristic_when_llm_disabled() -> None:
    config = make_test_config(
        verification=VerificationSettings(use_llm_claims=False)
    )
    claims = await extract_claims(
        "Photosynthesis converts sunlight into sugar. Chlorophyll absorbs light energy.", config
    )
    assert len(claims) >= 1


# ── extract_claims — LLM path failures ───────────────────────────────────────


async def test_extract_claims_falls_back_when_llm_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the Ollama HTTP call fails, extract_claims falls back to heuristic."""
    import httpx

    async def _raise(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr("httpx.AsyncClient.post", _raise)

    config = make_test_config(verification=VerificationSettings(use_llm_claims=True))
    claims = await extract_claims(
        "Photosynthesis converts light into sugar. Chlorophyll is green.", config
    )
    assert len(claims) >= 1


async def test_extract_claims_falls_back_on_malformed_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LLM returning non-JSON should trigger heuristic fallback."""
    from unittest.mock import AsyncMock, MagicMock

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json = MagicMock(
        return_value={"choices": [{"message": {"content": "not valid json {"}}]}
    )

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_response)

    monkeypatch.setattr("httpx.AsyncClient", lambda **kwargs: mock_client)

    config = make_test_config(verification=VerificationSettings(use_llm_claims=True))
    claims = await extract_claims(
        "Photosynthesis converts light into sugar. Chlorophyll is green.", config
    )
    assert len(claims) >= 1


async def test_extract_claims_uses_llm_when_valid_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When Ollama returns valid JSON with enough claims, use them."""
    from unittest.mock import AsyncMock, MagicMock

    llm_claims = ["Claim one is true.", "Claim two is also true.", "Claim three confirmed."]
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json = MagicMock(
        return_value={
            "choices": [{"message": {"content": json.dumps({"claims": llm_claims})}}]
        }
    )

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_response)

    monkeypatch.setattr("httpx.AsyncClient", lambda **kwargs: mock_client)

    config = make_test_config(verification=VerificationSettings(use_llm_claims=True))
    claims = await extract_claims("anything", config)
    assert claims == llm_claims
