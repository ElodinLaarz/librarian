"""Tests for the web-backed Verifier."""

from __future__ import annotations

import pytest

from src.config import VerificationSettings
from src.models.enums import VerificationVerdict
from src.services.verifier import (
    ClaimResult,
    Verifier,
    _verdict_from_snippets,
)
from src.services.web_search import (
    UnavailableWebSearchClient,
    WebSearchClient,
    WebSearchResult,
)
from tests.conftest import make_test_config


class _FakeWeb(WebSearchClient):
    def __init__(self, snippet: str) -> None:
        self._snippet = snippet

    async def search(self, query: str, max_results: int = 3) -> list[WebSearchResult]:
        return [
            WebSearchResult(
                title="ref",
                url="https://example.com/a",
                snippet=self._snippet,
            )
        ]

    def is_available(self) -> bool:
        return True


# ── _verdict_from_snippets ───────────────────────────────────────────────────


def test_verdict_supported_when_claim_words_overlap() -> None:
    verdict = _verdict_from_snippets(
        "water boils at 100 degrees",
        "Water boils at 100 degrees Celsius at sea level",
    )
    assert verdict == VerificationVerdict.SUPPORTED


def test_verdict_unverifiable_when_no_overlap() -> None:
    verdict = _verdict_from_snippets("quantum entanglement", "recipes for chocolate cake")
    assert verdict == VerificationVerdict.UNVERIFIABLE


def test_verdict_contradicted_when_negation_and_overlap() -> None:
    verdict = _verdict_from_snippets(
        "vaccines cause autism",
        "This claim is false: no evidence vaccines cause autism; myth debunked repeatedly",
    )
    assert verdict == VerificationVerdict.CONTRADICTED


def test_verdict_supported_without_negation_words() -> None:
    verdict = _verdict_from_snippets(
        "earth orbits the sun",
        "earth orbits the sun in approximately 365 days",
    )
    assert verdict == VerificationVerdict.SUPPORTED


# ── _aggregate_confidence ────────────────────────────────────────────────────


def _claim(verdict: VerificationVerdict) -> ClaimResult:
    return ClaimResult(claim="stub", verdict=verdict, evidence="stub")


def test_aggregate_all_supported_raises_confidence() -> None:
    config = make_test_config()
    score = Verifier(config)._aggregate_confidence(
        [_claim(VerificationVerdict.SUPPORTED) for _ in range(3)]
    )
    assert score > 0.5


def test_aggregate_contradicted_lowers_confidence() -> None:
    config = make_test_config()
    score = Verifier(config)._aggregate_confidence(
        [_claim(VerificationVerdict.CONTRADICTED) for _ in range(3)]
    )
    assert score < 0.5


def test_aggregate_empty_returns_mock_confidence() -> None:
    config = make_test_config()
    score = Verifier(config)._aggregate_confidence([])
    assert score == pytest.approx(config.verification.mock_confidence)


def test_aggregate_clamps_to_zero() -> None:
    config = make_test_config()
    score = Verifier(config)._aggregate_confidence(
        [_claim(VerificationVerdict.CONTRADICTED) for _ in range(10)]
    )
    assert score == pytest.approx(0.0)


# ── Verifier.verify — offline / disabled paths ───────────────────────────────


@pytest.mark.asyncio
async def test_verifier_skips_when_search_unavailable() -> None:
    config = make_test_config(verification=VerificationSettings(enabled=True))
    v = Verifier(config, UnavailableWebSearchClient())
    r = await v.verify("Anything.")
    assert r.skipped is True
    assert r.confidence == pytest.approx(config.verification.mock_confidence)


@pytest.mark.asyncio
async def test_verifier_skips_when_no_web_client() -> None:
    config = make_test_config()
    r = await Verifier(config, web_client=None).verify("The sky is blue.")
    assert r.skipped is True


@pytest.mark.asyncio
async def test_verifier_skips_when_verification_disabled() -> None:
    config = make_test_config(verification=VerificationSettings(enabled=False))
    r = await Verifier(config).verify("Anything goes.")
    assert r.skipped is True
    assert r.confidence == pytest.approx(config.ingest.unverified_confidence)


# ── Verifier.verify — live path (stub web) ───────────────────────────────────


@pytest.mark.asyncio
async def test_verifier_scores_supported_claims() -> None:
    config = make_test_config(
        verification=VerificationSettings(enabled=True, use_llm_claims=False),
    )
    web = _FakeWeb("Photosynthesis converts light energy into chemical energy in plants.")
    v = Verifier(config, web)
    r = await v.verify(
        "Photosynthesis converts light energy into chemical energy in plants and algae."
    )
    assert r.skipped is False
    assert r.confidence >= 0.5


@pytest.mark.asyncio
async def test_verifier_confidence_is_bounded() -> None:
    config = make_test_config(verification=VerificationSettings(enabled=True, use_llm_claims=False))
    r = await Verifier(config, _FakeWeb("evidence supported here")).verify(
        "Evidence supported claim here."
    )
    assert 0.0 <= r.confidence <= 1.0


# ── Verifier.noop ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_noop_verifier_always_skips() -> None:
    config = make_test_config()
    r = await Verifier.noop(config).verify("anything")
    assert r.skipped is True
    assert r.confidence == pytest.approx(config.verification.noop_confidence)
