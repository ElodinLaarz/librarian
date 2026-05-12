"""Tests for the web-backed Verifier."""

from __future__ import annotations

from typing import Any

import pytest

from src.config import VerificationSettings
from src.models.enums import VerificationVerdict
from src.services import verifier as verifier_module
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
    # Stricter heuristic: snippet that merely echoes the claim (no extra evidence)
    # should bias UNVERIFIABLE, not SUPPORTED.
    verdict = _verdict_from_snippets(
        "water boils at 100 degrees",
        "Water boils at 100 degrees Celsius at sea level",
    )
    assert verdict == VerificationVerdict.UNVERIFIABLE


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
        "earth orbits",
        "earth orbits the sun in approximately 365 days at an average speed of 30 km per second",
    )
    assert verdict == VerificationVerdict.SUPPORTED


# ── LLM verdict path ─────────────────────────────────────────────────────────


class _StubResponse:
    def __init__(
        self,
        payload: dict[str, Any] | None = None,
        *,
        raw_text: str | None = None,
    ) -> None:
        self._payload = payload
        self._raw_text = raw_text

    def raise_for_status(self) -> None:  # pragma: no cover - trivial
        return None

    def json(self) -> dict[str, Any]:
        if self._payload is None:
            raise ValueError("invalid json")
        return self._payload


class _StubClient:
    """Minimal async-context-manager stand-in for httpx.AsyncClient."""

    def __init__(
        self,
        response: _StubResponse,
        *,
        calls: list[dict[str, Any]] | None = None,
    ) -> None:
        self._response = response
        self._calls = calls if calls is not None else []

    async def __aenter__(self) -> _StubClient:
        return self

    async def __aexit__(self, *_: Any) -> None:
        return None

    async def post(self, url: str, json: dict[str, Any]) -> _StubResponse:  # noqa: A002 - mirror httpx api
        self._calls.append({"url": url, "json": json})
        return self._response


def _llm_chat_payload(verdict: str, reasoning: str = "") -> dict[str, Any]:
    body = '{"verdict":"' + verdict + '","reasoning":"' + reasoning + '"}'
    return {"choices": [{"message": {"content": body}}]}


@pytest.mark.asyncio
async def test_verdict_unverifiable_when_snippet_just_echoes_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LLM says unverifiable when snippet only echoes claim words."""
    config = make_test_config(
        verification=VerificationSettings(
            enabled=True, use_llm_claims=False, use_llm_verdict=True, claim_model="stub-model"
        ),
    )

    calls: list[dict[str, Any]] = []
    response = _StubResponse(_llm_chat_payload("unverifiable"))

    def _factory(*_: Any, **__: Any) -> _StubClient:
        return _StubClient(response, calls=calls)

    monkeypatch.setattr(verifier_module.httpx, "AsyncClient", _factory)

    web = _FakeWeb("Water boils at 100 degrees Celsius at sea level")
    v = Verifier(config, web)
    r = await v.verify("Water boils at 100 degrees Celsius.")

    assert r.skipped is False
    assert calls, "expected LLM verdict endpoint to be called"
    assert all(c.verdict == VerificationVerdict.UNVERIFIABLE for c in r.claims)


@pytest.mark.asyncio
async def test_verdict_llm_path_falls_back_on_json_parse_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-JSON LLM body -> fall back to (tightened) heuristic."""
    config = make_test_config(
        verification=VerificationSettings(
            enabled=True, use_llm_claims=False, use_llm_verdict=True, claim_model="stub-model"
        ),
    )

    bad_body = {"choices": [{"message": {"content": "not json at all"}}]}
    response = _StubResponse(bad_body)

    def _factory(*_: Any, **__: Any) -> _StubClient:
        return _StubClient(response)

    monkeypatch.setattr(verifier_module.httpx, "AsyncClient", _factory)

    # Snippet has high overlap and no negation tokens, but only echoes the claim.
    # Under the tightened heuristic this is UNVERIFIABLE.
    web = _FakeWeb("Water boils at 100 degrees Celsius at sea level")
    v = Verifier(config, web)
    r = await v.verify("Water boils at 100 degrees Celsius.")

    assert r.skipped is False
    assert all(c.verdict == VerificationVerdict.UNVERIFIABLE for c in r.claims)


@pytest.mark.asyncio
async def test_verdict_contradicted_via_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """LLM 'contradicted' wins even without negation tokens in the snippet."""
    config = make_test_config(
        verification=VerificationSettings(
            enabled=True, use_llm_claims=False, use_llm_verdict=True, claim_model="stub-model"
        ),
    )

    response = _StubResponse(_llm_chat_payload("contradicted"))

    def _factory(*_: Any, **__: Any) -> _StubClient:
        return _StubClient(response)

    monkeypatch.setattr(verifier_module.httpx, "AsyncClient", _factory)

    # Snippet has overlap, no negation tokens — heuristic alone wouldn't flag it.
    web = _FakeWeb(
        "Vaccines cause autism is a topic that has been studied extensively in research."
    )
    v = Verifier(config, web)
    r = await v.verify("Vaccines cause autism.")

    assert r.skipped is False
    assert r.claims
    assert all(c.verdict == VerificationVerdict.CONTRADICTED for c in r.claims)


@pytest.mark.asyncio
async def test_use_llm_verdict_flag_disables_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """When use_llm_verdict=False, no LLM call is made."""
    config = make_test_config(
        verification=VerificationSettings(
            enabled=True, use_llm_claims=False, use_llm_verdict=False, claim_model="stub-model"
        ),
    )

    def _factory(*_: Any, **__: Any) -> _StubClient:  # pragma: no cover - must not run
        raise AssertionError("LLM verdict path must not be invoked when flag is off")

    monkeypatch.setattr(verifier_module.httpx, "AsyncClient", _factory)

    web = _FakeWeb("evidence supported here")
    v = Verifier(config, web)
    r = await v.verify("Evidence supported claim here.")
    assert r.skipped is False


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


# ── VerificationResult.to_summary (issue #41) ────────────────────────────────


def test_to_summary_counts_each_verdict_kind() -> None:
    """Summary counts SUPPORTED / CONTRADICTED / UNVERIFIABLE separately."""
    from src.services.verifier import VerificationResult

    result = VerificationResult(
        confidence=0.7,
        claims=[
            ClaimResult(
                claim="c1",
                verdict=VerificationVerdict.SUPPORTED,
                evidence="https://example.com/a",
            ),
            ClaimResult(
                claim="c2",
                verdict=VerificationVerdict.SUPPORTED,
                evidence="https://example.com/b",
            ),
            ClaimResult(
                claim="c3",
                verdict=VerificationVerdict.CONTRADICTED,
                evidence="https://example.com/c",
            ),
            ClaimResult(
                claim="c4",
                verdict=VerificationVerdict.UNVERIFIABLE,
                evidence="no_results",
            ),
        ],
        skipped=False,
    )

    summary = result.to_summary()
    assert summary.skipped is False
    assert summary.claim_count == 4
    assert summary.supported == 2
    assert summary.contradicted == 1
    assert summary.unverifiable == 1
    # Only http(s) evidence makes it into evidence_urls; sentinels are skipped.
    assert summary.evidence_urls == [
        "https://example.com/a",
        "https://example.com/b",
        "https://example.com/c",
    ]


def test_to_summary_caps_evidence_urls_at_limit() -> None:
    """Evidence URL list must not exceed VERIFICATION_EVIDENCE_URL_LIMIT."""
    from src.models.tome import VERIFICATION_EVIDENCE_URL_LIMIT
    from src.services.verifier import VerificationResult

    claims = [
        ClaimResult(
            claim=f"c{i}",
            verdict=VerificationVerdict.SUPPORTED,
            evidence=f"https://example.com/{i}",
        )
        for i in range(VERIFICATION_EVIDENCE_URL_LIMIT + 5)
    ]
    summary = VerificationResult(confidence=0.5, claims=claims, skipped=False).to_summary()

    assert len(summary.evidence_urls) == VERIFICATION_EVIDENCE_URL_LIMIT


def test_to_summary_marks_skipped_with_no_claims() -> None:
    """``skipped=True`` propagates and claim counts are zero."""
    from src.services.verifier import VerificationResult

    summary = VerificationResult(confidence=0.5, claims=[], skipped=True).to_summary()
    assert summary.skipped is True
    assert summary.claim_count == 0
    assert summary.supported == 0
    assert summary.contradicted == 0
    assert summary.unverifiable == 0
    assert summary.evidence_urls == []
