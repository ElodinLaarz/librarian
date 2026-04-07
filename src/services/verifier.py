from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from src.config import LibrarianConfig
from src.models.enums import VerificationVerdict
from src.services.claim_extraction import extract_claims
from src.services.web_search import WebSearchClient


@dataclass
class ClaimResult:
    claim: str
    verdict: VerificationVerdict
    evidence: str


@dataclass
class VerificationResult:
    confidence: float
    claims: list[ClaimResult]
    skipped: bool


class Verifier:
    """Quality control layer: estimate truthfulness via web search snippets."""

    def __init__(
        self,
        config: LibrarianConfig,
        web_client: WebSearchClient | None = None,
    ) -> None:
        self._config = config
        self._web = web_client

    async def verify(self, content: str) -> VerificationResult:
        """Run the full verification pipeline on a piece of content."""
        if not self._config.verification.enabled:
            return VerificationResult(
                confidence=self._config.ingest.unverified_confidence,
                claims=[],
                skipped=True,
            )

        if self._web is None or not self._web.is_available():
            return self._make_offline_result()

        claims = await extract_claims(content, self._config)
        if not claims:
            return self._make_offline_result()

        results: list[ClaimResult] = []
        for claim in claims:
            results.append(await self._check_claim(claim))

        confidence = self._aggregate_confidence(results)
        return VerificationResult(confidence=confidence, claims=results, skipped=False)

    async def _check_claim(self, claim: str) -> ClaimResult:
        """Search the web for a single claim and score it from snippets."""
        web = self._web
        assert web is not None
        try:
            hits = await web.search(claim, max_results=3)
        except Exception as exc:
            logging.warning("Web search failed for claim: %s", exc)
            return ClaimResult(
                claim=claim,
                verdict=VerificationVerdict.UNVERIFIABLE,
                evidence="search_failed",
            )

        if not hits:
            return ClaimResult(
                claim=claim,
                verdict=VerificationVerdict.UNVERIFIABLE,
                evidence="no_results",
            )

        blob = " ".join(f"{h.title} {h.snippet}" for h in hits)
        verdict = _verdict_from_snippets(claim, blob)
        evidence = hits[0].url[:500]
        return ClaimResult(claim=claim, verdict=verdict, evidence=evidence)

    def _aggregate_confidence(self, results: list[ClaimResult]) -> float:
        """Map claim verdicts to a single score in [0, 1]."""
        if not results:
            return self._config.verification.mock_confidence

        impact = {
            VerificationVerdict.SUPPORTED: 0.12,
            VerificationVerdict.CONTRADICTED: -0.22,
        }
        score = 0.5
        for r in results:
            score += impact.get(r.verdict, 0.0)
        return max(0.0, min(1.0, score))

    def _make_offline_result(self) -> VerificationResult:
        """Synthetic confidence when verification is unavailable."""
        return VerificationResult(
            confidence=self._config.verification.mock_confidence, claims=[], skipped=True
        )

    @staticmethod
    def noop(config: LibrarianConfig, web_client: WebSearchClient | None = None) -> Verifier:
        """Verifier that skips checks and returns full confidence."""

        class _NoopVerifier(Verifier):
            async def verify(self, content: str) -> VerificationResult:
                return VerificationResult(
                    confidence=config.verification.noop_confidence,
                    claims=[],
                    skipped=True,
                )

        return _NoopVerifier(config, web_client)


_NEGATION_PATTERN = re.compile(
    r"\b(false|incorrect|not true|myth|debunked|contradicts|disproven|no evidence)\b",
    re.IGNORECASE,
)


def _verdict_from_snippets(claim: str, combined_snippets: str) -> VerificationVerdict:
    """Lightweight snippet heuristic (no second LLM call)."""
    claim_words = {w for w in re.findall(r"[a-zA-Z]{4,}", claim.lower())}
    text_lower = combined_snippets.lower()

    if _NEGATION_PATTERN.search(combined_snippets) and claim_words:
        overlap = sum(1 for w in claim_words if w in text_lower)
        if overlap >= 1:
            return VerificationVerdict.CONTRADICTED

    overlap = sum(1 for w in claim_words if w in text_lower)
    if overlap >= 2:
        return VerificationVerdict.SUPPORTED
    if overlap >= 1:
        return VerificationVerdict.UNVERIFIABLE
    return VerificationVerdict.UNVERIFIABLE
