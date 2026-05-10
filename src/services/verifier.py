from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

import httpx

from src.config import LibrarianConfig
from src.models.enums import VerificationVerdict
from src.services.claim_extraction import extract_claims
from src.services.web_search import WebSearchClient, WebSearchResult


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

        verification = self._config.verification
        verdict: VerificationVerdict | None = None
        if verification.use_llm_verdict:
            model = verification.verdict_model or verification.claim_model
            if model:
                verdict = await _llm_verdict(claim, hits, self._config)

        if verdict is None:
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

_VERDICT_ALIASES = {
    "supported": VerificationVerdict.SUPPORTED,
    "support": VerificationVerdict.SUPPORTED,
    "true": VerificationVerdict.SUPPORTED,
    "contradicted": VerificationVerdict.CONTRADICTED,
    "contradict": VerificationVerdict.CONTRADICTED,
    "false": VerificationVerdict.CONTRADICTED,
    "refuted": VerificationVerdict.CONTRADICTED,
    "unverifiable": VerificationVerdict.UNVERIFIABLE,
    "unknown": VerificationVerdict.UNVERIFIABLE,
    "insufficient": VerificationVerdict.UNVERIFIABLE,
}


async def _llm_verdict(
    claim: str,
    hits: list[WebSearchResult],
    config: LibrarianConfig,
) -> VerificationVerdict | None:
    """Ask the configured LLM for a strict JSON verdict on a single claim.

    Returns None on any HTTP, parsing, or schema failure so the caller can
    fall back to the heuristic.
    """
    verification = config.verification
    base = verification.ollama_base_url.rstrip("/")
    model = verification.verdict_model or verification.claim_model
    if not model:
        return None

    evidence_snippets = [{"title": h.title, "snippet": h.snippet, "url": h.url} for h in hits[:3]]
    prompt = (
        "You are a fact-checking verifier. Given a CLAIM and up to three EVIDENCE "
        "snippets retrieved from the web, decide whether the evidence supports, "
        "contradicts, or is insufficient to verify the claim. "
        "Snippets that merely echo the claim's wording without adding independent "
        "facts are NOT sufficient support — return 'unverifiable' in that case. "
        'Output JSON only with shape {"verdict":"supported|contradicted|unverifiable",'
        '"reasoning":"..."}.\n\n'
        f"CLAIM:\n{claim}\n\nEVIDENCE:\n{json.dumps(evidence_snippets, ensure_ascii=False)}"
    )
    url = f"{base}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
    }
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
    except (httpx.HTTPError, OSError, ValueError, KeyError) as exc:
        logging.debug("_llm_verdict HTTP/parse error: %s", exc)
        return None

    try:
        message = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None

    message = message.strip()
    if message.startswith("```"):
        message = re.sub(r"^```(?:json)?\s*", "", message)
        message = re.sub(r"\s*```$", "", message)

    try:
        parsed = json.loads(message)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None

    if not isinstance(parsed, dict):
        return None
    raw_verdict = parsed.get("verdict")
    if not isinstance(raw_verdict, str):
        return None
    return _VERDICT_ALIASES.get(raw_verdict.strip().lower())


def _verdict_from_snippets(claim: str, combined_snippets: str) -> VerificationVerdict:
    """Lightweight snippet heuristic (fallback when LLM verdict unavailable).

    Biased toward UNVERIFIABLE: SUPPORTED requires multi-word overlap, no
    negation tokens, AND the snippet must extend meaningfully beyond the
    claim (otherwise it's just an echo of the search query).
    """
    claim_words = {w for w in re.findall(r"[a-zA-Z]{4,}", claim.lower())}
    text_lower = combined_snippets.lower()

    if _NEGATION_PATTERN.search(combined_snippets) and claim_words:
        overlap = sum(1 for w in claim_words if w in text_lower)
        if overlap >= 1:
            return VerificationVerdict.CONTRADICTED

    overlap = sum(1 for w in claim_words if w in text_lower)
    if (
        overlap >= 2
        and not _NEGATION_PATTERN.search(combined_snippets)
        and len(combined_snippets) > 2 * len(claim)
    ):
        return VerificationVerdict.SUPPORTED
    return VerificationVerdict.UNVERIFIABLE
