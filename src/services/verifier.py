from dataclasses import dataclass

from src.models.enums import VerificationVerdict


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
    """Quality control layer that estimates the truthfulness of incoming content
    by cross-referencing factual claims against web sources."""

    async def verify(self, content: str) -> VerificationResult:
        """Run the full verification pipeline on a piece of content."""
        # TODO: Implement full web search verification in a future PR
        return self._make_offline_result()

    async def _extract_claims(self, content: str) -> list[str]:
        """Extract 3-7 key factual claims using a zero-shot claim extraction prompt."""
        raise NotImplementedError

    async def _check_claim(self, claim: str) -> ClaimResult:
        """Search the web for a single claim and score it as supported/contradicted/unverifiable."""
        raise NotImplementedError

    def _aggregate_confidence(self, results: list[ClaimResult]) -> float:
        """Compute an aggregate confidence score from individual claim results."""
        raise NotImplementedError

    def _make_offline_result(self) -> VerificationResult:
        """Return a synthetic 0.6-confidence result when verification is unavailable."""
        return VerificationResult(confidence=0.6, claims=[], skipped=True)

    @staticmethod
    def noop() -> "Verifier":
        """Return a no-op Verifier that skips all verification and returns full confidence."""

        class _NoopVerifier(Verifier):
            async def verify(self, content: str) -> VerificationResult:
                return VerificationResult(confidence=1.0, claims=[], skipped=True)

        return _NoopVerifier()
