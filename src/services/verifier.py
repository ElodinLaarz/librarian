from dataclasses import dataclass

from src.config import LibrarianConfig
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

    def __init__(self, config: LibrarianConfig) -> None:
        self._config = config

    async def verify(self, content: str) -> VerificationResult:
        """Run the full verification pipeline on a piece of content."""
        # TODO: Implement full web search verification in a future PR
        return self._make_offline_result()

    async def _extract_claims(self, content: str) -> list[str]:
        """Extract a few key factual claims using a zero-shot claim extraction prompt."""
        # The number of claims should be between constants.MIN_CLAIMS and constants.MAX_CLAIMS.
        raise NotImplementedError

    async def _check_claim(self, claim: str) -> ClaimResult:
        """Search the web for a single claim and score it as supported/contradicted/unverifiable."""
        raise NotImplementedError

    def _aggregate_confidence(self, results: list[ClaimResult]) -> float:
        """Compute an aggregate confidence score from individual claim results."""
        raise NotImplementedError

    def _make_offline_result(self) -> VerificationResult:
        """Return a synthetic confidence result when verification is unavailable."""
        return VerificationResult(
            confidence=self._config.verification.mock_confidence, claims=[], skipped=True
        )

    @staticmethod
    def noop(config: LibrarianConfig) -> "Verifier":
        """Return a no-op Verifier that skips all verification and returns full confidence."""

        class _NoopVerifier(Verifier):
            async def verify(self, content: str) -> VerificationResult:
                return VerificationResult(
                    confidence=self._config.verification.noop_confidence,
                    claims=[],
                    skipped=True,
                )

        return _NoopVerifier(config)
