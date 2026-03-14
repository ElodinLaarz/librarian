from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.config import VerificationSettings
from src.services.web_search import WebSearchClient


class ClaimResult(BaseModel):
    claim: str
    verdict: Literal["supported", "contradicted", "unverifiable"]
    evidence: str


@dataclass
class VerificationResult:
    confidence: float
    claims: list[ClaimResult]
    skipped: bool


class Verifier:
    """Quality control layer that estimates the truthfulness of incoming content
    by cross-referencing factual claims against web sources."""

    def __init__(
        self,
        settings: VerificationSettings,
        web_search: WebSearchClient,
    ) -> None:
        self._settings = settings
        self._web_search = web_search

    async def verify(self, content: str) -> VerificationResult:
        """Run the full verification pipeline on a piece of content."""
        ...

    async def _extract_claims(self, content: str) -> list[str]:
        """Extract 3-7 key factual claims using a zero-shot claim extraction prompt."""
        ...

    async def _check_claim(self, claim: str) -> ClaimResult:
        """Search the web for a single claim and score it as supported/contradicted/unverifiable."""
        ...

    def _aggregate_confidence(self, results: list[ClaimResult]) -> float:
        """Compute an aggregate confidence score from individual claim results."""
        ...

    def _make_offline_result(self) -> VerificationResult:
        """Return a synthetic 0.6-confidence result when verification is unavailable."""
        ...
