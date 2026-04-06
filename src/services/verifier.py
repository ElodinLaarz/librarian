import asyncio
from dataclasses import dataclass

import instructor
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from src import constants
from src.config import LibrarianConfig
from src.models.enums import VerificationVerdict
from src.services.web_search import BraveWebSearchClient


class ClaimsList(BaseModel):
    claims: list[str] = Field(
        ..., description="List of key factual claims extracted from the content."
    )


class ClaimEvaluation(BaseModel):
    verdict: VerificationVerdict
    evidence: str = Field(
        ..., description="A short summary of the evidence found or why it is unverifiable."
    )





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
        self._search_client = BraveWebSearchClient(config)
        self._llm_client = instructor.from_openai(
            AsyncOpenAI(
                base_url=self._config.llm.base_url,
                api_key=self._config.llm.api_key,
            )
        )



    async def verify(self, content: str) -> VerificationResult:
        """Run the full verification pipeline on a piece of content."""
        if not self._config.verification.enabled:
            return VerificationResult(confidence=1.0, claims=[], skipped=True)

        claims = await self._extract_claims(content)
        if not claims:
            return self._make_offline_result()

        results = await asyncio.gather(*[self._check_claim(claim) for claim in claims])


        confidence = self._aggregate_confidence(results)

        return VerificationResult(
            confidence=confidence,
            claims=results,
            skipped=False,
        )


    async def _extract_claims(self, content: str) -> list[str]:
        """Extract a few key factual claims using a zero-shot claim extraction prompt."""
        try:
            prompt = (
                f"Extract {constants.MIN_CLAIMS} to {constants.MAX_CLAIMS} key factual claims "
                f"from the following text that can be verified independently:\n\n{content}"
            )
            response = await self._llm_client.chat.completions.create(
                model=self._config.llm.model_name,
                response_model=ClaimsList,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            )
            return response.claims
        except Exception as e:
            print(f"Error extracting claims: {e}")
            return []



    async def _check_claim(self, claim: str) -> ClaimResult:
        """Search the web for a single claim and score it as supported/contradicted/unverifiable."""
        if not self._search_client.is_available():
            return ClaimResult(
                claim=claim,
                verdict=VerificationVerdict.UNVERIFIABLE,
                evidence="Search client not available.",
            )

        results = await self._search_client.search(claim)
        if not results:
            return ClaimResult(
                claim=claim,
                verdict=VerificationVerdict.UNVERIFIABLE,
                evidence="No search results found.",
            )

        # Concatenate snippets for evaluation
        context = "\n".join([f"- {r.snippet}" for r in results])

        try:
            prompt = (
                f"Evaluate the following claim based on the provided search results.\n"
                f"Claim: {claim}\n\n"
                f"Search Results:\n{context}\n\n"
                f"Provide a verdict and evidence."
            )
            response = await self._llm_client.chat.completions.create(
                model=self._config.llm.model_name,
                response_model=ClaimEvaluation,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            )
            return ClaimResult(
                claim=claim, verdict=response.verdict, evidence=response.evidence
            )
        except Exception as e:
            print(f"Error evaluating claim: {e}")
            return ClaimResult(
                claim=claim,
                verdict=VerificationVerdict.UNVERIFIABLE,
                evidence=f"Error during evaluation: {e}",
            )



    def _aggregate_confidence(self, results: list[ClaimResult]) -> float:
        """Compute an aggregate confidence score from individual claim results."""
        if not results:
            return self._config.verification.mock_confidence

        total_score = 0.0
        for r in results:
            if r.verdict == VerificationVerdict.SUPPORTED:
                total_score += 1.0
            elif r.verdict == VerificationVerdict.CONTRADICTED:
                total_score += 0.0
            elif r.verdict == VerificationVerdict.UNVERIFIABLE:
                total_score += 0.5

        return total_score / len(results)


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
