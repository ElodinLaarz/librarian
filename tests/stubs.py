"""Test stubs: deterministic, in-memory implementations of all service dependencies."""

from __future__ import annotations

from uuid import UUID

import numpy as np

from src.config import DatabaseSettings, EmbeddingSettings, LibrarianConfig
from src.models.enums import VerificationVerdict
from src.models.research_job import ResearchJob
from src.models.tome import Tome
from src.services.embedding import EmbeddingService
from src.services.ingestor import Ingestor
from src.services.verifier import ClaimResult, VerificationResult, Verifier
from src.services.web_search import WebSearchClient, WebSearchResult
from src.storage.research_job_repository import ResearchJobRepository
from src.storage.tome_repository import TomeRepository


class StubTomeRepository(TomeRepository):
    """In-memory repository.

    Call `seed_near_duplicates` before the test to pre-populate tomes that will be
    returned by `find_near_duplicates`.  When a seeded tome is deleted, it is
    removed from both the main store and the near-duplicate list so that subsequent
    calls to `find_near_duplicates` no longer return it (prevents infinite reshard).
    """

    def __init__(self, *, fail_deletes: bool = False) -> None:
        self._tomes: dict[UUID, Tome] = {}
        self._near_duplicates: list[Tome] = []
        self._fail_deletes = fail_deletes

    def seed_near_duplicates(self, tomes: list[Tome]) -> None:
        """Register tomes as near-duplicates AND add them to the main store."""
        for t in tomes:
            self._tomes[t.id] = t
        self._near_duplicates = list(tomes)

    def all_tomes(self) -> list[Tome]:
        return list(self._tomes.values())

    async def insert(self, tome: Tome) -> UUID:
        self._tomes[tome.id] = tome
        return tome.id

    async def delete(self, tome_id: UUID) -> bool:
        if self._fail_deletes:
            return False
        if tome_id not in self._tomes:
            return False
        del self._tomes[tome_id]
        self._near_duplicates = [t for t in self._near_duplicates if t.id != tome_id]
        return True

    async def get_by_id(self, tome_id: UUID) -> Tome | None:
        return self._tomes.get(tome_id)

    async def search(
        self,
        query: str,
        top_k: int = 5,
        min_confidence: float = 0.5,
        category: str | None = None,
    ) -> list[tuple[Tome, float]]:
        results = [
            (t, 1.0)
            for t in self._tomes.values()
            if t.confidence >= min_confidence and (category is None or t.category == category)
        ]
        return results[:top_k]

    async def find_near_duplicates(self, tome: Tome) -> list[Tome]:
        return list(self._near_duplicates)

    def close(self) -> None:
        self._tomes.clear()
        self._near_duplicates.clear()


class StubEmbeddingService(EmbeddingService):
    """Returns a zero vector of the configured dimensionality."""

    def __init__(self, dimensions: int = 768) -> None:
        super().__init__(EmbeddingSettings(dimensions=dimensions))

    async def initialize(self) -> None:
        pass

    async def embed(self, text: str) -> np.ndarray:
        return np.zeros(self._settings.dimensions, dtype=np.float32)


class StubVerifier(Verifier):
    """Returns a fixed confidence score without making any network calls."""

    def __init__(self, confidence: float = 0.8) -> None:
        # Bypass Verifier.__init__ — no settings or web_search needed.
        self._confidence = confidence

    async def verify(self, content: str) -> VerificationResult:
        return VerificationResult(
            confidence=self._confidence,
            claims=[
                ClaimResult(
                    claim="stub claim",
                    verdict=VerificationVerdict.SUPPORTED,
                    evidence="stub evidence",
                )
            ],
            skipped=False,
        )


class StubIngestor(Ingestor):
    """Ingestor with deterministic, LLM-free overrides for all abstract methods."""

    async def _reshard(self, blob: str) -> list[str]:
        """Split on double newlines; discard blank segments."""
        return [seg.strip() for seg in blob.split("\n\n") if seg.strip()]

    async def _classify_and_tag(
        self,
        chunk: str,
        category_hint: str | None = None,
        tags_hint: list[str] | None = None,
    ) -> tuple[str, list[str]]:
        if tags_hint:
            return category_hint or "general", list(tags_hint)
        return category_hint or "general", ["stub"]

    async def _generate_title_and_summary(self, chunk: str) -> tuple[str, str]:
        title = chunk[:50].strip()
        summary = f"Summary: {chunk[:80].strip()}"
        return (title, summary)


class StubWebSearchClient(WebSearchClient):
    """Returns a fixed list of results without making any network calls."""

    def __init__(self, results: list[WebSearchResult] | None = None) -> None:
        self._results = results or []
        self._available = bool(results is not None)

    async def search(self, query: str, max_results: int = 3) -> list[WebSearchResult]:
        return self._results[:max_results]

    def is_available(self) -> bool:
        return self._available


class StubResearchJobRepository(ResearchJobRepository):
    """In-memory research job store."""

    def __init__(self) -> None:
        self._jobs: dict[UUID, ResearchJob] = {}

    async def insert(self, job: ResearchJob) -> UUID:
        self._jobs[job.id] = job
        return job.id

    async def update(self, job: ResearchJob) -> None:
        self._jobs[job.id] = job

    async def get_by_id(self, job_id: UUID) -> ResearchJob | None:
        return self._jobs.get(job_id)

    def all_jobs(self) -> list[ResearchJob]:
        return list(self._jobs.values())

    def close(self) -> None:
        pass


def make_stub_ingestor(
    *,
    config: LibrarianConfig | None = None,
    confidence: float = 0.8,
    repo: StubTomeRepository | None = None,
    dimensions: int | None = None,
) -> tuple[StubIngestor, StubTomeRepository, StubVerifier]:
    """Convenience factory — returns (ingestor, repo, verifier) wired together."""
    config = config or LibrarianConfig(
        database=DatabaseSettings(uri="mongodb://localhost:27017"),
    )
    repo = repo or StubTomeRepository()
    verifier = StubVerifier(confidence=confidence)
    if dimensions is None:
        dimensions = config.embedding.dimensions
    embedding_service = StubEmbeddingService(dimensions=dimensions)
    ingestor = StubIngestor(config, embedding_service, verifier, repo)
    return ingestor, repo, verifier
