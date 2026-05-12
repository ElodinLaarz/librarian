from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from uuid import UUID

from src.models.tome import Tome


@dataclass(slots=True)
class DuplicateScanResult:
    groups: list[list[Tome]]
    scanned: int
    exact_content_groups: int = 0
    fact_overlap_groups: int = 0
    semantic_groups: int = 0
    ignored_high_frequency_facts: int = 0


class TomeRepository(ABC):
    """Abstract CRUD and vector search operations for Tomes.

    Concrete implementations bind this to a specific database backend.
    """

    @abstractmethod
    async def insert(self, tome: Tome) -> UUID:
        """Insert a new Tome and return its ID."""
        ...

    @abstractmethod
    async def delete(self, tome_id: UUID) -> bool:
        """Permanently remove a Tome by ID. Returns True if a document was deleted."""
        ...

    @abstractmethod
    async def get_by_id(self, tome_id: UUID) -> Tome | None:
        """Retrieve a single Tome by its ID."""
        ...

    @abstractmethod
    async def search(
        self,
        query: str,
        top_k: int = 5,
        min_confidence: float = 0.5,
        category: str | None = None,
    ) -> list[tuple[Tome, float]]:
        """Perform search. Returns (Tome, score) pairs sorted by relevance."""
        ...

    @abstractmethod
    async def find_near_duplicates(self, tome: Tome, threshold: float | None = None) -> list[Tome]:
        """Find existing Tomes with cosine similarity above the threshold."""
        ...

    @abstractmethod
    async def find_all_near_duplicates(self, threshold: float = 0.95) -> DuplicateScanResult:
        """Find all duplicate groups in the library for tidy-time consolidation.

        Implementations should return groups with consistent semantics across
        exact content, fact overlap, and semantic similarity detection.
        """
        ...

    @abstractmethod
    async def list_all(
        self,
        limit: int = 100,
        offset: int = 0,
        *,
        category: str | None = None,
        min_confidence: float = 0.0,
        research_job_id: UUID | None = None,
        thread_id: UUID | None = None,
    ) -> list[Tome]:
        """Return a page of Tomes matching the given filters.

        Results are sorted by ``created_at`` descending (newest first) for
        consistent pagination across calls, unless ``thread_id`` is set in which
        case results are sorted by ``thread_position`` ascending (conversational order).
        Filters compose with AND. Embeddings should be included so the caller can
        decide whether to strip them — keeps this method usable from non-MCP code paths.
        """
        ...

    @abstractmethod
    async def count(
        self,
        *,
        category: str | None = None,
        min_confidence: float = 0.0,
        research_job_id: UUID | None = None,
        thread_id: UUID | None = None,
    ) -> int:
        """Return the total number of Tomes matching the given filters.

        Companion to :meth:`list_all` for pagination — implementations must
        apply the same filter predicates so callers can compute ``has_more``
        from ``offset + len(page) < total``.
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Frees any resources held by the repository."""
        pass
