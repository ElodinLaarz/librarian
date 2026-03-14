from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src.models.tome import Tome


class TomeRepository(ABC):
    """Abstract CRUD and vector search operations for Tomes.

    Concrete implementations bind this to a specific database backend.
    """

    @abstractmethod
    async def insert(self, tome: Tome) -> str:
        """Insert a new Tome and return its ID."""
        ...

    @abstractmethod
    async def update(self, tome_id: str, updates: dict[str, Any]) -> bool:
        """Partially update a Tome by ID. Returns True if a document was modified."""
        ...

    @abstractmethod
    async def get_by_id(self, tome_id: str) -> Tome | None:
        """Retrieve a single Tome by its ID."""
        ...

    @abstractmethod
    async def vector_search(
        self,
        embedding: list[float],
        top_k: int = 5,
        category: str | None = None,
        min_confidence: float = 0.5,
    ) -> list[tuple[Tome, float]]:
        """Perform ANN vector search. Returns (Tome, score) pairs sorted by similarity."""
        ...

    @abstractmethod
    async def find_near_duplicates(
        self, embedding: list[float], threshold: float = 0.95
    ) -> list[Tome]:
        """Find existing Tomes with cosine similarity above the threshold."""
        ...

    @abstractmethod
    async def find_by_research_job(self, job_id: str) -> list[Tome]:
        """Retrieve all Tomes produced by a specific research job."""
        ...

    @abstractmethod
    async def supersede(self, old_id: str, new_id: str) -> None:
        """Mark an old Tome as superseded by a new one."""
        ...
