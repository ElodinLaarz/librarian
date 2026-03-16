from __future__ import annotations

from abc import ABC, abstractmethod
from uuid import UUID

import numpy as np

from src.models.tome import Tome


class TomeRepository(ABC):
    """Abstract CRUD and vector search operations for Tomes.

    Concrete implementations bind this to a specific database backend.
    """

    @abstractmethod
    async def get_embedding(self, text: str) -> np.ndarray:
        """Generate an embedding for the given text."""
        ...

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
    ) -> list[tuple[Tome, float]]:
        """Perform ANN vector search. Returns (Tome, score) pairs sorted by similarity."""
        ...

    @abstractmethod
    async def find_near_duplicates(self, tome: Tome) -> list[Tome]:
        """Find existing Tomes with cosine similarity above the threshold."""
        ...
