from __future__ import annotations

from pathlib import Path
from uuid import UUID

import numpy as np

from src.config import DatabaseSettings
from src.models.tome import Tome
from src.storage.tome_repository import TomeRepository

_DEDUP_SIMILARITY_THRESHOLD = 0.85


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity in [-1, 1]. Returns 0.0 if either vector is zero or shapes differ."""
    if a.shape != b.shape:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


class FsTomeRepository(TomeRepository):
    """File-system implementation of the TomeRepository.

    Stores each Tome as a JSON file in ~/.librarian_mcp/tomes/
    """

    def __init__(self, settings: DatabaseSettings) -> None:
        # Default to ~/.librarian_mcp if uri is "localhost" or empty
        if settings.uri in ("localhost", ""):
            self._base_path = Path.home() / ".librarian_mcp"
        else:
            self._base_path = Path(settings.uri).expanduser()
        self._tomes_dir = self._base_path / settings.tomes_collection
        self._tomes_dir.mkdir(parents=True, exist_ok=True)

    def _get_path(self, tome_id: UUID) -> Path:
        return self._tomes_dir / f"{tome_id}.json"

    async def insert(self, tome: Tome) -> UUID:
        """Save a Tome as a JSON file."""
        path = self._get_path(tome.id)
        path.write_text(tome.model_dump_json(indent=2))
        return tome.id

    async def delete(self, tome_id: UUID) -> bool:
        """Deletes a Tome by ID. Returns True if a file was removed."""
        path = self._get_path(tome_id)
        if path.exists():
            path.unlink()
            return True
        return False

    async def get_by_id(self, tome_id: UUID) -> Tome | None:
        """Read a Tome from its JSON file."""
        path = self._get_path(tome_id)
        if not path.exists():
            return None
        return Tome.model_validate_json(path.read_text())

    async def search(
        self,
        query: str,
        top_k: int = 5,
        min_confidence: float = 0.5,
        category: str | None = None,
    ) -> list[tuple[Tome, float]]:
        """Brute-force scan with confidence and category filters.

        Scores are placeholder until a query embedding is available
        via the repository interface.
        """
        results: list[tuple[Tome, float]] = []
        for path in self._tomes_dir.glob("*.json"):
            try:
                tome = Tome.model_validate_json(path.read_text())
                if tome.confidence < min_confidence:
                    continue
                if category is not None and tome.category != category:
                    continue
                # Mock similarity score for skeleton
                results.append((tome, 1.0))
            except Exception:
                continue
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    async def find_near_duplicates(self, tome: Tome) -> list[Tome]:
        """Find existing Tomes with cosine similarity above _DEDUP_SIMILARITY_THRESHOLD."""
        duplicates: list[Tome] = []
        tome_embedding = tome.embedding
        tome_norm = float(np.linalg.norm(tome_embedding))
        for path in self._tomes_dir.glob("*.json"):
            try:
                existing = Tome.model_validate_json(path.read_text())
                if existing.id == tome.id:
                    continue
                existing_embedding = existing.embedding
                denom = float(np.linalg.norm(existing_embedding) * tome_norm)
                if denom == 0.0:
                    sim = 0.0
                else:
                    sim = float(np.dot(existing_embedding, tome_embedding) / denom)
                if sim >= _DEDUP_SIMILARITY_THRESHOLD:
                    duplicates.append(existing)
            except Exception:
                continue
        return duplicates

    def close(self) -> None:
        pass
