from __future__ import annotations

from pathlib import Path
from uuid import UUID

import numpy as np

from src.config import DatabaseSettings
from src.models.tome import Tome
from src.services.embedding import EmbeddingService
from src.storage.filesystem.utils import resolve_base_path
from src.storage.tome_repository import TomeRepository

_DEDUP_SIMILARITY_THRESHOLD = 0.85


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity in [-1, 1]. Returns 0.0 if shapes differ or either vector is zero."""
    if a.shape != b.shape:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


class FsTomeRepository(TomeRepository):
    """File-system implementation of the TomeRepository.

    Stores each Tome as a JSON file under ``<base_path>/tomes/``.
    Uses the injected :class:`~src.services.embedding.EmbeddingService` to
    embed search queries so cosine similarity scores are real rather than
    placeholder 1.0 values.
    """

    def __init__(self, settings: DatabaseSettings, embedding_service: EmbeddingService) -> None:
        self._embedding_service = embedding_service
        self._tomes_dir = resolve_base_path(settings.uri) / settings.tomes_collection
        self._tomes_dir.mkdir(parents=True, exist_ok=True)

    def _get_path(self, tome_id: UUID) -> Path:
        return self._tomes_dir / f"{tome_id}.json"

    async def insert(self, tome: Tome) -> UUID:
        """Save a Tome as a JSON file."""
        self._get_path(tome.id).write_text(tome.model_dump_json(indent=2))
        return tome.id

    async def delete(self, tome_id: UUID) -> bool:
        """Deletes a Tome by ID. Returns True if a file was removed."""
        path = self._get_path(tome_id)
        if not path.exists():
            return False
        path.unlink()
        return True

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
        """Brute-force cosine-similarity scan over all stored Tomes.

        Embeds *query* with the injected :class:`EmbeddingService` and ranks
        results by cosine similarity to each Tome's stored embedding.
        Tomes without an embedding are still included but scored 0.0.
        """
        query_vec: np.ndarray | None = None
        try:
            raw = await self._embedding_service.embed(query)
            query_vec = np.array(raw, dtype=np.float64)
        except Exception:
            pass

        results: list[tuple[Tome, float]] = []
        for path in self._tomes_dir.glob("*.json"):
            try:
                tome = Tome.model_validate_json(path.read_text())
            except Exception:
                continue

            if tome.confidence < min_confidence:
                continue
            if category is not None and tome.category != category:
                continue

            score = 0.0
            if query_vec is not None and tome.embedding is not None:
                score = _cosine_similarity(query_vec, np.array(tome.embedding, dtype=np.float64))

            results.append((tome, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    async def find_near_duplicates(self, tome: Tome) -> list[Tome]:
        """Find existing Tomes with cosine similarity above _DEDUP_SIMILARITY_THRESHOLD."""
        if tome.embedding is None:
            return []
        tome_vec = np.array(tome.embedding, dtype=np.float64)
        duplicates: list[Tome] = []
        for path in self._tomes_dir.glob("*.json"):
            try:
                existing = Tome.model_validate_json(path.read_text())
            except Exception:
                continue
            if existing.id == tome.id or existing.embedding is None:
                continue
            sim = _cosine_similarity(tome_vec, np.array(existing.embedding, dtype=np.float64))
            if sim >= _DEDUP_SIMILARITY_THRESHOLD:
                duplicates.append(existing)
        return duplicates

    def close(self) -> None:
        pass
