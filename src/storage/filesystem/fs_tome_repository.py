from __future__ import annotations

from pathlib import Path
from uuid import UUID

from src.config import DatabaseSettings
from src.models.tome import Tome
from src.storage.tome_repository import TomeRepository


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
    ) -> list[tuple[Tome, float]]:
        """Perform a brute-force search across all JSON files.

        Note: In a real implementation, this would use the query embedding
        to perform vector similarity search.
        """
        results: list[tuple[Tome, float]] = []
        for path in self._tomes_dir.glob("*.json"):
            try:
                tome = Tome.model_validate_json(path.read_text())
                if tome.confidence >= min_confidence:
                    # Mock similarity score for skeleton
                    results.append((tome, 1.0))
            except Exception:
                continue

        # Sort by score descending and limit to top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    async def find_near_duplicates(self, tome: Tome) -> list[Tome]:
        """Find existing Tomes with high similarity.

        Scans all files to find potential duplicates.
        """
        duplicates: list[Tome] = []
        for path in self._tomes_dir.glob("*.json"):
            try:
                existing = Tome.model_validate_json(path.read_text())
                # In a real implementation, compare embeddings here
                if existing.id != tome.id and existing.title == tome.title:
                    duplicates.append(existing)
            except Exception:
                continue
        return duplicates
