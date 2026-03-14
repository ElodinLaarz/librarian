from __future__ import annotations

from pathlib import Path

from src.models.tome import Tome
from src.storage.filesystem.fs_database_client import FsDatabaseClient
from src.storage.tome_repository import TomeRepository


class FsTomeRepository(TomeRepository):
    """File-system implementation of the TomeRepository.

    Stores each Tome as a JSON file in ~/.librarian_mcp/tomes/
    """

    def __init__(self, client: FsDatabaseClient) -> None:
        self._client = client
        self._tomes_dir = client.base_path / client._settings.tomes_collection

    def _get_path(self, tome_id: str) -> Path:
        return self._tomes_dir / f"{tome_id}.json"

    async def insert(self, tome: Tome) -> str:
        """Save a Tome as a JSON file."""
        path = self._get_path(tome.id)
        path.write_text(tome.model_dump_json(indent=2))
        return tome.id

    async def update(self, tome_id: str, updates: dict) -> bool:
        """Update a Tome JSON file."""
        tome = await self.get_by_id(tome_id)
        if not tome:
            return False

        updated_data = tome.model_dump()
        updated_data.update(updates)
        updated_tome = Tome.model_validate(updated_data)
        await self.insert(updated_tome)
        return True

    async def get_by_id(self, tome_id: str) -> Tome | None:
        """Read a Tome from its JSON file."""
        path = self._get_path(tome_id)
        if not path.exists():
            return None
        return Tome.model_validate_json(path.read_text())

    async def search(
        self,
        embedding: str,
        top_k: int = 5,
        category: str | None = None,
        min_confidence: float = 0.5,
    ) -> list[tuple[Tome, float]]:
        """Perform a brute-force vector search across all JSON files."""

    async def find_near_duplicates(
        self, embedding: list[float], threshold: float = 0.95
    ) -> list[Tome]:
        """Find near-duplicates by scanning all files."""

    async def supersede(self, old_id: str, new_id: str) -> None:
        """Mark an old Tome as superseded by a new one."""
        await self.update(old_id, {"superseded_by": new_id})
