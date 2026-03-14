from __future__ import annotations

import os
from pathlib import Path

from src.config import DatabaseSettings


class FsDatabaseClient:
    """File-system implementation of the DatabaseClient.

    Ensures the storage directory exists.
    """

    def __init__(self, settings: DatabaseSettings) -> None:
        super().__init__(settings)
        # Default to ~/.librarian_mcp if uri is "localhost" or empty
        if self._settings.uri in ("localhost", ""):
            self._base_path = Path.home() / ".librarian_mcp"
        else:
            self._base_path = Path(self._settings.uri).expanduser()

    async def connect(self) -> None:
        """Ensure the storage directories exist."""
        self._base_path.mkdir(parents=True, exist_ok=True)
        (self._base_path / self._settings.tomes_collection).mkdir(exist_ok=True)

    async def disconnect(self) -> None:
        """No-op for file system."""

    async def ensure_indexes(self) -> None:
        """No-op for file system (no native indexing)."""

    async def ping(self) -> bool:
        """Check if the storage directory is writable."""
        return os.access(self._base_path, os.W_OK)

    @property
    def base_path(self) -> Path:
        return self._base_path
