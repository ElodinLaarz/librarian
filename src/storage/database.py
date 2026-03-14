from __future__ import annotations

from abc import ABC, abstractmethod

from src.config import DatabaseSettings


class DatabaseClient(ABC):
    """Abstract interface for the backing datastore.

    Concrete implementations (MongoDB, PostgreSQL, SQLite, etc.) must
    provide all methods below.
    """

    def __init__(self, settings: DatabaseSettings) -> None:
        self._settings = settings

    @abstractmethod
    async def connect(self) -> None:
        """Establish a connection to the database."""
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Close the database connection."""
        ...

    @abstractmethod
    async def ensure_indexes(self) -> None:
        """Create required indexes (vector, text, compound) if they don't exist."""
        ...

    @abstractmethod
    async def ping(self) -> bool:
        """Health-check the database connection."""
        ...
