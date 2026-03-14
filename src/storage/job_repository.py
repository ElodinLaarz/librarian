from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from src.models.research_job import ResearchJob


class JobRepository(ABC):
    """Abstract CRUD operations for ResearchJobs.

    Concrete implementations bind this to a specific database backend.
    """

    @abstractmethod
    async def create(self, job: ResearchJob) -> str:
        """Insert a new ResearchJob and return its ID."""
        ...

    @abstractmethod
    async def get_by_id(self, job_id: str) -> ResearchJob | None:
        """Retrieve a ResearchJob by its ID."""
        ...

    @abstractmethod
    async def set_running(self, job_id: str) -> None:
        """Transition a job to 'running' status."""
        ...

    @abstractmethod
    async def set_completed(self, job_id: str, tome_ids: list[str], finished_at: datetime) -> None:
        """Mark a job as completed with its produced Tome IDs."""
        ...

    @abstractmethod
    async def set_failed(self, job_id: str, error: str, finished_at: datetime) -> None:
        """Mark a job as failed with an error message."""
        ...

    @abstractmethod
    async def add_queries(self, job_id: str, queries: list[str]) -> None:
        """Append search queries to a job's record."""
        ...
