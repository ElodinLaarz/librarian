from __future__ import annotations

from abc import ABC, abstractmethod
from uuid import UUID

from src.models.research_job import ResearchJob


class ResearchJobRepository(ABC):
    """Persistence for asynchronous research job records."""

    @abstractmethod
    async def insert(self, job: ResearchJob) -> UUID: ...

    @abstractmethod
    async def update(self, job: ResearchJob) -> None: ...

    @abstractmethod
    async def get_by_id(self, job_id: UUID) -> ResearchJob | None: ...
