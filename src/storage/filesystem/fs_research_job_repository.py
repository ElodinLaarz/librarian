from __future__ import annotations

from pathlib import Path
from uuid import UUID

from src.config import DatabaseSettings
from src.models.research_job import ResearchJob
from src.storage.research_job_repository import ResearchJobRepository


class FsResearchJobRepository(ResearchJobRepository):
    """File-system implementation of the ResearchJobRepository."""

    def __init__(self, settings: DatabaseSettings) -> None:
        if settings.uri in ("localhost", ""):
            self._base_path = Path.home() / ".librarian_mcp"
        elif settings.uri.startswith("file://"):
            self._base_path = Path(settings.uri[7:]).expanduser()
        else:
            self._base_path = Path(settings.uri).expanduser()

        self._jobs_dir = self._base_path / settings.jobs_collection
        self._jobs_dir.mkdir(parents=True, exist_ok=True)

    def _get_path(self, job_id: UUID) -> Path:
        return self._jobs_dir / f"{job_id}.json"

    async def insert(self, job: ResearchJob) -> UUID:
        path = self._get_path(job.id)
        path.write_text(job.model_dump_json(indent=2))
        return job.id

    async def update(self, job: ResearchJob) -> None:
        path = self._get_path(job.id)
        path.write_text(job.model_dump_json(indent=2))

    async def get_by_id(self, job_id: UUID) -> ResearchJob | None:
        path = self._get_path(job_id)
        if not path.exists():
            return None
        return ResearchJob.model_validate_json(path.read_text())

    def close(self) -> None:
        pass
