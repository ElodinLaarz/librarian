from __future__ import annotations

from pathlib import Path
from uuid import UUID

from src.config import DatabaseSettings
from src.models.research_job import ResearchJob
from src.storage.filesystem.utils import resolve_base_path
from src.storage.research_job_repository import ResearchJobRepository


class FsResearchJobRepository(ResearchJobRepository):
    """File-system implementation of ResearchJobRepository.

    Stores each job as a JSON file under ``<base_path>/research_jobs/``.
    """

    def __init__(self, settings: DatabaseSettings) -> None:
        self._jobs_dir = resolve_base_path(settings.uri) / "research_jobs"
        self._jobs_dir.mkdir(parents=True, exist_ok=True)

    def _get_path(self, job_id: UUID) -> Path:
        return self._jobs_dir / f"{job_id}.json"

    async def insert(self, job: ResearchJob) -> UUID:
        """Create a new job."""
        self._get_path(job.id).write_text(job.model_dump_json(indent=2))
        return job.id

    async def update(self, job: ResearchJob) -> None:
        """Replace the existing job with the full current state."""
        path = self._get_path(job.id)
        if not path.exists():
            raise ValueError(f"Job {job.id} does not exist")
        path.write_text(job.model_dump_json(indent=2))

    async def get_by_id(self, job_id: UUID) -> ResearchJob | None:
        """Load a job if it exists."""
        path = self._get_path(job_id)
        if not path.exists():
            return None
        return ResearchJob.model_validate_json(path.read_text())

    def all_jobs(self) -> list[ResearchJob]:
        """Load all jobs.

        Skips a file on any per-file failure: I/O errors (e.g. permission, missing after
        glob), JSON parse errors, or schema validation errors.
        """
        jobs: list[ResearchJob] = []
        for path in self._jobs_dir.glob("*.json"):
            try:
                jobs.append(ResearchJob.model_validate_json(path.read_text()))
            except Exception:
                continue
        return jobs

    def close(self) -> None:
        pass
