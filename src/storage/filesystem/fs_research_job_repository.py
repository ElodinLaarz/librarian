from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from uuid import UUID

from src.config import DatabaseSettings
from src.models.research_job import ResearchJob
from src.storage.errors import (
    BackendUnavailableError,
    NotFoundError,
    StorageError,
)
from src.storage.filesystem.utils import atomic_write_text, resolve_base_path
from src.storage.research_job_repository import ResearchJobRepository

logger = logging.getLogger(__name__)


class FsResearchJobRepository(ResearchJobRepository):
    """File-system implementation of ResearchJobRepository.

    Stores each job as a JSON file under ``<base_path>/research_jobs/``.
    """

    def __init__(self, settings: DatabaseSettings) -> None:
        self._jobs_dir = resolve_base_path(settings.uri) / settings.jobs_collection
        try:
            self._jobs_dir.mkdir(parents=True, exist_ok=True)
        except FileNotFoundError as exc:
            raise BackendUnavailableError(
                f"Filesystem storage root unreachable: {self._jobs_dir}"
            ) from exc
        except PermissionError as exc:
            raise BackendUnavailableError(
                f"Filesystem storage root not writable: {self._jobs_dir}"
            ) from exc
        except OSError as exc:
            raise StorageError(f"Filesystem storage root error: {self._jobs_dir}") from exc

    def _get_path(self, job_id: UUID) -> Path:
        return self._jobs_dir / f"{job_id}.json"

    async def insert(self, job: ResearchJob) -> UUID:
        """Create a new job."""
        path = self._get_path(job.id)
        try:
            await asyncio.to_thread(atomic_write_text, path, job.model_dump_json(indent=2))
        except (FileNotFoundError, PermissionError) as exc:
            raise BackendUnavailableError(
                f"Filesystem storage root unreachable or not writable: {self._jobs_dir}"
            ) from exc
        except OSError as exc:
            raise StorageError(f"Failed to write job {job.id}") from exc
        return job.id

    async def update(self, job: ResearchJob) -> None:
        """Replace the existing job with the full current state."""
        path = self._get_path(job.id)
        exists = await asyncio.to_thread(path.exists)
        if not exists:
            raise NotFoundError(f"Job {job.id} does not exist")
        try:
            await asyncio.to_thread(atomic_write_text, path, job.model_dump_json(indent=2))
        except (FileNotFoundError, PermissionError) as exc:
            raise BackendUnavailableError(
                f"Filesystem storage root unreachable or not writable: {self._jobs_dir}"
            ) from exc
        except OSError as exc:
            raise StorageError(f"Failed to update job {job.id}") from exc

    async def get_by_id(self, job_id: UUID) -> ResearchJob | None:
        """Load a job if it exists."""
        path = self._get_path(job_id)
        exists = await asyncio.to_thread(path.exists)
        if not exists:
            return None
        try:
            content = await asyncio.to_thread(path.read_text)
        except OSError as exc:
            # Existence check passed but read failed (race, permission).
            # Surface as a storage failure rather than silently masquerading
            # as "not found".
            raise StorageError(f"Failed to read job {job_id}") from exc
        try:
            return ResearchJob.model_validate_json(content)
        except Exception:
            # JSON parse / schema mismatch — preserve previous "treat as
            # absent" contract rather than raising, but leave a trace so
            # corruption is distinguishable from a genuinely missing job.
            logger.warning("Job file %s is corrupt; treating as absent", path, exc_info=True)
            return None

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
                logger.warning("Skipping unreadable job file %s", path, exc_info=True)
                continue
        return jobs

    def close(self) -> None:
        pass
