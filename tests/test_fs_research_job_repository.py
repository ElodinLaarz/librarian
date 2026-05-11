"""FsResearchJobRepository — resilient listing of job JSON files."""

from __future__ import annotations

import threading
import uuid
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from src.config import DatabaseSettings
from src.models.research_job import ResearchJob
from src.storage.filesystem.fs_research_job_repository import FsResearchJobRepository


@pytest.fixture
def repo(tmp_path: Path) -> FsResearchJobRepository:
    return FsResearchJobRepository(DatabaseSettings(uri=str(tmp_path)))


def test_all_jobs_skips_invalid_json(repo: FsResearchJobRepository, tmp_path: Path) -> None:
    jobs_dir = repo._jobs_dir
    (jobs_dir / "bad.json").write_text("not json {")
    good_id = uuid.uuid4()
    good = ResearchJob(id=good_id, topic="ok")
    (jobs_dir / f"{good_id}.json").write_text(good.model_dump_json())
    out = repo.all_jobs()
    assert len(out) == 1
    assert out[0].id == good_id


def test_all_jobs_skips_valid_json_schema_mismatch(
    repo: FsResearchJobRepository, tmp_path: Path
) -> None:
    jobs_dir = repo._jobs_dir
    (jobs_dir / "wrong-shape.json").write_text('{"foo": true}')
    good_id = uuid.uuid4()
    good = ResearchJob(id=good_id, topic="ok")
    (jobs_dir / f"{good_id}.json").write_text(good.model_dump_json())
    out = repo.all_jobs()
    assert len(out) == 1
    assert out[0].topic == "ok"


def test_all_jobs_skips_when_read_raises_os_error(
    repo: FsResearchJobRepository, tmp_path: Path
) -> None:
    """Simulates unreadable file or delete-between-glob-and-read without crashing."""
    jobs_dir = repo._jobs_dir
    bad = jobs_dir / "unreadable.json"
    bad.write_text("{}")
    good_id = uuid.uuid4()
    good = ResearchJob(id=good_id, topic="ok")
    good_path = jobs_dir / f"{good_id}.json"
    good_path.write_text(good.model_dump_json())

    real_read_text = Path.read_text

    def selective_read_text(self: Path, *args: Any, **kwargs: Any) -> str:
        if self.resolve() == bad.resolve():
            raise PermissionError("mock unreadable")
        return real_read_text(self, *args, **kwargs)

    with mock.patch.object(Path, "read_text", selective_read_text):
        out = repo.all_jobs()

    assert len(out) == 1
    assert out[0].id == good_id


@pytest.mark.asyncio
async def test_insert_does_not_block_event_loop(repo: FsResearchJobRepository) -> None:
    """insert must run blocking write_text in a worker thread, not on the loop."""
    main_thread = threading.main_thread()
    observed: dict[str, threading.Thread] = {}

    real_write_text = Path.write_text

    def recording_write_text(self: Path, *args: Any, **kwargs: Any) -> int:
        observed["thread"] = threading.current_thread()
        return real_write_text(self, *args, **kwargs)

    job = ResearchJob(id=uuid.uuid4(), topic="async-insert")
    with mock.patch.object(Path, "write_text", recording_write_text):
        await repo.insert(job)

    assert observed["thread"] is not main_thread, (
        "Path.write_text ran on the event loop thread — insert is not actually async"
    )


@pytest.mark.asyncio
async def test_update_does_not_block_event_loop(repo: FsResearchJobRepository) -> None:
    """update must run blocking write_text and exists in a worker thread."""
    main_thread = threading.main_thread()
    job = ResearchJob(id=uuid.uuid4(), topic="async-update")
    # Seed the file synchronously so update() finds it.
    repo._get_path(job.id).write_text(job.model_dump_json())

    observed: dict[str, threading.Thread] = {}
    real_write_text = Path.write_text

    def recording_write_text(self: Path, *args: Any, **kwargs: Any) -> int:
        observed["thread"] = threading.current_thread()
        return real_write_text(self, *args, **kwargs)

    with mock.patch.object(Path, "write_text", recording_write_text):
        await repo.update(job)

    assert observed["thread"] is not main_thread, (
        "Path.write_text ran on the event loop thread — update is not actually async"
    )


@pytest.mark.asyncio
async def test_get_by_id_does_not_block_event_loop(repo: FsResearchJobRepository) -> None:
    """get_by_id must run blocking read_text in a worker thread."""
    main_thread = threading.main_thread()
    job = ResearchJob(id=uuid.uuid4(), topic="async-get")
    repo._get_path(job.id).write_text(job.model_dump_json())

    observed: dict[str, threading.Thread] = {}
    real_read_text = Path.read_text

    def recording_read_text(self: Path, *args: Any, **kwargs: Any) -> str:
        observed["thread"] = threading.current_thread()
        return real_read_text(self, *args, **kwargs)

    with mock.patch.object(Path, "read_text", recording_read_text):
        result = await repo.get_by_id(job.id)

    assert result is not None
    assert observed["thread"] is not main_thread, (
        "Path.read_text ran on the event loop thread — get_by_id is not actually async"
    )
