"""FsResearchJobRepository — resilient listing of job JSON files."""

from __future__ import annotations

import uuid
from pathlib import Path
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

    def selective_read_text(self: Path, *args: object, **kwargs: object) -> str:
        if self.resolve() == bad.resolve():
            raise PermissionError("mock unreadable")
        return real_read_text(self, *args, **kwargs)

    with mock.patch.object(Path, "read_text", selective_read_text):
        out = repo.all_jobs()

    assert len(out) == 1
    assert out[0].id == good_id
