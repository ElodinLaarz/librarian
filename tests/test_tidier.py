"""Tests for the Tidier service."""

from __future__ import annotations

import uuid

import numpy as np
import pytest

from src.config import TidySettings
from src.models.enums import SourceType
from src.models.tome import Tome
from src.services.tidier import Tidier
from tests.stubs import make_stub_ingestor


def _make_tome(content: str) -> Tome:
    return Tome(
        id=uuid.uuid4(),
        title="Tome",
        content=content,
        summary="Summary",
        category="general",
        tags=["stub"],
        source_url=None,
        source_type=SourceType.AGENT_INPUT,
        confidence=0.8,
        embedding=np.zeros(768, dtype=np.float32),
    )


@pytest.mark.asyncio
async def test_run_cleanup_no_tomes() -> None:
    ingestor, repo, _ = make_stub_ingestor()
    tidier = Tidier(ingestor, repo, TidySettings())

    report = await tidier.run_cleanup()

    assert report["scanned"] == 0
    assert report["groups_found"] == 0
    assert report["groups_consolidated"] == 0
    assert report["tomes_removed"] == 0
    assert report["failed_groups"] == 0


@pytest.mark.asyncio
async def test_run_cleanup_no_duplicates() -> None:
    ingestor, repo, _ = make_stub_ingestor()
    tome = _make_tome("Unique content")
    await repo.insert(tome)

    tidier = Tidier(ingestor, repo, TidySettings())
    report = await tidier.run_cleanup()

    assert report["scanned"] == 1
    assert report["groups_found"] == 0
    assert report["groups_consolidated"] == 0
    assert report["tomes_removed"] == 0


@pytest.mark.asyncio
async def test_run_cleanup_consolidates_duplicates() -> None:
    ingestor, repo, _ = make_stub_ingestor()
    t1 = _make_tome("Identical fact")
    t2 = _make_tome("Identical fact")
    await repo.insert(t1)
    await repo.insert(t2)

    repo.seed_duplicate_groups([[t1, t2]])

    tidier = Tidier(ingestor, repo, TidySettings())
    report = await tidier.run_cleanup()

    assert report["groups_found"] == 1
    assert report["groups_consolidated"] == 1
    assert report["tomes_removed"] == 1

    assert await repo.get_by_id(t1.id) is None
    assert await repo.get_by_id(t2.id) is None
    assert len(repo.all_tomes()) == 1
    assert repo.all_tomes()[0].content == "Identical fact"


@pytest.mark.asyncio
async def test_run_cleanup_does_not_report_negative_tomes_removed_when_group_expands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ingestor, repo, _ = make_stub_ingestor()
    original_a = _make_tome("Fact A")
    original_b = _make_tome("Fact A")
    repo.seed_duplicate_groups([[original_a, original_b]])

    tidier = Tidier(ingestor, repo, TidySettings())

    async def expand_consolidate(
        tomes: list[Tome],
        skip_verify: bool = False,
    ) -> list[Tome]:
        del tomes
        del skip_verify
        return [
            _make_tome("Reshard 1"),
            _make_tome("Reshard 2"),
            _make_tome("Reshard 3"),
        ]

    monkeypatch.setattr(ingestor, "consolidate", expand_consolidate)

    report = await tidier.run_cleanup()

    assert report["groups_found"] == 1
    assert report["groups_consolidated"] == 1
    assert report["tomes_removed"] == 0


@pytest.mark.asyncio
async def test_run_cleanup_respects_limit() -> None:
    ingestor, repo, _ = make_stub_ingestor()
    group_a = [_make_tome("Fact A"), _make_tome("Fact A")]
    group_b = [_make_tome("Fact B"), _make_tome("Fact B")]
    repo.seed_duplicate_groups([group_a, group_b])

    tidier = Tidier(ingestor, repo, TidySettings(limit_per_run=1))
    report = await tidier.run_cleanup(limit=1)

    assert report["scanned"] == 4
    assert report["groups_found"] == 2
    assert report["groups_consolidated"] == 1
    assert report["skipped_groups"] == 1


@pytest.mark.asyncio
async def test_run_cleanup_handles_ingestor_error(monkeypatch: pytest.MonkeyPatch) -> None:
    ingestor, repo, _ = make_stub_ingestor()
    t1 = _make_tome("Fact A")
    t2 = _make_tome("Fact A duplicate")
    repo.seed_duplicate_groups([[t1, t2]])

    tidier = Tidier(ingestor, repo, TidySettings())

    async def fail_consolidate(tomes, skip_verify=False):  # type: ignore[no-untyped-def]
        raise Exception("Consolidation failed")

    monkeypatch.setattr(ingestor, "consolidate", fail_consolidate)

    report = await tidier.run_cleanup()

    assert report["scanned"] >= 1
    assert report["groups_consolidated"] == 0
    assert report["failed_groups"] == 1


@pytest.mark.asyncio
async def test_run_cleanup_skips_overlapping_groups() -> None:
    ingestor, repo, _ = make_stub_ingestor()
    shared = _make_tome("Shared fact")
    other_a = _make_tome("Fact A")
    other_b = _make_tome("Fact B")
    repo.seed_duplicate_groups([[shared, other_a], [shared, other_b]])

    tidier = Tidier(ingestor, repo, TidySettings())
    report = await tidier.run_cleanup()

    assert report["groups_found"] == 2
    assert report["groups_consolidated"] == 1
    assert report["skipped_groups"] == 1
