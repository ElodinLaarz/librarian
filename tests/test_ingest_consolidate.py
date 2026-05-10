"""Tests for Ingestor.consolidate."""

from __future__ import annotations

import asyncio

import pytest

from src.models.tome import Tome
from src.services.ingestor import ReshardError
from tests.stubs import make_stub_ingestor
from tests.test_utils import make_tome as _make_tome


@pytest.mark.asyncio
async def test_consolidate_empty() -> None:
    ingestor, _, _ = make_stub_ingestor()
    result = await ingestor.consolidate([])
    assert result == []


@pytest.mark.asyncio
async def test_consolidate_single_tome() -> None:
    ingestor, _, _ = make_stub_ingestor()
    t = _make_tome("Only one")
    result = await ingestor.consolidate([t])
    assert result == [t]


@pytest.mark.asyncio
async def test_consolidate_multiple_tomes_happy_path() -> None:
    ingestor, repo, _ = make_stub_ingestor()
    t1 = _make_tome("Part 1")
    t2 = _make_tome("Part 2")
    await repo.insert(t1)
    await repo.insert(t2)

    # StubIngestor splits on \n\n. Combined will be "Part 1\n\nPart 2"
    result = await ingestor.consolidate([t1, t2])

    assert len(result) == 2
    assert result[0].content == "Part 1"
    assert result[1].content == "Part 2"

    # Originals should be deleted
    assert await repo.get_by_id(t1.id) is None
    assert await repo.get_by_id(t2.id) is None
    # New ones should be in repo
    assert len(repo.all_tomes()) == 2


@pytest.mark.asyncio
async def test_consolidate_reshard_empty_returns_original(monkeypatch: pytest.MonkeyPatch) -> None:
    ingestor, repo, _ = make_stub_ingestor()
    t1 = _make_tome("Fact A")
    t2 = _make_tome("Fact B")
    await repo.insert(t1)
    await repo.insert(t2)

    # Mock _reshard to return empty list
    async def mock_reshard(blob: str) -> list[str]:
        return []

    monkeypatch.setattr(ingestor, "_reshard", mock_reshard)

    result = await ingestor.consolidate([t1, t2])

    assert result == [t1, t2]
    # Originals should NOT be deleted
    assert await repo.get_by_id(t1.id) is not None
    assert await repo.get_by_id(t2.id) is not None


@pytest.mark.asyncio
async def test_consolidate_delete_failure() -> None:
    ingestor, repo, _ = make_stub_ingestor()
    t1 = _make_tome("Fact A")
    t2 = _make_tome("Fact B")
    await repo.insert(t1)
    await repo.insert(t2)

    # Mock delete to fail
    repo._fail_deletes = True

    with pytest.raises(ReshardError, match="Failed to delete tomes"):
        await ingestor.consolidate([t1, t2])


@pytest.mark.asyncio
async def test_consolidate_limits_build_concurrency(monkeypatch: pytest.MonkeyPatch) -> None:
    ingestor, repo, _ = make_stub_ingestor()
    ingestor._config.ingest.build_concurrency = 2
    tomes = [_make_tome(f"Fact {i}") for i in range(4)]
    for tome in tomes:
        await repo.insert(tome)

    current = 0
    peak = 0

    async def tracked_build(text: str, opts: object) -> Tome | None:
        nonlocal current, peak
        del opts
        current += 1
        peak = max(peak, current)
        await asyncio.sleep(0.01)
        current -= 1
        return _make_tome(text)

    monkeypatch.setattr(ingestor, "_build_tome", tracked_build)

    result = await ingestor.consolidate(tomes)

    assert len(result) == 4
    assert peak <= 2
