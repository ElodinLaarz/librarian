"""Edge case tests for the Ingestor service."""

from __future__ import annotations

import asyncio
import uuid

import numpy as np
import pytest

from src.models.enums import IngestStatus, SourceType
from src.models.tome import Tome
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
async def test_reshard_insert_failure_preserves_originals() -> None:
    ingestor, repo, _ = make_stub_ingestor()
    existing = _make_tome("Original")
    repo.seed_near_duplicates([existing])

    # Mock insert to fail
    async def fail_insert(tome):
        raise Exception("DB insert failed")

    repo.insert = fail_insert

    # Let's use a more robust way to check if delete was called.
    delete_called = False
    original_delete = repo.delete

    async def track_delete(tome_id):
        nonlocal delete_called
        delete_called = True
        return await original_delete(tome_id)

    repo.delete = track_delete

    output = await ingestor.ingest("New content")

    assert output.status == IngestStatus.REJECTED
    assert "DB insert failed" in output.reject_reason
    assert not delete_called
    # Check if original is still there
    assert existing.id in repo._tomes


@pytest.mark.asyncio
async def test_concurrent_ingest_duplicates() -> None:
    """Two concurrent ingests of similar content should both succeed, even if duplicates appear."""
    ingestor, repo, _ = make_stub_ingestor()

    content = "Atomic fact."

    # Run two ingests concurrently
    results = await asyncio.gather(ingestor.ingest(content), ingestor.ingest(content))

    assert results[0].status == IngestStatus.STORED
    assert results[1].status == IngestStatus.STORED

    # Depending on timing, we might have 1 or 2 tomes.
    # In this stub environment, they both likely see 0 duplicates initially.
    # Or one sees 0, one sees the first one.
    tomes = repo.all_tomes()
    assert len(tomes) >= 1


@pytest.mark.asyncio
async def test_reshard_error_contains_replacements() -> None:
    """When a ReshardError occurs, it should contain the replacements that WERE stored."""
    ingestor, repo, _ = make_stub_ingestor()
    existing = _make_tome("Existing")
    repo.seed_near_duplicates([existing])

    # Force delete to fail and raise ReshardError
    repo._fail_deletes = True

    # ingestor._dedup_and_store raises ReshardError if delete fails
    # ingestor.ingest catches ReshardError and adds replacements to 'stored'

    output = await ingestor.ingest("New content")

    assert output.status == IngestStatus.PARTIAL
    assert "Failed to delete tomes" in output.reject_reason
    assert len(output.tomes) > 0  # Replacements were still returned

    # Check that replacements ARE in the repo
    for t in output.tomes:
        assert await repo.get_by_id(t.id) is not None


@pytest.mark.asyncio
async def test_reshard_with_empty_blob() -> None:
    """Resharding an empty blob should be handled gracefully."""
    ingestor, _, _ = make_stub_ingestor()
    # ingest() already handles empty blob, but let's test _reshard directly if we can
    shards = await ingestor._reshard("")
    assert shards == []
