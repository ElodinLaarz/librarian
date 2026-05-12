"""Edge case tests for the Ingestor service."""

from __future__ import annotations

import asyncio

import pytest

from src.models.enums import IngestStatus
from tests.stubs import make_stub_ingestor
from tests.test_utils import make_tome as _make_tome


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
    """Two concurrent ingests of similar content should both succeed.

    They might create duplicates depending on timing.
    """
    ingestor, repo, _ = make_stub_ingestor()

    content = "Atomic fact."

    # Run two ingests concurrently
    results = await asyncio.gather(ingestor.ingest(content), ingestor.ingest(content))

    assert results[0].status == IngestStatus.STORED
    assert results[1].status == IngestStatus.STORED

    # Depending on timing, we might have 1 or 2 tomes.
    # In this stub environment, they both likely see 0 duplicates initially.
    assert len(repo._tomes) >= 1


@pytest.mark.asyncio
async def test_reshard_delete_failure_reports_partial() -> None:
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
        assert t.id in repo._tomes


@pytest.mark.asyncio
async def test_minimum_content_size() -> None:
    """Enforce minimum 100-character tome size to prevent fragments.

    Validates issue #35: minimum tome content length prevents short pollution.
    Aligns with README spec of 100–400 words per tome.
    """
    ingestor, repo, _ = make_stub_ingestor()

    # Test 1: 2-char content should be REJECTED
    output_short = await ingestor.ingest("hi")
    assert output_short.status == IngestStatus.REJECTED
    assert "minimum" in output_short.reject_reason.lower()

    # Test 2: 100-char content (at floor) should be STORED
    content_valid = "a" * 100
    output_valid = await ingestor.ingest(content_valid)
    assert output_valid.status == IngestStatus.STORED
    assert len(output_valid.tomes) > 0

    # Test 3: 99-char content (just below floor) should be REJECTED
    content_below = "b" * 99
    output_below = await ingestor.ingest(content_below)
    assert output_below.status == IngestStatus.REJECTED
    assert "minimum" in output_below.reject_reason.lower()
