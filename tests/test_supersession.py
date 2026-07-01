"""TDD tests for tome supersession (issue #44).

Tests the soft-delete strategy: when a new tome supersedes an old one,
the old tome is marked with superseded_by field but not hard-deleted,
and search filters it out by default (include_superseded=False).
"""

from uuid import UUID, uuid4

import pytest

from src.models.enums import IngestStatus, SourceType
from src.models.tome import Tome
from src.services.ingestor import IngestCallOptions
from src.storage.errors import StorageError
from src.storage.tome_repository import TomeRepository
from tests.stubs import StubIngestor, StubTomeRepository


@pytest.mark.asyncio
async def test_supersession_filtering(
    repo: TomeRepository,
) -> None:
    """Test that superseded tomes are filtered from search by default.

    RED TEST: Creates tome A (Python 3.8 EOL: 2021), then creates and ingests
    tome B (Python 3.8 EOL: 2024) with supersedes_tome_ids=[A.id].

    Asserts:
    - search("Python EOL", include_superseded=False) returns B only
    - search("Python EOL", include_superseded=True) returns both A and B
    - tome A has superseded_by == B.id
    """

    # Create tome A
    tome_a = Tome(
        id=uuid4(),
        title="Python 3.8 End of Life",
        content="Python 3.8 reaches end of life on 2021-10-04.",
        summary="Python 3.8 EOL: 2021",
        category="programming",
        source_type=SourceType.AGENT_INPUT,
        confidence=0.95,
        tags=["python", "eol"],
    )
    await repo.insert(tome_a)

    # Create tome B (supersedes A)
    tome_b = Tome(
        id=uuid4(),
        title="Python 3.8 End of Life (Updated)",
        content="Python 3.8 reaches end of life on 2024-10-04 (extended support).",
        summary="Python 3.8 EOL: 2024",
        category="programming",
        source_type=SourceType.AGENT_INPUT,
        confidence=0.98,
        tags=["python", "eol"],
    )
    await repo.insert(tome_b)

    # Mark A as superseded by B
    success = await repo.mark_superseded(tome_a.id, tome_b.id)
    assert success, "mark_superseded should return True"

    # Verify A's superseded_by field is set
    tome_a_fetched = await repo.get_by_id(tome_a.id)
    assert tome_a_fetched is not None
    assert tome_a_fetched.superseded_by == tome_b.id, "A.superseded_by should equal B.id"

    # Search without include_superseded (default False) should return B only
    results = await repo.search(
        query="Python EOL",
        top_k=10,
        min_confidence=0.0,
        category=None,
        include_superseded=False,
    )
    result_ids = [t.id for t, _ in results]
    assert tome_b.id in result_ids, "Search should include superseding tome B"
    assert tome_a.id not in result_ids, "Search should NOT include superseded tome A (default)"

    # Search with include_superseded=True should return both A and B
    results_with_superseded = await repo.search(
        query="Python EOL",
        top_k=10,
        min_confidence=0.0,
        category=None,
        include_superseded=True,
    )
    result_ids_with_superseded = [t.id for t, _ in results_with_superseded]
    assert tome_b.id in result_ids_with_superseded, "Should include B with include_superseded=True"
    assert tome_a.id in result_ids_with_superseded, "Should include A with include_superseded=True"


@pytest.mark.asyncio
async def test_ingest_reports_partial_when_supersede_target_missing(
    ingestor: StubIngestor,
) -> None:
    """A supersede target that does not exist must degrade the status to partial.

    Previously the failure was logged and swallowed: the response said
    ``stored`` while the caller believed the old tome had been retired — it
    remained live in every search.
    """
    output = await ingestor.ingest(
        "Water boils at 100 C at sea level.",
        IngestCallOptions(skip_verify=True, supersedes_tome_ids=[str(uuid4())]),
    )
    assert output.tomes, "the new content itself must still be stored"
    assert output.status == IngestStatus.PARTIAL
    assert output.reject_reason is not None
    assert "not found" in output.reject_reason


@pytest.mark.asyncio
async def test_ingest_reports_partial_when_supersede_raises_storage_error(
    ingestor: StubIngestor,
    repo: StubTomeRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A storage failure during supersession must surface in the response."""

    async def failing_mark_superseded(tome_id: UUID, by_tome_id: UUID) -> bool:
        raise StorageError("mark_superseded backend down")

    monkeypatch.setattr(repo, "mark_superseded", failing_mark_superseded)

    output = await ingestor.ingest(
        "Water boils at 100 C at sea level.",
        IngestCallOptions(skip_verify=True, supersedes_tome_ids=[str(uuid4())]),
    )
    assert output.tomes, "the new content itself must still be stored"
    assert output.status == IngestStatus.PARTIAL
    assert output.reject_reason is not None
    assert "superseded" in output.reject_reason
    assert "backend down" in output.reject_reason
