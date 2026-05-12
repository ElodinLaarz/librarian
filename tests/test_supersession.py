"""TDD tests for tome supersession (issue #44).

Tests the soft-delete strategy: when a new tome supersedes an old one,
the old tome is marked with superseded_by field but not hard-deleted,
and search filters it out by default (include_superseded=False).
"""

from uuid import uuid4

import pytest

from src.models.enums import SourceType
from src.models.tome import Tome
from src.storage.tome_repository import TomeRepository


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
