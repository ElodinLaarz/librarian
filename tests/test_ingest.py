"""Tests for the ingest pipeline (agent ↔ Librarian MCP boundary)."""

from __future__ import annotations

import uuid

import numpy as np
import pytest

from src.config import LibrarianConfig, VerificationSettings
from src.models.enums import IngestStatus, SourceType, VerificationVerdict
from src.models.tome import Tome
from src.services.verifier import ClaimResult, VerificationResult
from tests.stubs import (
    StubEmbeddingService,
    StubIngestor,
    StubTomeRepository,
    StubVerifier,
    make_stub_ingestor,
)

# ── helpers ──────────────────────────────────────────────────────────────────


def _make_tome(content: str, confidence: float = 0.8) -> Tome:
    """Build a minimal Tome for use in near-duplicate seeding."""
    return Tome(
        id=uuid.uuid4(),
        title="Existing Tome",
        content=content,
        summary="Summary",
        category="general",
        tags=["stub"],
        source_url=None,
        source_type=SourceType.AGENT_INPUT,
        confidence=confidence,
        embedding=np.zeros(768, dtype=np.float32),
    )


# ── single-chunk happy path ───────────────────────────────────────────────────


async def test_single_chunk_stores_one_tome(
    ingestor: StubIngestor, repo: StubTomeRepository
) -> None:
    output = await ingestor.ingest("The sky is blue.")

    assert output.status == IngestStatus.STORED
    assert output.reject_reason is None
    assert len(output.tomes) == 1
    assert len(repo.all_tomes()) == 1


async def test_tome_fields_populated(ingestor: StubIngestor, repo: StubTomeRepository) -> None:
    output = await ingestor.ingest("Photosynthesis converts sunlight into glucose.")

    tome = output.tomes[0]
    assert tome.content == "Photosynthesis converts sunlight into glucose."
    assert tome.category == "general"
    assert tome.tags == ["stub"]
    assert tome.source_type == SourceType.AGENT_INPUT
    assert tome.title != ""
    assert tome.summary != ""


async def test_confidence_from_verifier_stored_on_tome(
    config: LibrarianConfig, repo: StubTomeRepository
) -> None:
    ingestor, _, _ = make_stub_ingestor(config=config, confidence=0.92, repo=repo)
    output = await ingestor.ingest("Water boils at 100 °C at sea level.")

    assert output.tomes[0].confidence == pytest.approx(0.92)


# ── multi-chunk splitting ────────────────────────────────────────────────────


async def test_multi_chunk_produces_multiple_tomes(
    ingestor: StubIngestor, repo: StubTomeRepository
) -> None:
    blob = "Fact one is important.\n\nFact two is also important."
    output = await ingestor.ingest(blob)

    assert output.status == IngestStatus.STORED
    assert len(output.tomes) == 2
    assert len(repo.all_tomes()) == 2


async def test_multi_chunk_content_matches_split(
    ingestor: StubIngestor,
) -> None:
    blob = "Alpha fact.\n\nBeta fact.\n\nGamma fact."
    output = await ingestor.ingest(blob)

    contents = {t.content for t in output.tomes}
    assert contents == {"Alpha fact.", "Beta fact.", "Gamma fact."}


# ── rejection paths ───────────────────────────────────────────────────────────


async def test_empty_content_rejected(ingestor: StubIngestor) -> None:
    output = await ingestor.ingest("")

    assert output.status == IngestStatus.REJECTED
    assert output.reject_reason is not None
    assert output.tomes == []


async def test_whitespace_only_rejected(ingestor: StubIngestor) -> None:
    output = await ingestor.ingest("   \n\n   ")

    assert output.status == IngestStatus.REJECTED
    assert output.tomes == []


async def test_verification_below_threshold_rejected(
    config: LibrarianConfig, repo: StubTomeRepository
) -> None:
    # reject_threshold default is 0.3; set confidence just below it
    ingestor, _, _ = make_stub_ingestor(config=config, confidence=0.1, repo=repo)
    output = await ingestor.ingest("Some content that fails verification.")

    assert output.status == IngestStatus.REJECTED
    assert output.tomes == []
    assert len(repo.all_tomes()) == 0


async def test_partial_status_when_some_chunks_fail(
    config: LibrarianConfig,
) -> None:
    """When some chunks pass and some fail verification, status is PARTIAL."""

    class VariableVerifier(StubVerifier):
        """Alternates between high and low confidence per call."""

        def __init__(self) -> None:
            self._call_count = 0

        async def verify(self, content: str) -> VerificationResult:  # type: ignore[override]
            self._call_count += 1
            confidence = 0.9 if self._call_count % 2 == 1 else 0.05
            return VerificationResult(
                confidence=confidence,
                claims=[
                    ClaimResult(
                        claim="c",
                        verdict=VerificationVerdict.SUPPORTED,
                        evidence="e",
                    )
                ],
                skipped=False,
            )

    repo = StubTomeRepository()
    ingestor = StubIngestor(config, StubEmbeddingService(), VariableVerifier(), repo)

    output = await ingestor.ingest("Good chunk.\n\nBad chunk (low confidence).")

    assert output.status == IngestStatus.PARTIAL
    assert len(output.tomes) == 1  # only the first chunk passed


# ── dedup / reshard paths ────────────────────────────────────────────────────


async def test_no_duplicate_inserts_directly(
    ingestor: StubIngestor, repo: StubTomeRepository
) -> None:
    """With no near-duplicates seeded, content is inserted without resharding."""
    output = await ingestor.ingest("A brand new unique fact.")

    assert output.status == IngestStatus.STORED
    assert len(repo.all_tomes()) == 1


async def test_duplicate_triggers_reshard_deletes_old(
    ingestor: StubIngestor, repo: StubTomeRepository
) -> None:
    """When a near-duplicate exists, the old tome is deleted during reshard."""
    existing = _make_tome("Existing fact about gravity.")
    repo.seed_near_duplicates([existing])

    await ingestor.ingest("New fact about gravity with extra detail.")

    # The original tome must have been deleted.
    assert await repo.get_by_id(existing.id) is None


async def test_duplicate_reshard_stores_combined_content(
    ingestor: StubIngestor, repo: StubTomeRepository
) -> None:
    """After reshard, the repository contains tomes derived from combined content."""
    existing = _make_tome("Fact A.")
    repo.seed_near_duplicates([existing])

    await ingestor.ingest("Fact B.")

    remaining = repo.all_tomes()
    # At least one tome should exist after the reshard.
    assert len(remaining) >= 1
    # Original ID must be gone.
    assert not any(t.id == existing.id for t in remaining)


async def test_reshard_with_multiple_duplicates(
    ingestor: StubIngestor, repo: StubTomeRepository
) -> None:
    """Multiple near-duplicates are all deleted and their content is resharded."""
    dup_a = _make_tome("Alpha.")
    dup_b = _make_tome("Beta.")
    repo.seed_near_duplicates([dup_a, dup_b])

    await ingestor.ingest("Gamma.")

    assert await repo.get_by_id(dup_a.id) is None
    assert await repo.get_by_id(dup_b.id) is None
    # Combined "Alpha.\n\nBeta.\n\nGamma." → 3 chunks via StubIngestor._chunk
    assert len(repo.all_tomes()) == 3


async def test_reshard_does_not_loop_infinitely(
    ingestor: StubIngestor, repo: StubTomeRepository
) -> None:
    """Deleting the old tome removes it from near-duplicate results; no infinite loop."""
    existing = _make_tome("Loop bait.")
    repo.seed_near_duplicates([existing])

    # Should complete without recursion errors.
    output = await ingestor.ingest("New content that would match.")
    assert output.status in {IngestStatus.STORED, IngestStatus.PARTIAL}


# ── custom reject threshold ───────────────────────────────────────────────────


async def test_custom_reject_threshold_honoured() -> None:
    """Setting a high reject_threshold causes moderate-confidence content to be rejected."""
    strict_settings = VerificationSettings(reject_threshold=0.9)
    config = LibrarianConfig(verification=strict_settings)
    repo = StubTomeRepository()
    ingestor, _, _ = make_stub_ingestor(config=config, confidence=0.5, repo=repo)

    output = await ingestor.ingest("Borderline content.")

    assert output.status == IngestStatus.REJECTED
    assert len(repo.all_tomes()) == 0


async def test_confidence_at_threshold_boundary_accepted() -> None:
    """Confidence exactly equal to reject_threshold is accepted (ge, not gt)."""
    threshold = 0.3
    config = LibrarianConfig(verification=VerificationSettings(reject_threshold=threshold))
    repo = StubTomeRepository()
    ingestor, _, _ = make_stub_ingestor(config=config, confidence=threshold, repo=repo)

    output = await ingestor.ingest("Borderline content.")

    assert output.status == IngestStatus.STORED
