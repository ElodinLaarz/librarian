"""Tests for the ingest pipeline (agent ↔ Librarian MCP boundary)."""

from __future__ import annotations

import uuid

import numpy as np
import pytest

from src import constants
from src.config import EmbeddingSettings, LibrarianConfig, VerificationSettings
from src.models.enums import IngestStatus, SourceType, VerificationVerdict
from src.models.tome import Tome
from src.services.verifier import ClaimResult, VerificationResult
from tests.conftest import make_test_config
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
        embedding=np.zeros(EmbeddingSettings().dimensions, dtype=np.float32),
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

        async def verify(self, content: str) -> VerificationResult:
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
    ingestor = StubIngestor(config, StubEmbeddingService(dimensions=config.embedding.dimensions), VariableVerifier(), repo)

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
    config = make_test_config(verification=strict_settings)
    repo = StubTomeRepository()
    ingestor, _, _ = make_stub_ingestor(config=config, confidence=0.5, repo=repo)

    output = await ingestor.ingest("Borderline content.")

    assert output.status == IngestStatus.REJECTED
    assert len(repo.all_tomes()) == 0


async def test_confidence_at_threshold_boundary_accepted() -> None:
    """Confidence exactly equal to reject_threshold is accepted (ge, not gt)."""
    threshold = 0.3
    config = make_test_config(verification=VerificationSettings(reject_threshold=threshold))
    repo = StubTomeRepository()
    ingestor, _, _ = make_stub_ingestor(config=config, confidence=threshold, repo=repo)

    output = await ingestor.ingest("Borderline content.")

    assert output.status == IngestStatus.STORED


# ── verification.enabled = False ─────────────────────────────────────────────


async def test_disabled_verification_stores_despite_low_confidence() -> None:
    """When verification is disabled, content is stored regardless of confidence."""
    config = make_test_config(
        verification=VerificationSettings(enabled=False, reject_threshold=0.99)
    )
    repo = StubTomeRepository()
    # StubVerifier would return 0.05 confidence — normally rejected, but verification is off.
    ingestor, _, _ = make_stub_ingestor(config=config, confidence=0.05, repo=repo)

    output = await ingestor.ingest("Content that would fail if verified.")

    assert output.status == IngestStatus.STORED
    assert len(repo.all_tomes()) == 1


async def test_disabled_verification_confidence_is_point_five() -> None:
    """Tomes stored without verification carry the default unverified confidence."""
    config = make_test_config(verification=VerificationSettings(enabled=False))
    repo = StubTomeRepository()
    ingestor, _, _ = make_stub_ingestor(config=config, confidence=0.5, repo=repo)

    output = await ingestor.ingest("Unverified fact.")

    assert output.tomes[0].confidence == pytest.approx(constants.DEFAULT_UNVERIFIED_CONFIDENCE)


# ── reshard safety ────────────────────────────────────────────────────────────


async def test_reshard_aborts_without_data_loss_when_all_replacements_rejected(
    config: LibrarianConfig,
) -> None:
    """If all reshard chunks fail verification, existing tomes are preserved."""
    repo = StubTomeRepository()
    existing = _make_tome("Existing valuable fact.")
    repo.seed_near_duplicates([existing])

    # Verifier always rejects, so no replacements will be built.
    ingestor, _, _ = make_stub_ingestor(config=config, confidence=0.0, repo=repo)

    output = await ingestor.ingest("New content (will fail verification).")

    # The old tome must still be in the repo — no data was lost.
    assert await repo.get_by_id(existing.id) is not None
    # The ingest result carries no new tomes for this chunk.
    assert existing.id not in {t.id for t in output.tomes}


async def test_reshard_aborts_without_data_loss_when_chunk_returns_empty(
    config: LibrarianConfig,
) -> None:
    """If _chunk returns nothing for the combined content, existing tomes are preserved."""
    repo = StubTomeRepository()
    existing = _make_tome("Existing fact.")
    repo.seed_near_duplicates([existing])

    class EmptyChunkIngestor(StubIngestor):
        _chunked_once = False

        async def _reshard(self, blob: str) -> list[str]:
            # First call (the original blob) returns one chunk so ingest proceeds.
            # Subsequent calls (reshard) return nothing.
            if not self._chunked_once:
                self._chunked_once = True
                return [blob.strip()]
            return []

    ingestor = EmptyChunkIngestor(
        config, StubEmbeddingService(), StubVerifier(confidence=0.9), repo
    )

    await ingestor.ingest("Incoming content.")

    # Existing tome must still be present.
    assert await repo.get_by_id(existing.id) is not None


async def test_reshard_returns_partial_status_on_delete_failure() -> None:
    """A failed tome deletion during reshard returns IngestOutput with error rather than raising."""
    repo = StubTomeRepository(fail_deletes=True)
    existing = _make_tome("Fact that cannot be deleted.")
    repo.seed_near_duplicates([existing])

    # Wire a fresh ingestor that uses the fail-deletes repo.
    test_config = make_test_config()
    bad_repo_ingestor = StubIngestor(
        test_config, StubEmbeddingService(dimensions=test_config.embedding.dimensions), StubVerifier(confidence=0.9), repo
    )

    output = await bad_repo_ingestor.ingest("Replacement content.")

    # Still stored the new tomes!
    assert output.status == IngestStatus.PARTIAL
    assert output.reject_reason is not None
    assert "Failed to delete tomes" in output.reject_reason
    assert len(output.tomes) >= 1


async def test_dedup_store_returning_empty_sets_partial_status(
    config: LibrarianConfig,
) -> None:
    """A shard whose _dedup_and_store returns [] must mark any_rejected=True.

    Previously an empty-list result fell into the `else` branch of the ingest
    loop, silently extending stored with nothing and leaving any_rejected False,
    so a single-shard ingest where all replacements were rejected could
    incorrectly return STORED.
    """
    repo = StubTomeRepository()
    existing = _make_tome("Original content.")
    repo.seed_near_duplicates([existing])

    # confidence=0.0 → all replacement shards are rejected by the verifier →
    # _dedup_and_store returns [] for the one shard that reaches it.
    ingestor, _, _ = make_stub_ingestor(config=config, confidence=0.0, repo=repo)

    output = await ingestor.ingest("Incoming content that triggers a reshard.")

    assert output.status != IngestStatus.STORED
