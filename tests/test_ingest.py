"""Tests for the ingest pipeline (agent ↔ Librarian MCP boundary)."""

from __future__ import annotations

import ast
import json
import uuid
from typing import Any
from unittest.mock import patch

import httpx
import numpy as np
import pytest
import yaml

from src import constants
from src.config import (
    EmbeddingSettings,
    IngestSettings,
    LibrarianConfig,
    VerificationSettings,
)
from src.models.enums import IngestStatus, SourceType, VerificationVerdict
from src.models.tome import Tome
from src.services.ingestor import IngestCallOptions, Ingestor
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


# ── verification summary persistence (issue #41) ──────────────────────────────


async def test_verification_summary_persisted_on_tome(
    config: LibrarianConfig, repo: StubTomeRepository
) -> None:
    """A verified ingest stores a VerificationSummary on the resulting Tome.

    The StubVerifier always returns one SUPPORTED claim; the persisted summary
    must reflect that count exactly, not be ``None``.
    """
    ingestor, _, _ = make_stub_ingestor(config=config, confidence=0.92, repo=repo)
    output = await ingestor.ingest("Water boils at 100 C at sea level.")

    tome = output.tomes[0]
    assert tome.verification is not None
    assert tome.verification.skipped is False
    assert tome.verification.claim_count == 1
    assert tome.verification.supported == 1
    assert tome.verification.contradicted == 0
    assert tome.verification.unverifiable == 0
    # The stub's evidence is the literal "stub evidence" — not a URL — so
    # nothing should be captured.
    assert tome.verification.evidence_urls == []


async def test_verification_summary_marks_skipped_when_verification_disabled() -> None:
    """With verification.enabled=False, the persisted summary records skipped=True."""
    config = make_test_config(verification=VerificationSettings(enabled=False))
    repo = StubTomeRepository()
    ingestor, _, _ = make_stub_ingestor(config=config, confidence=0.5, repo=repo)

    output = await ingestor.ingest("Unverified fact.")

    tome = output.tomes[0]
    assert tome.verification is not None
    assert tome.verification.skipped is True
    assert tome.verification.claim_count == 0


async def test_verification_summary_marks_skipped_when_skip_verify_flag_set(
    config: LibrarianConfig,
) -> None:
    """A per-request skip_verify=True flag still produces a summary with skipped=True."""
    repo = StubTomeRepository()
    ingestor = Ingestor(
        config,
        StubEmbeddingService(dimensions=config.embedding.dimensions),
        StubVerifier(confidence=0.01),
        repo,
    )
    output = await ingestor.ingest(
        "Skip-verify content.",
        IngestCallOptions(skip_verify=True),
    )

    tome = output.tomes[0]
    assert tome.verification is not None
    assert tome.verification.skipped is True
    assert tome.verification.claim_count == 0


async def test_verification_summary_survives_search(
    config: LibrarianConfig, repo: StubTomeRepository
) -> None:
    """Persist a tome via ingest, then read it back via the repo.search path
    and confirm the verification summary survives the round trip.

    This is the search-surface half of acceptance for issue #41: agents need
    the summary to be visible from ``library_search`` output, not just from
    the in-memory Tome returned by ingest.
    """
    ingestor, _, _ = make_stub_ingestor(config=config, confidence=0.92, repo=repo)
    await ingestor.ingest("Water boils at 100 C at sea level.")

    hits = await repo.search(query="boil", top_k=5, min_confidence=0.0)
    assert hits, "Expected at least one search hit"
    found = hits[0][0]
    assert found.verification is not None
    assert found.verification.skipped is False
    assert found.verification.claim_count == 1
    assert found.verification.supported == 1


async def test_tome_without_verification_field_loads_as_none() -> None:
    """Existing serialized tomes without the ``verification`` key must round-trip.

    Backward-compat for the migration case called out in issue #41: tomes
    written before this field existed should load with ``verification=None``,
    not raise a validation error.
    """
    legacy_json = (
        '{"id": "00000000-0000-4000-8000-000000000001",'
        '"title": "Legacy",'
        '"content": "Old fact.",'
        '"summary": "Old.",'
        '"category": "general",'
        '"tags": [],'
        '"source_url": null,'
        '"source_type": "agent_input",'
        '"confidence": 0.8,'
        '"research_job_id": null,'
        '"embedding": null,'
        '"created_at": "2024-01-01T00:00:00Z"}'
    )
    legacy = Tome.model_validate_json(legacy_json)
    assert legacy.verification is None


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
    ingestor = StubIngestor(
        config,
        StubEmbeddingService(dimensions=config.embedding.dimensions),
        VariableVerifier(),
        repo,
    )

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


async def test_skip_verify_bypasses_stub_verifier(config: LibrarianConfig) -> None:
    """Per-request skip_verify stores content even if the verifier would reject."""
    repo = StubTomeRepository()
    ingestor = Ingestor(
        config,
        StubEmbeddingService(dimensions=config.embedding.dimensions),
        StubVerifier(confidence=0.01),
        repo,
    )
    output = await ingestor.ingest(
        "Content that would be rejected if verified.",
        IngestCallOptions(skip_verify=True),
    )
    assert output.status == IngestStatus.STORED
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

        async def _reshard(self, blob: str, opts: IngestCallOptions | None = None) -> list[str]:
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


async def test_reshard_rolls_back_partial_inserts_on_failure() -> None:
    """If a replacement insert fails mid-reshard, previously-inserted replacements
    must be compensating-deleted, originals must remain, and ingest must report
    REJECTED with a reason mentioning the insert abort."""
    test_config = make_test_config()

    # Pre-existing duplicates that the new content will collide with.
    original_a = _make_tome("Original fact A.")
    original_b = _make_tome("Original fact B.")

    # Stub repo: second insert call raises, simulating a mid-flight failure.
    repo = StubTomeRepository(fail_inserts_after=1)
    repo.seed_near_duplicates([original_a, original_b])

    ingestor = StubIngestor(
        test_config,
        StubEmbeddingService(dimensions=test_config.embedding.dimensions),
        StubVerifier(confidence=0.9),
        repo,
    )

    # Two new shards (split on \n\n by StubIngestor._reshard) → combined content
    # produces multiple replacement tomes, so >1 inserts will be attempted.
    output = await ingestor.ingest("New shard one.\n\nNew shard two.")

    # (a) No replacement tome should remain — compensating delete must have run.
    original_ids = {original_a.id, original_b.id}
    leftover_replacements = [t for t in repo.all_tomes() if t.id not in original_ids]
    assert leftover_replacements == [], (
        f"Expected zero replacement tomes after rollback, found: {leftover_replacements}"
    )

    # (b) Both original duplicates must still be present (delete step never ran).
    assert await repo.get_by_id(original_a.id) is not None
    assert await repo.get_by_id(original_b.id) is not None

    # (c) Output reflects rejection with a reason mentioning the insert abort.
    assert output.status in (IngestStatus.REJECTED, IngestStatus.PARTIAL)
    assert output.reject_reason is not None
    assert "insert" in output.reject_reason.lower()


async def test_reshard_returns_partial_status_on_delete_failure() -> None:
    """A failed tome deletion during reshard returns IngestOutput with error rather than raising."""
    repo = StubTomeRepository(fail_deletes=True)
    existing = _make_tome("Fact that cannot be deleted.")
    repo.seed_near_duplicates([existing])

    # Wire a fresh ingestor that uses the fail-deletes repo.
    test_config = make_test_config()
    bad_repo_ingestor = StubIngestor(
        test_config,
        StubEmbeddingService(dimensions=test_config.embedding.dimensions),
        StubVerifier(confidence=0.9),
        repo,
    )

    output = await bad_repo_ingestor.ingest("Replacement content.")

    # Still stored the new tomes!
    assert output.status == IngestStatus.PARTIAL
    assert output.reject_reason is not None
    assert "Failed to delete tomes" in output.reject_reason
    assert len(output.tomes) >= 1


# ── persistent http client ────────────────────────────────────────────────────


async def test_http_client_lazy_until_first_use(config: LibrarianConfig) -> None:
    """The persistent httpx client is not constructed at __init__ time."""
    repo = StubTomeRepository()
    ingestor = Ingestor(
        config,
        StubEmbeddingService(dimensions=config.embedding.dimensions),
        StubVerifier(confidence=0.9),
        repo,
    )
    assert ingestor._http_client is None
    await ingestor.aclose()  # safe no-op when never created


async def test_extract_facts_llm_reuses_single_http_client(
    monkeypatch: pytest.MonkeyPatch, config: LibrarianConfig
) -> None:
    """Multiple `_extract_facts_llm` calls share one `httpx.AsyncClient` instance."""
    from unittest.mock import MagicMock

    import httpx as _httpx

    posted_clients: list[object] = []

    async def _fake_post(self: _httpx.AsyncClient, url: str, **kwargs: object) -> MagicMock:
        posted_clients.append(self)
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json = MagicMock(
            return_value={
                "choices": [{"message": {"content": '{"facts": ["fact one.", "fact two."]}'}}]
            }
        )
        return response

    monkeypatch.setattr("httpx.AsyncClient.post", _fake_post)

    repo = StubTomeRepository()
    ingestor = Ingestor(
        config,
        StubEmbeddingService(dimensions=config.embedding.dimensions),
        StubVerifier(confidence=0.9),
        repo,
    )
    try:
        first = await ingestor._extract_facts_llm("Some text to shard.")
        client_after_first = ingestor._http_client
        second = await ingestor._extract_facts_llm("More text to shard.")
        client_after_second = ingestor._http_client

        assert first == ["fact one.", "fact two."]
        assert second == ["fact one.", "fact two."]
        assert client_after_first is not None
        assert client_after_first is client_after_second
        assert len(posted_clients) == 2
        assert posted_clients[0] is posted_clients[1]
        assert posted_clients[0] is client_after_first
    finally:
        await ingestor.aclose()


async def test_aclose_closes_and_resets_http_client(
    monkeypatch: pytest.MonkeyPatch, config: LibrarianConfig
) -> None:
    """`aclose` closes the persistent client and resets the slot to None."""
    from unittest.mock import MagicMock

    import httpx as _httpx

    async def _fake_post(self: _httpx.AsyncClient, url: str, **kwargs: object) -> MagicMock:
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json = MagicMock(
            return_value={"choices": [{"message": {"content": '{"facts": ["a."]}'}}]}
        )
        return response

    monkeypatch.setattr("httpx.AsyncClient.post", _fake_post)

    repo = StubTomeRepository()
    ingestor = Ingestor(
        config,
        StubEmbeddingService(dimensions=config.embedding.dimensions),
        StubVerifier(confidence=0.9),
        repo,
    )
    await ingestor._extract_facts_llm("Text.")
    client = ingestor._http_client
    assert client is not None
    assert not client.is_closed

    await ingestor.aclose()

    assert ingestor._http_client is None
    assert client.is_closed


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


# ── minimum tome size enforcement (issue #35) ────────────────────────────────


def _word_count(text: str) -> int:
    return len(text.split())


async def test_llm_short_facts_are_bundled() -> None:
    """LLM-emitted sub-floor facts must be merged so each tome meets the floor."""
    from src.config import IngestSettings

    # Use a small floor so 5 short facts can be bundled into multiple tomes.
    ingest = IngestSettings(min_shard_chars=40, min_shard_words=8, use_llm_chunking=True)
    cfg = make_test_config(ingest=ingest)

    long_blob = "y" * (ingest.min_shard_chars * 5)

    short_facts = [
        "fact one is here and pretty short though.",
        "fact two is here and pretty short though.",
        "fact three is here and pretty short.",
        "fact four is here and pretty short.",
        "fact five is here and pretty short.",
    ]

    class LLMShortFactsIngestor(StubIngestor):
        async def _extract_facts_llm(self, blob: str) -> list[str] | None:
            return list(short_facts)

        async def _reshard(self, blob: str, opts: IngestCallOptions | None = None) -> list[str]:
            # Delegate to production _reshard so _apply_min_floor bundling runs.
            return await Ingestor._reshard(self, blob, opts)

    repo = StubTomeRepository()
    ingestor = LLMShortFactsIngestor(
        cfg,
        StubEmbeddingService(dimensions=cfg.embedding.dimensions),
        StubVerifier(confidence=0.9),
        repo,
    )

    output = await ingestor.ingest(long_blob)

    assert output.status == IngestStatus.STORED
    # Every stored tome must meet the floor.
    for tome in output.tomes:
        assert (
            len(tome.content) >= ingest.min_shard_chars
            and _word_count(tome.content) >= ingest.min_shard_words
        ), f"tome below floor: chars={len(tome.content)}, words={_word_count(tome.content)}"
    # Total content must preserve all five facts (no data loss).
    combined = " ".join(t.content for t in output.tomes)
    for fact in short_facts:
        assert fact in combined


async def test_all_llm_shards_below_floor_falls_back_to_heuristic(
    config: LibrarianConfig,
) -> None:
    """If LLM returns only sub-floor fragments and bundling cannot help, fall back."""
    long_blob = "alpha beta gamma delta epsilon. " * 200  # well over the floor

    splitter_called = {"count": 0}

    class FallbackIngestor(StubIngestor):
        async def _reshard_llm(self, blob: str) -> list[str] | None:
            # Return a single tiny fragment that cannot satisfy the floor and
            # is much smaller than the blob → bundling can't recover it.
            return ["tiny."]

        async def _reshard(self, blob: str, opts: IngestCallOptions | None = None) -> list[str]:
            return await Ingestor._reshard(self, blob, opts)

    # Patch the splitter symbol that ingestor.py actually uses. The production
    # code imports RecursiveCharacterTextSplitter at module load, so patching
    # the langchain_text_splitters module attribute would no longer affect it.
    from src.services import ingestor as _ingestor_mod

    original_cls = _ingestor_mod.RecursiveCharacterTextSplitter

    class _SpySplitter(original_cls):  # type: ignore[misc, valid-type]
        def split_text(self, text: str) -> list[str]:
            splitter_called["count"] += 1
            return super().split_text(text)

    _ingestor_mod.RecursiveCharacterTextSplitter = _SpySplitter
    try:
        repo = StubTomeRepository()
        ingestor = FallbackIngestor(
            config,
            StubEmbeddingService(dimensions=config.embedding.dimensions),
            StubVerifier(confidence=0.9),
            repo,
        )
        output = await ingestor.ingest(long_blob)
    finally:
        _ingestor_mod.RecursiveCharacterTextSplitter = original_cls

    assert splitter_called["count"] >= 1
    assert output.status == IngestStatus.STORED


async def test_heuristic_tail_shard_below_floor_merged_with_previous(
    config: LibrarianConfig,
) -> None:
    """Heuristic-path tail fragment below the floor must be bundled with the previous shard."""
    # Build a config that disables LLM chunking and uses a small shard_size so we
    # exercise the splitter without a giant blob.
    from src.config import IngestSettings

    ingest = IngestSettings(
        use_llm_chunking=False,
        shard_size=400,
        shard_overlap=0,
        min_shard_chars=300,
        min_shard_words=40,
    )
    cfg = make_test_config(ingest=ingest)

    # Produce a blob slightly larger than shard_size so the splitter creates two
    # shards: one near shard_size, plus a tiny tail.
    blob = ("alpha. " * 60) + "tail."  # 60 * 7 = 420 chars + tail
    assert len(blob) > ingest.shard_size

    repo = StubTomeRepository()
    ingestor = Ingestor(
        cfg,
        StubEmbeddingService(dimensions=cfg.embedding.dimensions),
        StubVerifier(confidence=0.9),
        repo,
    )

    output = await ingestor.ingest(blob)

    assert output.status == IngestStatus.STORED
    assert len(output.tomes) == 1, (
        f"expected tail to be merged with previous, got {len(output.tomes)} tomes: "
        f"{[len(t.content) for t in output.tomes]}"
    )


async def test_validate_rejects_below_floor() -> None:
    """_validate must raise when content is below the floor and allow_short=False."""
    from src.config import IngestSettings

    floor_config = make_test_config(ingest=IngestSettings(min_shard_chars=400, min_shard_words=80))
    repo = StubTomeRepository()
    ingestor = Ingestor(
        floor_config,
        StubEmbeddingService(dimensions=floor_config.embedding.dimensions),
        StubVerifier(confidence=0.9),
        repo,
    )

    short_content = "too short."  # well below floor
    tome = Tome(
        id=uuid.uuid4(),
        title="t",
        content=short_content,
        summary="s",
        category="general",
        tags=["stub"],
        source_url=None,
        source_type=SourceType.AGENT_INPUT,
        confidence=0.9,
        embedding=np.zeros(floor_config.embedding.dimensions, dtype=np.float32),
    )

    with pytest.raises(ValueError):
        ingestor._validate(tome, allow_short=False)


async def test_legitimate_short_blob_passes_through() -> None:
    """A genuine short input (e.g. a tweet) must be stored as a single tome."""
    from src.config import IngestSettings

    floor_config = make_test_config(ingest=IngestSettings(min_shard_chars=400, min_shard_words=80))
    short_blob = "Cats sleep ~16 hours per day."  # ~30 chars
    assert len(short_blob) < floor_config.ingest.min_shard_chars

    repo = StubTomeRepository()
    ingestor = Ingestor(
        floor_config,
        StubEmbeddingService(dimensions=floor_config.embedding.dimensions),
        StubVerifier(confidence=0.9),
        repo,
    )

    output = await ingestor.ingest(short_blob)

    assert output.status == IngestStatus.STORED
    assert len(output.tomes) == 1
    assert output.tomes[0].content.strip() == short_blob


# ── LLM-based title and summary generation (issue #36) ───────────────────────


def _make_summary_ingestor(
    config: LibrarianConfig, repo: StubTomeRepository | None = None
) -> tuple[Ingestor, StubTomeRepository]:
    """Build a *real* Ingestor (no _generate_title_and_summary override).

    Only verifier/embedder/repo are stubbed — _generate_title_and_summary and
    _reshard run their real implementations.
    """
    repo = repo or StubTomeRepository()
    ingestor = Ingestor(
        config,
        StubEmbeddingService(dimensions=config.embedding.dimensions),
        StubVerifier(confidence=0.9),
        repo,
    )
    return ingestor, repo


def _ollama_chat_response(content: str) -> httpx.Response:
    """Build a fake Ollama-compatible chat completion response."""
    body = {"choices": [{"message": {"content": content}}]}
    return httpx.Response(200, json=body, request=httpx.Request("POST", "http://x/y"))


async def test_summary_is_not_content_prefix_when_llm_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When use_llm_summary=True, summary must come from the LLM, not truncation."""
    ingest_settings = IngestSettings(
        use_llm_chunking=False,
        use_llm_summary=True,
        use_llm_classification=False,
        min_shard_chars=0,  # Tests allow arbitrary content length
        min_shard_words=0,
    )
    config = make_test_config(
        verification=VerificationSettings(enabled=False),
        ingest=ingest_settings,
    )
    ingestor, repo = _make_summary_ingestor(config)

    llm_payload = json.dumps(
        {
            "title": "Photosynthesis basics",
            "summary": "Plants convert sunlight to glucose via photosynthesis.",
        }
    )

    async def fake_post(self: httpx.AsyncClient, url: str, **kwargs: object) -> httpx.Response:
        return _ollama_chat_response(llm_payload)

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    blob = "X" * config.ingest.summary_length + " then a long tail of additional content " * 5
    output = await ingestor.ingest(blob)

    assert output.status == IngestStatus.STORED
    assert len(output.tomes) == 1
    tome = output.tomes[0]
    assert tome.summary != tome.content[: config.ingest.summary_length]
    assert tome.summary == "Plants convert sunlight to glucose via photosynthesis."
    assert tome.title == "Photosynthesis basics"


async def test_summary_falls_back_on_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A network error causes fallback to truncation behaviour."""
    ingest_settings = IngestSettings(
        use_llm_chunking=False,
        use_llm_summary=True,
        use_llm_classification=False,
        min_shard_chars=0,  # Tests allow arbitrary content length
        min_shard_words=0,
    )
    config = make_test_config(
        verification=VerificationSettings(enabled=False),
        ingest=ingest_settings,
    )
    ingestor, _ = _make_summary_ingestor(config)

    async def fake_post(self: httpx.AsyncClient, url: str, **kwargs: object) -> httpx.Response:
        raise httpx.ConnectError("nope")

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    blob = "Z" * (config.ingest.summary_length + 50)
    output = await ingestor.ingest(blob)

    assert output.status == IngestStatus.STORED
    tome = output.tomes[0]
    assert tome.summary == tome.content[: config.ingest.summary_length]


async def test_summary_falls_back_on_bad_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-JSON LLM output causes fallback to truncation behaviour."""
    ingest_settings = IngestSettings(
        use_llm_chunking=False,
        use_llm_summary=True,
        use_llm_classification=False,
        min_shard_chars=0,  # Tests allow arbitrary content length
        min_shard_words=0,
    )
    config = make_test_config(
        verification=VerificationSettings(enabled=False),
        ingest=ingest_settings,
    )
    ingestor, _ = _make_summary_ingestor(config)

    async def fake_post(self: httpx.AsyncClient, url: str, **kwargs: object) -> httpx.Response:
        return _ollama_chat_response("not json at all")

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    blob = "Q" * (config.ingest.summary_length + 50)
    output = await ingestor.ingest(blob)

    assert output.status == IngestStatus.STORED
    tome = output.tomes[0]
    assert tome.summary == tome.content[: config.ingest.summary_length]


async def test_use_llm_summary_flag_off_skips_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """When use_llm_summary=False, the summary LLM endpoint must not be called."""
    # Disable classification too — it's an unrelated LLM path that would also
    # post to the same endpoint and false-positive this assertion.
    ingest_settings = IngestSettings(
        use_llm_chunking=False,
        use_llm_summary=False,
        use_llm_classification=False,
        min_shard_chars=0,  # Tests allow arbitrary content length
        min_shard_words=0,
    )
    config = make_test_config(
        verification=VerificationSettings(enabled=False),
        ingest=ingest_settings,
    )
    ingestor, _ = _make_summary_ingestor(config)

    async def fake_post(self: httpx.AsyncClient, url: str, **kwargs: object) -> httpx.Response:
        raise AssertionError("LLM endpoint must not be called when use_llm_summary=False")

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    blob = "M" * (config.ingest.summary_length + 50)
    output = await ingestor.ingest(blob)

    assert output.status == IngestStatus.STORED
    tome = output.tomes[0]
    assert tome.summary == tome.content[: config.ingest.summary_length]


# ── syntax-aware chunking (issue #47) ────────────────────────────────────────


def _make_real_ingestor(
    *,
    shard_size: int = 400,
    shard_overlap: int = 0,
    use_llm_chunking: bool = False,
) -> Ingestor:
    """Build a real Ingestor (not StubIngestor) to exercise _reshard logic."""
    config = make_test_config(
        ingest=IngestSettings(
            shard_size=shard_size,
            shard_overlap=shard_overlap,
            use_llm_chunking=use_llm_chunking,
        )
    )
    return Ingestor(
        config,
        StubEmbeddingService(dimensions=config.embedding.dimensions),
        StubVerifier(confidence=0.9),
        StubTomeRepository(),
    )


async def test_python_module_chunks_are_ast_parseable() -> None:
    """Each chunk of a multi-function Python module must be ast.parse-able.

    Generic character splitting truncates `def`/`class` mid-body. A
    language-aware splitter splits at structural boundaries.
    """
    blob = (
        "def alpha(x):\n"
        '    """First function."""\n'
        "    y = x + 1\n"
        "    z = y * 2\n"
        "    return z\n"
        "\n\n"
        "def beta(a, b):\n"
        '    """Second function."""\n'
        "    result = a * b\n"
        "    for i in range(10):\n"
        "        result += i\n"
        "    return result\n"
        "\n\n"
        "def gamma():\n"
        '    """Third function."""\n'
        "    data = [1, 2, 3, 4, 5]\n"
        "    return sum(data)\n"
    )
    ingestor = _make_real_ingestor(shard_size=120, shard_overlap=0)

    shards = await ingestor._reshard(blob, IngestCallOptions())

    assert shards, "Expected at least one shard"
    for s in shards:
        # Must parse as Python; raises SyntaxError if a def line was truncated.
        ast.parse(s)


async def test_markdown_preserves_fenced_code_blocks() -> None:
    """Each markdown chunk must have a balanced count of triple-backticks."""
    blob = (
        "# Heading\n\n"
        "Intro paragraph with explanation.\n\n"
        "```python\n"
        "def hello():\n"
        "    print('hello world')\n"
        "    return 42\n"
        "```\n\n"
        "Middle paragraph between code blocks for context.\n\n"
        "```python\n"
        "class Greeter:\n"
        "    def greet(self):\n"
        "        return 'hi'\n"
        "```\n\n"
        "Closing remarks.\n"
    )
    ingestor = _make_real_ingestor(shard_size=120, shard_overlap=0)

    shards = await ingestor._reshard(blob, IngestCallOptions())

    assert shards, "Expected at least one shard"
    for s in shards:
        assert s.count("```") % 2 == 0, f"Unbalanced fences in shard: {s!r}"


async def test_yaml_chunks_split_on_top_level_keys() -> None:
    """YAML with multiple top-level keys must split into yaml.safe_load-able chunks."""
    blob = (
        "alpha:\n"
        "  description: first key\n"
        "  value: 1\n"
        "\n"
        "beta:\n"
        "  description: second key\n"
        "  value: 2\n"
        "\n"
        "gamma:\n"
        "  description: third key\n"
        "  value: 3\n"
        "\n"
        "delta:\n"
        "  description: fourth key\n"
        "  value: 4\n"
        "\n"
        "epsilon:\n"
        "  description: fifth key\n"
        "  value: 5\n"
    )
    ingestor = _make_real_ingestor(shard_size=80, shard_overlap=0)

    shards = await ingestor._reshard(blob, IngestCallOptions())

    assert shards, "Expected at least one shard"
    for s in shards:
        loaded = yaml.safe_load(s)
        # Each shard must be parseable YAML (mapping or None/scalar).
        assert loaded is None or isinstance(loaded, dict | list | str | int | float | bool)


async def test_json_chunks_split_on_top_level_elements() -> None:
    """JSON array of N objects must split into chunks that each parse (when wrapped)."""
    objects = [{"id": i, "name": f"item-{i}", "value": i * 10} for i in range(10)]
    blob = json.dumps(objects, indent=2)
    ingestor = _make_real_ingestor(shard_size=200, shard_overlap=0)

    shards = await ingestor._reshard(blob, IngestCallOptions())

    assert shards, "Expected at least one shard"
    # Each shard must contain valid JSON object(s). Wrap in [] if needed for parsing.
    for s in shards:
        s_clean = s.strip().rstrip(",")
        # Try direct parse; otherwise wrap as array element(s).
        try:
            json.loads(s_clean)
        except json.JSONDecodeError:
            wrapped = f"[{s_clean.rstrip(',')}]"
            json.loads(wrapped)


async def test_json_array_greedy_packing_avoids_fragmentation() -> None:
    """Many small JSON array elements must be coalesced rather than emitted one-per-chunk.

    Per Gemini code-review feedback on PR #62: previously each array element
    became its own Tome, producing extreme fragmentation. The packer should
    yield far fewer shards than the element count for a generously-sized
    shard_size, while still respecting the size cap.
    """
    objects = [{"id": i, "n": i} for i in range(50)]
    blob = json.dumps(objects, indent=2)
    shard_size = 400
    ingestor = _make_real_ingestor(shard_size=shard_size, shard_overlap=0)

    shards = await ingestor._reshard(blob, IngestCallOptions())

    assert shards, "Expected at least one shard"
    # Should be far fewer than one-per-element (50). Be generous: at most ~half.
    assert len(shards) < len(objects) // 2, (
        f"Expected greedy packing; got {len(shards)} shards for {len(objects)} elements"
    )
    # Every shard must be a valid JSON array (or the size-guard fallback) and
    # within the size cap (allow small overshoot from the recursive splitter
    # boundary — but the packed-array path must respect the cap exactly).
    for s in shards:
        parsed = json.loads(s)
        assert isinstance(parsed, list)
        assert len(s) <= shard_size, f"Shard exceeds shard_size={shard_size}: {len(s)}"


async def test_force_format_override_routes_to_correct_splitter() -> None:
    """force_format='python' on a prose blob must invoke the PYTHON language splitter."""
    from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

    prose = (
        "This is a long prose paragraph about gardening. "
        "It contains no code at all and would normally be treated as text. "
        "However the caller has overridden the format detection."
    )
    ingestor = _make_real_ingestor(shard_size=80, shard_overlap=0)

    real_from_language = RecursiveCharacterTextSplitter.from_language
    seen_languages: list[Language] = []

    def spy(language: Language, **kwargs: object) -> RecursiveCharacterTextSplitter:
        seen_languages.append(language)
        return real_from_language(language, **kwargs)

    with patch.object(RecursiveCharacterTextSplitter, "from_language", side_effect=spy):
        shards = await ingestor._reshard(prose, IngestCallOptions(force_format="python"))

    assert shards
    assert Language.PYTHON in seen_languages, (
        f"Expected PYTHON splitter to be invoked; saw {seen_languages!r}"
    )


async def test_use_llm_chunking_skipped_for_detected_code() -> None:
    """When detected format is code, _extract_facts_llm must NOT be awaited."""
    blob = (
        "import os\n"
        "import sys\n"
        "\n"
        "def main():\n"
        "    print('hello')\n"
        "    return 0\n"
        "\n"
        "class Foo:\n"
        "    def bar(self):\n"
        "        return 42\n"
    )
    ingestor = _make_real_ingestor(shard_size=200, shard_overlap=0, use_llm_chunking=True)

    with patch.object(Ingestor, "_extract_facts_llm", autospec=True, return_value=None) as llm_spy:
        shards = await ingestor._reshard(blob, IngestCallOptions())

    assert shards, "Expected at least one shard"
    assert not llm_spy.called, "_extract_facts_llm must NOT be invoked for detected code"


# ── LLM classification & tagging (issue #37) ─────────────────────────────────
# These tests exercise the real Ingestor._classify_and_tag (NOT the stub
# override) by subclassing Ingestor and overriding only the unrelated I/O
# methods so the LLM-classification path is the one under test.


class _LlmClassifyIngestor(Ingestor):
    """Real-classify Ingestor: stubs _reshard / title-and-summary only."""

    async def _reshard(self, blob: str, opts: IngestCallOptions | None = None) -> list[str]:
        return [seg.strip() for seg in blob.split("\n\n") if seg.strip()]

    async def _generate_title_and_summary(self, chunk: str) -> tuple[str, str]:
        title = chunk[:50].strip()
        return (title, f"Summary: {chunk[:80].strip()}")


def _make_llm_classify_ingestor(
    config: LibrarianConfig | None = None,
    *,
    confidence: float = 0.8,
) -> tuple[_LlmClassifyIngestor, StubTomeRepository]:
    cfg = config or make_test_config()
    repo = StubTomeRepository()
    embedding = StubEmbeddingService(dimensions=cfg.embedding.dimensions)
    verifier = StubVerifier(confidence=confidence)
    ingestor = _LlmClassifyIngestor(cfg, embedding, verifier, repo)
    return ingestor, repo


def _ollama_response(content: str) -> Any:
    """Build a fake httpx.Response-like object holding an Ollama chat payload."""

    class _FakeResp:
        status_code = 200

        def __init__(self, content_str: str) -> None:
            self._content_str = content_str

        def raise_for_status(self) -> None:  # noqa: D401
            return None

        def json(self) -> dict[str, Any]:
            return {"choices": [{"message": {"content": self._content_str}}]}

    return _FakeResp(content)


def _patch_httpx_post(monkeypatch: pytest.MonkeyPatch, handler: Any) -> dict[str, int]:
    """Replace httpx.AsyncClient.post with `handler` and return a call-count dict."""
    counter = {"calls": 0}

    async def _fake_post(self: Any, url: str, **kwargs: Any) -> Any:  # noqa: ANN401
        counter["calls"] += 1
        if callable(handler):
            return handler(url, **kwargs)
        return handler

    import httpx as _httpx

    monkeypatch.setattr(_httpx.AsyncClient, "post", _fake_post)
    return counter


async def test_classify_uses_llm_when_no_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    """LLM result drives category and tags when no hints are supplied."""
    config = make_test_config()
    ingestor, repo = _make_llm_classify_ingestor(config)

    payload = '{"category": "Science", "tags": ["photosynthesis", "biology"]}'
    _patch_httpx_post(monkeypatch, _ollama_response(payload))

    output = await ingestor.ingest("Photosynthesis converts sunlight into glucose.")

    assert output.status == IngestStatus.STORED
    tome = output.tomes[0]
    assert tome.category == "Science"
    assert "photosynthesis" in tome.tags
    assert "biology" in tome.tags


async def test_classify_hint_overrides_llm_category(monkeypatch: pytest.MonkeyPatch) -> None:
    """A category_hint always wins over whatever the LLM returns."""
    config = make_test_config()
    ingestor, _ = _make_llm_classify_ingestor(config)

    payload = '{"category": "Science", "tags": ["a", "b"]}'
    _patch_httpx_post(monkeypatch, _ollama_response(payload))

    output = await ingestor.ingest(
        "Some text.",
        IngestCallOptions(category_hint="History"),
    )

    assert output.tomes[0].category == "History"


async def test_classify_tags_hint_merged_with_llm_tags_deduped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tags hint is merged with LLM tags; duplicates removed; order preserved."""
    config = make_test_config()
    ingestor, _ = _make_llm_classify_ingestor(config)

    payload = '{"category": "Science", "tags": ["photosynthesis", "biology"]}'
    _patch_httpx_post(monkeypatch, _ollama_response(payload))

    output = await ingestor.ingest(
        "Some text about plants.",
        IngestCallOptions(tags_hint=["x", "photosynthesis"]),
    )

    tags = output.tomes[0].tags
    # Hints come first, LLM tags added afterwards, dedupe preserves first-seen order.
    assert tags[0] == "x"
    assert tags[1] == "photosynthesis"
    assert "biology" in tags
    assert tags.count("photosynthesis") == 1


async def test_classify_out_of_taxonomy_falls_back_to_uncategorized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If LLM picks a category outside the taxonomy, default category is used; tags kept."""
    config = make_test_config()
    ingestor, _ = _make_llm_classify_ingestor(config)

    payload = '{"category": "Sportsball", "tags": ["football", "rules"]}'
    _patch_httpx_post(monkeypatch, _ollama_response(payload))

    output = await ingestor.ingest("Some text.")

    tome = output.tomes[0]
    assert tome.category == config.ingest.default_category
    assert "football" in tome.tags
    assert "rules" in tome.tags


async def test_classify_llm_failure_falls_back_to_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A network error during classification must not crash ingest."""
    import httpx as _httpx

    config = make_test_config()
    ingestor, _ = _make_llm_classify_ingestor(config)

    def _raise(url: str, **kwargs: Any) -> Any:
        raise _httpx.HTTPError("boom")

    _patch_httpx_post(monkeypatch, _raise)

    output = await ingestor.ingest("Resilient text.")

    assert output.status == IngestStatus.STORED
    tome = output.tomes[0]
    assert tome.category == config.ingest.default_category
    assert tome.tags == list(config.ingest.default_tags)


async def test_use_llm_classification_flag_off_skips_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the flag is disabled, no HTTP call is made and defaults are used."""
    from src.config import IngestSettings

    config = make_test_config(
        ingest=IngestSettings(use_llm_classification=False, min_shard_chars=0, min_shard_words=0)
    )
    ingestor, _ = _make_llm_classify_ingestor(config)

    counter = _patch_httpx_post(
        monkeypatch,
        _ollama_response('{"category":"Science","tags":["x"]}'),
    )

    output = await ingestor.ingest("Some text.")

    # The classifier never reached out to the model.
    assert counter["calls"] == 0
    tome = output.tomes[0]
    assert tome.category == config.ingest.default_category
    assert tome.tags == list(config.ingest.default_tags)


async def test_classify_extracts_json_from_filler_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LLM responses with conversational filler around the JSON are still parsed."""
    config = make_test_config()
    ingestor, _ = _make_llm_classify_ingestor(config)

    # LLM wraps JSON in filler + markdown fences mid-stream.
    payload = (
        "Sure! Here is the classification you asked for:\n"
        '```json\n{"category": "Science", "tags": ["physics"]}\n```\n'
        "Hope that helps!"
    )
    _patch_httpx_post(monkeypatch, _ollama_response(payload))

    output = await ingestor.ingest("Some text about quanta.")

    tome = output.tomes[0]
    assert tome.category == "Science"
    assert "physics" in tome.tags


async def test_classify_taxonomy_match_is_case_insensitive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LLM-returned category matches taxonomy regardless of casing; canonical form kept."""
    config = make_test_config()
    ingestor, _ = _make_llm_classify_ingestor(config)

    # Lowercase category from the model; "Science" in taxonomy.
    payload = '{"category": "science", "tags": ["chem"]}'
    _patch_httpx_post(monkeypatch, _ollama_response(payload))

    output = await ingestor.ingest("Some text.")

    # Stored category preserves the canonical taxonomy casing.
    assert output.tomes[0].category == "Science"


# ── prompt / fence-stripper escape regression (issue #20) ────────────────────


class TestPromptEscapes:
    """Regression tests for issue #20: literal ``\\n``/``\\s`` escapes in ingestor."""

    async def test_prompt_uses_real_newlines(
        self,
        config: LibrarianConfig,
        repo: StubTomeRepository,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The LLM prompt must contain real newlines, not literal ``\\n`` escapes."""
        sample = "Photosynthesis converts sunlight into glucose. " * 4
        captured: dict[str, Any] = {}

        class _FakeResponse:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict[str, Any]:
                return {"choices": [{"message": {"content": '{"facts": ["a fact"]}'}}]}

        async def _fake_post(self: httpx.AsyncClient, url: str, **kwargs: Any) -> _FakeResponse:
            captured["url"] = url
            captured["payload"] = kwargs.get("json")
            return _FakeResponse()

        monkeypatch.setattr(httpx.AsyncClient, "post", _fake_post)

        ingestor = Ingestor(
            config,
            StubEmbeddingService(dimensions=config.embedding.dimensions),
            StubVerifier(confidence=0.9),
            repo,
        )

        result = await ingestor._extract_facts_llm(sample)

        assert result == ["a fact"]
        content = captured["payload"]["messages"][0]["content"]
        # No literal backslash-n sequences in the prompt the LLM sees.
        assert "\\n" not in content
        # Real newlines are present (header + blank line + TEXT: + body).
        assert content.count("\n") >= 3
        assert content.endswith("TEXT:\n" + sample[:8000])

    @pytest.mark.parametrize(
        "fenced",
        [
            '```json\n{"facts": ["one", "two"]}\n```',
            '```\n{"facts": ["one", "two"]}\n```',
            '```json {"facts": ["one", "two"]} ```',
        ],
    )
    async def test_fence_stripper_strips_whitespace(
        self,
        config: LibrarianConfig,
        repo: StubTomeRepository,
        monkeypatch: pytest.MonkeyPatch,
        fenced: str,
    ) -> None:
        """``_extract_facts_llm`` must strip ``\\n``/space whitespace inside JSON code fences."""

        class _FakeResponse:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict[str, Any]:
                return {"choices": [{"message": {"content": fenced}}]}

        async def _fake_post(self: httpx.AsyncClient, url: str, **kwargs: Any) -> _FakeResponse:
            return _FakeResponse()

        monkeypatch.setattr(httpx.AsyncClient, "post", _fake_post)

        ingestor = Ingestor(
            config,
            StubEmbeddingService(dimensions=config.embedding.dimensions),
            StubVerifier(confidence=0.9),
            repo,
        )

        # With the correct ``\s`` regex, the fence is stripped cleanly and the
        # JSON parses to ["one", "two"]. With the buggy ``\\s`` literal-backslash
        # regex, the fence is left in place, ``json.loads`` raises, and
        # ``_extract_facts_llm`` silently returns None.
        result = await ingestor._extract_facts_llm("any text")
        assert result == ["one", "two"]


# ── issue #32: LLM fact-extraction semantics ─────────────────────────────────


class TestLlmFactExtractionRename:
    """Regression tests for issue #32: rename + configurable cap + safe defaults."""

    def test_extract_facts_llm_method_exists(self) -> None:
        """``_reshard_llm`` is renamed to ``_extract_facts_llm`` to reflect that
        the LLM rewrites content into atomic facts (it does not split the source).
        """
        assert hasattr(Ingestor, "_extract_facts_llm"), (
            "Ingestor must expose _extract_facts_llm (renamed from _reshard_llm)"
        )
        assert not hasattr(Ingestor, "_reshard_llm"), (
            "Old name _reshard_llm must be removed after the rename"
        )

    def test_use_llm_chunking_defaults_to_false(self) -> None:
        """LLM fact extraction rewrites content semantically, so it must be opt-in."""
        assert IngestSettings().use_llm_chunking is False
        assert make_test_config().ingest.use_llm_chunking is False

    def test_llm_extraction_max_chars_is_configurable(self) -> None:
        """The previously-hardcoded 8000-char truncation must be a configurable field."""
        settings = IngestSettings()
        assert hasattr(settings, "llm_extraction_max_chars"), (
            "IngestSettings must expose llm_extraction_max_chars"
        )
        assert settings.llm_extraction_max_chars == 8000

    async def test_llm_extraction_uses_configured_max_chars(
        self,
        repo: StubTomeRepository,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The configured ``llm_extraction_max_chars`` value must drive truncation,
        not the legacy hard-coded 8000 constant.
        """
        captured: dict[str, Any] = {}

        class _FakeResponse:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict[str, Any]:
                return {"choices": [{"message": {"content": '{"facts": ["a fact"]}'}}]}

        async def _fake_post(self: httpx.AsyncClient, url: str, **kwargs: Any) -> _FakeResponse:
            captured["payload"] = kwargs.get("json")
            return _FakeResponse()

        monkeypatch.setattr(httpx.AsyncClient, "post", _fake_post)

        ingest_settings = IngestSettings(llm_extraction_max_chars=42)
        config = make_test_config(ingest=ingest_settings)
        ingestor = Ingestor(
            config,
            StubEmbeddingService(dimensions=config.embedding.dimensions),
            StubVerifier(confidence=0.9),
            repo,
        )

        blob = "X" * 200
        await ingestor._extract_facts_llm(blob)

        content = captured["payload"]["messages"][0]["content"]
        # The prompt body must contain only the first 42 chars of the source.
        assert content.endswith("TEXT:\n" + blob[:42])
        # The full 200-char blob must NOT have been forwarded.
        assert blob not in content

    def test_enabling_llm_chunking_emits_startup_warning(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Constructing an Ingestor with ``use_llm_chunking=True`` must log a
        WARNING that the feature rewrites content semantics.
        """
        import logging as _logging

        config = make_test_config(ingest=IngestSettings(use_llm_chunking=True))
        with caplog.at_level(_logging.WARNING, logger="src.services.ingestor"):
            Ingestor(
                config,
                StubEmbeddingService(dimensions=config.embedding.dimensions),
                StubVerifier(confidence=0.9),
                StubTomeRepository(),
            )

        warning_records = [r for r in caplog.records if r.levelno >= _logging.WARNING]
        assert any("use_llm_chunking" in r.getMessage() for r in warning_records), (
            "Expected a startup WARNING mentioning use_llm_chunking"
        )


# ── thread grouping / conversational memory (issue #49) ──────────────────────────


async def test_ingest_with_thread_grouping(
    config: LibrarianConfig, repo: StubTomeRepository
) -> None:
    """Ingest 5 tomes with thread_id set; verify sequential thread_position.

    Users group related facts into conversations/threads via thread_id.
    Each tome in a thread gets a sequential position (0, 1, 2, 3, 4).
    library_list(thread_id=<uuid>) returns all tomes in order.
    """
    ingestor = Ingestor(
        config,
        StubEmbeddingService(dimensions=config.embedding.dimensions),
        StubVerifier(confidence=0.9),
        repo,
    )

    thread_id = str(uuid.uuid4())

    # Ingest 5 separate chunks, each with the same thread_id
    contents = [
        "First turn: what is photosynthesis?",
        "Second turn: plants need sunlight.",
        "Third turn: chlorophyll absorbs light.",
        "Fourth turn: glucose is produced.",
        "Fifth turn: oxygen is released.",
    ]

    for content in contents:
        opts = IngestCallOptions(
            skip_verify=False,
            source_type=SourceType.AGENT_INPUT,
        )
        # Simulate thread_id being passed through opts
        # (Once the model is updated, thread_id will be a field on opts)
        opts.thread_id = uuid.UUID(hex=thread_id)
        await ingestor.ingest(content, opts)

    # Verify 5 tomes were created
    all_tomes = repo.all_tomes()
    assert len(all_tomes) == 5, f"Expected 5 tomes, got {len(all_tomes)}"

    # Verify each has the thread_id set
    for tome in all_tomes:
        assert tome.thread_id == uuid.UUID(hex=thread_id), (
            f"Expected thread_id={thread_id}, got {tome.thread_id}"
        )

    # Verify sequential thread_position (0, 1, 2, 3, 4)
    positions = sorted([t.thread_position for t in all_tomes])
    assert positions == [0, 1, 2, 3, 4], f"Expected thread_position [0,1,2,3,4], got {positions}"

    # Call library_list with thread_id filter, expect all 5 in order
    filtered = await repo.list_all(thread_id=uuid.UUID(hex=thread_id))
    assert len(filtered) == 5, f"Expected 5 filtered tomes, got {len(filtered)}"

    # Verify order: should be sorted by thread_position ascending
    filtered_positions = [t.thread_position for t in filtered]
    assert filtered_positions == [0, 1, 2, 3, 4], (
        f"Expected filtered tomes in order [0,1,2,3,4], got {filtered_positions}"
    )
