"""End-to-end pipeline test: ingest -> search -> research polling.

Drives the real MCP tool handlers registered on ``LibrarianServer`` rather than
exercising service-layer helpers directly.  Storage is filesystem-backed
(``FsTomeRepository`` / ``FsResearchJobRepository``) so the test runs without
Mongo, Ollama, or any external service.  The LLM-bound features (verifier,
classification, summary, fact extraction, query planning, web search) are
disabled via config or stubbed at module scope.

Covers issue #34: end-to-end coverage of the full pipeline.
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.config import (
    DatabaseSettings,
    EmbeddingSettings,
    IngestSettings,
    LibrarianConfig,
    TidySettings,
    VerificationSettings,
)
from src.models.tool_schemas import IngestInput, ResearchInput, SearchInput
from src.server import LibrarianServer
from src.services.ingestor import Ingestor
from src.services.researcher import Researcher
from src.services.tidier import Tidier
from src.services.verifier import Verifier
from src.services.web_search import WebSearchResult
from src.storage.filesystem.fs_research_job_repository import FsResearchJobRepository
from src.storage.filesystem.fs_tome_repository import FsTomeRepository
from tests.stubs import (
    StubEmbeddingService,
    StubVerifier,
    StubWebSearchClient,
)

# ── helpers ─────────────────────────────────────────────────────────────────


def _make_e2e_config(tmp_path: Path) -> LibrarianConfig:
    """Build a fully-isolated config: filesystem storage, no LLM, no tidy loop."""
    return LibrarianConfig(
        database=DatabaseSettings(
            uri=str(tmp_path),
            tomes_collection="tomes",
            jobs_collection="research_jobs",
            tls=False,
        ),
        # 8-dim vectors keep cosine math cheap and predictable.
        embedding=EmbeddingSettings(dimensions=8, provider="dummy"),
        # All LLM-bound ingest features OFF so the pipeline is fully offline.
        ingest=IngestSettings(
            use_llm_chunking=False,
            use_llm_summary=False,
            use_llm_classification=False,
        ),
        # Verifier still attaches a fixed confidence via the stub, but disable
        # the production verifier path so no Ollama call is attempted.
        verification=VerificationSettings(enabled=False),
        # Disable the background tidy loop — we don't want stray tasks in test.
        tidy=TidySettings(enabled=False),
    )


def _build_server(tmp_path: Path) -> LibrarianServer:
    """Construct a LibrarianServer and manually wire its dependencies.

    We bypass FastMCP's lifespan context manager because it would call
    ``build_embedding_service`` and ``build_web_search_client``, which can
    reach for Ollama / external HTTP.  Wiring matches ``LibrarianServer.lifespan``
    in ``src/server.py:133-176``.
    """
    config = _make_e2e_config(tmp_path)
    server = LibrarianServer(config)

    # Deterministic, zero-vector embeddings (8-dim).
    embedding_service = StubEmbeddingService(dimensions=config.embedding.dimensions)
    server._embedding_service = embedding_service

    # Filesystem-backed repositories — the real production classes.
    server.tome_repo = FsTomeRepository(config.database, embedding_service, config.tidy)
    server.job_repo = FsResearchJobRepository(config.database)

    # Stub the web client with deterministic canned results.
    web = StubWebSearchClient(
        results=[
            WebSearchResult(
                title="Stub Source: Photosynthesis",
                url="https://example.com/photosynthesis",
                snippet="Photosynthesis converts light into chemical energy.",
            ),
            WebSearchResult(
                title="Stub Source: Calvin Cycle",
                url="https://example.com/calvin",
                snippet="The Calvin cycle fixes carbon dioxide into sugars.",
            ),
        ]
    )

    async def _fake_fetch(url: str) -> str:
        # Deterministic body — wide enough to survive resharding into >= 1 tome.
        return (
            f"Article from {url}. "
            "Photosynthesis converts light energy into chemical energy stored in glucose. "
            "Plants use chlorophyll to absorb sunlight and convert carbon dioxide and "
            "water into oxygen and sugar. The Calvin cycle is the second stage where "
            "carbon fixation occurs in the stroma of the chloroplast."
        )

    web.fetch_page_content = _fake_fetch  # type: ignore[method-assign]

    # Stub verifier so we never touch the verification HTTP stack even if a
    # caller flips ``verification.enabled`` back on. The real Verifier is the
    # type the production wiring uses (see src/server.py:155-156).
    verifier: Verifier = StubVerifier(confidence=0.9)

    server.ingestor = Ingestor(config, embedding_service, verifier, server.tome_repo)
    server.researcher = Researcher(config, web, server.ingestor, server.job_repo)
    server.tidier = Tidier(server.ingestor, server.tome_repo, config.tidy)

    return server


def _tool(server: LibrarianServer, name: str) -> Any:
    """Look up a registered MCP tool handler by name."""
    return next(t.fn for t in server.mcp._tool_manager.list_tools() if t.name == name)


async def _teardown(server: LibrarianServer) -> None:
    """Cancel any background tasks and release the ingestor HTTP client."""
    for task in list(server._bg_tasks):
        task.cancel()
    server._bg_tasks.clear()
    if server.ingestor is not None:
        await server.ingestor.aclose()


# ── fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def server(tmp_path: Path) -> LibrarianServer:
    return _build_server(tmp_path)


# ── main E2E walk ───────────────────────────────────────────────────────────


async def test_e2e_ingest_search_research_pipeline(server: LibrarianServer) -> None:
    """Walks the full pipeline through the registered MCP tool handlers.

    Step 1 — library_ingest: a small blob is stored as at least one Tome.
    Step 2 — library_search: the ingested content is retrievable by query.
    Step 3 — library_research: an async job runs to completion and the
             resulting tomes are persisted and discoverable by search.
    """
    library_ingest = _tool(server, "library_ingest")
    library_search = _tool(server, "library_search")
    library_research = _tool(server, "library_research")

    try:
        # ── Step 1: ingest ────────────────────────────────────────────────
        ingest_input = IngestInput(
            content=(
                "The mitochondrion is the powerhouse of the cell. "
                "It produces ATP via oxidative phosphorylation."
            ),
            skip_verify=True,
            category="science",
            tags=["biology", "cell"],
        )
        ingest_out = await library_ingest(ingest_input)
        assert ingest_out.tomes, f"ingest produced no tomes: {ingest_out!r}"
        ingested_ids = {t.id for t in ingest_out.tomes}

        # ── Step 2: search ───────────────────────────────────────────────
        search_out = await library_search(SearchInput(query="cell biology", min_confidence=0.0))
        # The ingested tome should appear in the search result set.
        returned_ids = {t.id for t in search_out.tomes}
        assert ingested_ids & returned_ids, (
            f"none of the ingested ids {ingested_ids} appear in search results {returned_ids}"
        )
        assert len(search_out.scores) == len(search_out.tomes)

        # ── Step 3: research (async) + polling cycle ─────────────────────
        kickoff = await library_research(ResearchInput(topic="Photosynthesis basics", async_=True))
        assert kickoff.job_id, "research kickoff returned no job_id"
        # First status is pending or running depending on scheduling — either
        # is acceptable, both are non-terminal.
        assert kickoff.status in {"pending", "running"}, (
            f"unexpected initial status: {kickoff.status!r}"
        )

        # Poll until terminal. ~5s wall-clock budget should be plenty.
        terminal_states = {"completed", "failed", "not_found", "invalid_job_id"}
        deadline = asyncio.get_running_loop().time() + 5.0
        polls = 0
        while True:
            status = await library_research(ResearchInput(job_id=kickoff.job_id))
            polls += 1
            if status.status in terminal_states:
                break
            if asyncio.get_running_loop().time() > deadline:
                pytest.fail(
                    f"research job did not finish within budget; "
                    f"last status={status.status!r} after {polls} polls"
                )
            await asyncio.sleep(0.05)

        # The job must have completed — failure here likely indicates a real
        # regression in the pipeline (web stub returned, ingest disabled LLM,
        # FS storage is local).
        assert status.status == "completed", (
            f"research job ended in {status.status!r}; error={status.error!r}"
        )
        assert status.tome_ids, "completed job stored no tomes"
        assert status.tomes, "completed job did not return tome payloads"
        assert status.sources, "completed job recorded no sources"

        # ── Step 3b: research tomes are searchable ───────────────────────
        research_search = await library_search(
            SearchInput(query="photosynthesis", min_confidence=0.0, top_k=20)
        )
        research_search_ids = {t.id for t in research_search.tomes}
        job_tome_ids = {uuid.UUID(t) for t in status.tome_ids}
        assert research_search_ids & job_tome_ids, (
            "research-produced tomes are not retrievable via library_search: "
            f"job_ids={job_tome_ids} search_ids={research_search_ids}"
        )
    finally:
        await _teardown(server)


# ── Auxiliary checks — exercise narrower slices of the same pipeline ────────


async def test_e2e_research_unknown_job_id_returns_not_found(
    server: LibrarianServer,
) -> None:
    """Polling an unknown UUID surfaces the documented sentinel status."""
    library_research = _tool(server, "library_research")
    try:
        result = await library_research(ResearchInput(job_id=uuid.uuid4().hex))
        assert result.status == "not_found"
    finally:
        await _teardown(server)


async def test_e2e_research_invalid_job_id_is_rejected(
    server: LibrarianServer,
) -> None:
    """Malformed job_id is surfaced as ``invalid_job_id`` rather than a crash."""
    library_research = _tool(server, "library_research")
    try:
        result = await library_research(ResearchInput(job_id="not-a-uuid"))
        assert result.status == "invalid_job_id"
    finally:
        await _teardown(server)


async def test_e2e_search_uses_real_embedding_dimensions(
    server: LibrarianServer,
) -> None:
    """Smoke check that the wired-up embedding service returns expected dims."""
    embedding = server._embedding_service
    assert embedding is not None
    vec = await embedding.embed("any text")
    # Matches the test config we constructed in ``_make_e2e_config``.
    assert vec.shape == (8,)
    assert vec.dtype == np.float32
    await _teardown(server)
