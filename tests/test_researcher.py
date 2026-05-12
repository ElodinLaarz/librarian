"""Tests for the Researcher service and plan_search_queries helper."""

from __future__ import annotations

import uuid

import pytest

from src.models.enums import ResearchDepth, ResearchJobStatus
from src.models.research_job import ResearchJob
from src.services.ingestor import IngestCallOptions
from src.services.researcher import Researcher, _default_queries, plan_search_queries
from src.services.web_search import WebSearchResult
from tests.conftest import make_test_config
from tests.stubs import (
    StubIngestor,
    StubResearchJobRepository,
    StubWebSearchClient,
    make_stub_ingestor,
)


def _make_job(
    topic: str = "test topic", depth: ResearchDepth = ResearchDepth.STANDARD
) -> ResearchJob:
    return ResearchJob(id=uuid.uuid4(), topic=topic, depth=depth)


def _make_researcher(
    web_results: list[WebSearchResult] | None = None,
) -> tuple[Researcher, StubResearchJobRepository]:
    config = make_test_config()
    ingestor, _, _ = make_stub_ingestor(config=config, confidence=0.8)
    jobs = StubResearchJobRepository()
    web = StubWebSearchClient(results=web_results)
    researcher = Researcher(config, web, ingestor, jobs)
    return researcher, jobs


# ── _default_queries ─────────────────────────────────────────────────────────


def test_default_queries_includes_topic() -> None:
    queries = _default_queries("machine learning", None)
    assert any("machine learning" in q for q in queries)


def test_default_queries_includes_context_when_provided() -> None:
    queries = _default_queries("Python", "async programming")
    assert any("async" in q or "Python" in q for q in queries)


def test_default_queries_returns_multiple() -> None:
    queries = _default_queries("topic", None)
    assert len(queries) >= 3


# ── plan_search_queries ───────────────────────────────────────────────────────


async def test_plan_search_queries_falls_back_when_ollama_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import httpx

    async def _raise(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise httpx.ConnectError("refused")

    monkeypatch.setattr("httpx.AsyncClient.post", _raise)
    config = make_test_config()
    queries = await plan_search_queries("black holes", None, config, max_queries=3)
    assert len(queries) == 3
    assert all(isinstance(q, str) for q in queries)


async def test_plan_search_queries_uses_llm_when_valid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json
    from unittest.mock import AsyncMock, MagicMock

    llm_queries = ["black hole formation", "black hole types", "black hole event horizon"]
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json = MagicMock(
        return_value={"choices": [{"message": {"content": json.dumps({"queries": llm_queries})}}]}
    )

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_response)

    monkeypatch.setattr("httpx.AsyncClient", lambda **kwargs: mock_client)

    config = make_test_config()
    queries = await plan_search_queries("black holes", None, config, max_queries=3)
    assert queries == llm_queries


# ── Researcher.run_job ───────────────────────────────────────────────────────


async def test_run_job_fails_when_web_unavailable() -> None:
    config = make_test_config()
    ingestor, _, _ = make_stub_ingestor(config=config)
    jobs = StubResearchJobRepository()
    # No results=None → is_available() returns False
    from tests.stubs import StubWebSearchClient

    web = StubWebSearchClient(results=None)
    researcher = Researcher(config, web, ingestor, jobs)

    job = _make_job()
    await jobs.insert(job)
    await researcher.run_job(job.id)

    updated = await jobs.get_by_id(job.id)
    assert updated is not None
    assert updated.status == ResearchJobStatus.FAILED
    assert updated.error is not None


async def test_run_job_completes_and_stores_tomes() -> None:
    results = [
        WebSearchResult(
            title="Python overview",
            url="https://example.com/python",
            snippet="Python is a high-level programming language.",
        )
    ]
    researcher, jobs = _make_researcher(web_results=results)

    job = _make_job(topic="Python programming language")
    await jobs.insert(job)

    # Monkeypatch fetch_page_content to avoid real HTTP
    async def _fake_fetch(url: str) -> str:
        return (
            "Python is a high-level, interpreted programming language. "
            "It was created by Guido van Rossum. "
            "Python emphasizes code readability and simplicity."
        )

    researcher._web.fetch_page_content = _fake_fetch  # type: ignore[method-assign]

    await researcher.run_job(job.id)

    updated = await jobs.get_by_id(job.id)
    assert updated is not None
    assert updated.status == ResearchJobStatus.COMPLETED
    assert len(updated.tome_ids) > 0
    assert len(updated.sources) > 0


async def test_run_job_fails_when_no_fetchable_content() -> None:
    results = [WebSearchResult(title="Empty page", url="https://example.com/empty", snippet="")]
    researcher, jobs = _make_researcher(web_results=results)

    # fetch_page_content returns empty string
    async def _empty_fetch(url: str) -> str:
        return ""

    researcher._web.fetch_page_content = _empty_fetch  # type: ignore[method-assign]

    job = _make_job()
    await jobs.insert(job)
    await researcher.run_job(job.id)

    updated = await jobs.get_by_id(job.id)
    assert updated is not None
    assert updated.status == ResearchJobStatus.FAILED


async def test_run_job_handles_unknown_job_id_gracefully() -> None:
    researcher, _ = _make_researcher()
    # Should not raise — logs the error and returns
    await researcher.run_job(uuid.uuid4())


async def test_run_job_derives_content_based_tags_per_tome() -> None:
    """Each ingested tome should have tags derived from its own content,
    not just the single ``[topic[:60]]`` slice (issue #42).

    The synthesised research body is split into multiple chunks by the
    ingestor's reshard pass; the resulting tomes must each carry tags
    that reflect their individual content. The job topic may also be
    present, but must not be the *only* tag.
    """
    results = [
        WebSearchResult(
            title="Async Python",
            url="https://example.com/async",
            snippet="async io concurrency",
        )
    ]
    config = make_test_config()

    # Custom stub: derive tags from the chunk content keywords. This
    # mirrors what the real ingestor would do via its LLM classifier,
    # but stays deterministic and offline for tests.
    class ContentTaggingIngestor(StubIngestor):
        async def _classify_and_tag(
            self,
            chunk: str,
            category_hint: str | None = None,
            tags_hint: list[str] | None = None,
        ) -> tuple[str, list[str]]:
            keywords = ["python", "async", "concurrency", "asyncio", "event-loop"]
            derived = [k for k in keywords if k in chunk.lower()]
            # Merge user-supplied hints with content-derived tags, hints
            # preserved first (mirrors real ingestor merge order).
            merged = list(dict.fromkeys([*(tags_hint or []), *derived]))
            return category_hint or "general", merged or ["stub"]

    from tests.stubs import StubEmbeddingService, StubTomeRepository, StubVerifier

    repo = StubTomeRepository()
    verifier = StubVerifier(confidence=0.8)
    embedding = StubEmbeddingService(dimensions=config.embedding.dimensions)
    ingestor = ContentTaggingIngestor(config, embedding, verifier, repo)
    jobs = StubResearchJobRepository()
    web = StubWebSearchClient(results=results)
    researcher = Researcher(config, web, ingestor, jobs)

    # Three distinct fetched chunks, each with its own dominant keyword.
    fetched_bodies = [
        "Python is a programming language with rich syntax.",
        "Asyncio enables async concurrent IO in modern code.",
        "An event-loop schedules coroutines in concurrency runtimes.",
    ]
    web._results = [
        WebSearchResult(title=f"page{i}", url=f"https://example.com/{i}", snippet="")
        for i in range(len(fetched_bodies))
    ]

    body_iter = iter(fetched_bodies)

    async def _fetch(url: str) -> str:
        return next(body_iter, "")

    researcher._web.fetch_page_content = _fetch  # type: ignore[method-assign]

    job = _make_job(topic="Python concurrency guide")
    await jobs.insert(job)
    await researcher.run_job(job.id)

    updated = await jobs.get_by_id(job.id)
    assert updated is not None
    assert updated.status == ResearchJobStatus.COMPLETED

    stored = repo.all_tomes()
    assert len(stored) >= 1, "Researcher must produce at least one tome"

    # Aggregate the set of distinct tags across all tomes. At minimum we
    # expect to see content-derived tokens from the fetched bodies, NOT
    # just the topic slice.
    all_tags: set[str] = set()
    for tome in stored:
        all_tags.update(tome.tags)

    # The single-topic-slice bug would produce exactly ``{topic[:60]}``.
    assert all_tags != {"Python concurrency guide"[:60]}, (
        f"Researcher only emitted the topic slice as a tag: {all_tags}"
    )

    content_derived = {"python", "async", "concurrency", "asyncio", "event-loop"}
    assert all_tags & content_derived, (
        f"Expected at least one content-derived tag from {content_derived}, got {all_tags}"
    )

    # Each individual tome must have more than just the topic tag — i.e.
    # at least one content-derived tag of its own.
    for tome in stored:
        non_topic = [t for t in tome.tags if t != "Python concurrency guide"[:60]]
        assert non_topic, f"Tome {tome.id} has only topic tag: {tome.tags}"


async def test_run_job_does_not_block_ingestor_classification() -> None:
    """Researcher must NOT pre-supply tags_hint as the single topic
    slice — doing so causes the ingestor's classifier short-circuit
    (issue #42).

    We assert here by recording the ``IngestCallOptions`` the
    researcher passes to the ingestor and checking that the
    ``tags_hint`` does not pin the tags to just the topic.
    """
    results = [
        WebSearchResult(title="t", url="https://example.com/x", snippet="s"),
    ]
    researcher, jobs = _make_researcher(web_results=results)

    seen_opts: list[IngestCallOptions] = []
    real_ingest = researcher._ingestor.ingest

    async def _record(text: str, opts: IngestCallOptions):  # type: ignore[no-untyped-def]
        seen_opts.append(opts)
        return await real_ingest(text, opts)

    researcher._ingestor.ingest = _record  # type: ignore[method-assign]

    async def _fetch(url: str) -> str:
        return "Some readable body with multiple sentences for ingestion."

    researcher._web.fetch_page_content = _fetch  # type: ignore[method-assign]

    job = _make_job(topic="my topic")
    await jobs.insert(job)
    await researcher.run_job(job.id)

    assert seen_opts, "Researcher did not invoke the ingestor"
    opts = seen_opts[0]
    # tags_hint must NOT be exactly [topic] — that pin causes the
    # ingestor LLM classifier to be short-circuited / hint-dominated.
    assert opts.tags_hint != ["my topic"], (
        "Researcher still passes a single-topic tags_hint, blocking "
        "the ingestor's content-based classification."
    )


async def test_run_job_respects_depth_budget() -> None:
    """SHALLOW depth should issue fewer queries than DEEP."""
    config = make_test_config()
    shallow_q, _ = Researcher(
        config,
        StubWebSearchClient(results=[]),
        make_stub_ingestor(config=config)[0],
        StubResearchJobRepository(),
    )._budget(ResearchDepth.SHALLOW)
    deep_q, _ = Researcher(
        config,
        StubWebSearchClient(results=[]),
        make_stub_ingestor(config=config)[0],
        StubResearchJobRepository(),
    )._budget(ResearchDepth.DEEP)
    assert shallow_q < deep_q
