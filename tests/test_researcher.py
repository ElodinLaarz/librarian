"""Tests for the Researcher service and plan_search_queries helper."""

from __future__ import annotations

import uuid

import pytest

from src.models.enums import ResearchDepth, ResearchJobStatus
from src.models.research_job import ResearchJob
from src.services.researcher import Researcher, _default_queries, plan_search_queries
from src.services.web_search import WebSearchResult
from tests.conftest import make_test_config
from tests.stubs import (
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
