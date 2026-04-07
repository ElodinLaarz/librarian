#!/usr/bin/env python3
"""Offline demo: search (empty) → manual ingest → search → research → search again.

Uses a temporary filesystem library, dummy embeddings, and a stub web client (no API keys,
no Docker, no Ollama). Run from repo root:

    uv run python scripts/demo_librarian.py
"""

from __future__ import annotations

import asyncio
import tempfile
import uuid
from pathlib import Path

from src.config import DatabaseSettings, EmbeddingSettings, LibrarianConfig
from src.models.enums import ResearchDepth, ResearchJobStatus, SourceType
from src.models.research_job import ResearchJob
from src.services.embedding import build_embedding_service
from src.services.ingestor import IngestCallOptions, Ingestor
from src.services.researcher import Researcher
from src.services.verifier import Verifier
from src.services.web_search import WebSearchClient, WebSearchResult
from src.storage.filesystem.fs_research_job_repository import FsResearchJobRepository
from src.storage.filesystem.fs_tome_repository import FsTomeRepository


class DemoWebSearchClient(WebSearchClient):
    """Single canned search result; offline body text (no HTTP)."""

    async def search(self, query: str, max_results: int = 5) -> list[WebSearchResult]:
        _ = query
        return [
            WebSearchResult(
                title="Demo page: vector search in knowledge bases",
                url="https://example.invalid/librarian-demo",
                snippet="Offline demo snippet for library.research.",
            )
        ][:max_results]

    def is_available(self) -> bool:
        return True

    async def fetch_page_content(self, url: str) -> str:
        _ = url
        return (
            "Semantic search maps queries and documents into the same embedding space. "
            "Similar meanings yield high cosine similarity even when wording differs. "
            "The Librarian stores each ingested unit as a Tome with an embedding vector."
        )


async def run_demo() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp).resolve()
        uri = f"file://{base.as_posix()}"
        config = LibrarianConfig(
            database=DatabaseSettings(uri=uri),
            embedding=EmbeddingSettings(provider="dummy", dimensions=32),
        )

        embedding = await build_embedding_service(config.embedding)

        tome_repo = FsTomeRepository(config.database, embedding)
        job_repo = FsResearchJobRepository(config.database)
        web = DemoWebSearchClient()
        verifier = Verifier(config, web)
        ingestor = Ingestor(config, embedding, verifier, tome_repo)
        researcher = Researcher(config, web, ingestor, job_repo)

        print("=== 1) library_search (expect empty library) ===")
        empty = await tome_repo.search("anything", top_k=3)
        print(f"hits: {len(empty)}")

        print("\n=== 2) library_ingest (manual fact about Mars) ===")
        ingest_out = await ingestor.ingest(
            "Mars is the fourth planet from the Sun. "
            "It is often called the Red Planet due to iron oxide on its surface.",
            IngestCallOptions(
                skip_verify=True,
                source_type=SourceType.AGENT_INPUT,
                category_hint="astronomy",
                tags_hint=["demo", "planets"],
            ),
        )
        print(f"status: {ingest_out.status}  tomes stored: {len(ingest_out.tomes)}")
        if ingest_out.tomes:
            t0 = ingest_out.tomes[0]
            print(f"  first tome title: {t0.title!r}")

        print("\n=== 3) library_search (query should match ingested content) ===")
        found = await tome_repo.search(
            "Which planet is known as the Red Planet?",
            top_k=3,
            min_confidence=0.0,
        )
        print(f"hits: {len(found)}")
        for tome, score in found[:3]:
            print(f"  score={score:.4f}  title={tome.title!r}  summary={tome.summary[:80]}...")

        print("\n=== 4) library_research (sync, stub web → ingest) ===")
        job = ResearchJob(
            id=uuid.uuid4(),
            topic="embedding-based retrieval",
            depth=ResearchDepth.SHALLOW,
            max_tomes=3,
        )
        await job_repo.insert(job)
        await researcher.run_job(job.id)
        job_done = await job_repo.get_by_id(job.id)
        assert job_done is not None
        print(f"job status: {job_done.status.value}  error: {job_done.error!r}")
        print(f"ingested tome ids: {len(job_done.tome_ids)}  sources: {len(job_done.sources)}")

        print("\n=== 5) library_search (stored research content) ===")
        again = await tome_repo.search(
            "cosine similarity documents",
            top_k=3,
            min_confidence=0.0,
        )
        print(f"hits: {len(again)}")
        for tome, score in again[:3]:
            print(f"  score={score:.4f}  title={tome.title!r}")

        if job_done.status != ResearchJobStatus.COMPLETED:
            raise SystemExit(f"research job did not complete: {job_done.error}")

        job_repo.close()
        tome_repo.close()


def main() -> None:
    asyncio.run(run_demo())
    print("\nDemo finished OK.")


if __name__ == "__main__":
    main()
