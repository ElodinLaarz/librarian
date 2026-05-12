from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import UTC, datetime

import httpx

from src.config import LibrarianConfig
from src.models.enums import ResearchDepth, ResearchJobStatus, SourceType
from src.services.ingestor import IngestCallOptions, Ingestor
from src.services.web_search import WebSearchClient
from src.storage.errors import StorageError
from src.storage.research_job_repository import ResearchJobRepository


class Researcher:
    """Web search, fetch, synthesise, and ingest pipeline for `library_research`."""

    def __init__(
        self,
        config: LibrarianConfig,
        web: WebSearchClient,
        ingestor: Ingestor,
        jobs: ResearchJobRepository,
    ) -> None:
        self._config = config
        self._web = web
        self._ingestor = ingestor
        self._jobs = jobs

    def _budget(self, depth: ResearchDepth) -> tuple[int, int]:
        r = self._config.researcher
        if depth == ResearchDepth.SHALLOW:
            return r.shallow_queries, r.shallow_urls_per_query
        if depth == ResearchDepth.DEEP:
            return r.deep_queries, r.deep_urls_per_query
        return r.standard_queries, r.standard_urls_per_query

    async def run_job(self, job_id: uuid.UUID) -> None:
        """Execute a persisted job (used for background and synchronous runs)."""
        job = await self._jobs.get_by_id(job_id)
        if job is None:
            logging.error("Research job %s not found", job_id)
            return

        job.status = ResearchJobStatus.RUNNING
        job.started_at = datetime.now(UTC)
        await self._jobs.update(job)

        if not self._web.is_available():
            job.status = ResearchJobStatus.FAILED
            job.error = (
                "Web search is not configured (set LIBRARIAN_WEB_SEARCH_API_KEY, or "
                "BRAVE_API_KEY / SERPER_API_KEY / TAVILY_API_KEY for the chosen provider)."
            )
            job.finished_at = datetime.now(UTC)
            await self._jobs.update(job)
            return

        try:
            n_queries, per_query = self._budget(job.depth)
            queries = await plan_search_queries(
                job.topic, job.context, self._config, max_queries=n_queries
            )
            job.queries = queries
            job.query_count = len(queries)
            await self._jobs.update(job)

            seen_urls: set[str] = set()
            sources: list[str] = []
            chunks: list[str] = []

            for q in queries:
                try:
                    hits = await self._web.search(q, max_results=per_query)
                except Exception as exc:
                    logging.warning("Search failed for %r: %s", q, exc)
                    continue
                for h in hits:
                    if h.url in seen_urls or not h.url:
                        continue
                    seen_urls.add(h.url)
                    body = await self._web.fetch_page_content(h.url)
                    if not body.strip():
                        continue
                    sources.append(h.url)
                    chunks.append(f"## {h.title}\nURL: {h.url}\n\n{body[:5000]}")

            combined = "\n\n---\n\n".join(chunks)
            cap = min(
                self._config.researcher.max_synthesis_chars,
                max(1, job.max_tomes) * 900,
            )
            combined = combined[:cap]

            if not combined.strip():
                job.status = ResearchJobStatus.FAILED
                job.error = "No readable content could be fetched from the web."
                job.sources = sources
                job.finished_at = datetime.now(UTC)
                await self._jobs.update(job)
                return

            # Preserve the topic as an additional tag (so users can still
            # filter tomes by research job topic) but let the ingestor's
            # classifier derive content-based tags per chunk. Issue #42 —
            # previously this was passed as ``tags_hint``, which pinned the
            # final tag list to ``[topic]`` and blocked classifier output.
            topic_tag = job.topic.strip()[:60]
            extra_tags = [topic_tag] if topic_tag else []
            opts = IngestCallOptions(
                skip_verify=False,
                source_type=SourceType.RESEARCHER,
                source_url=sources[0] if sources else None,
                research_job_id=job.id,
                category_hint=job.category,
                extra_tags=extra_tags,
            )
            out = await self._ingestor.ingest(combined, opts)

            job.tome_ids = [t.id for t in out.tomes]
            job.sources = list(dict.fromkeys(sources))[:100]
            job.finished_at = datetime.now(UTC)

            if not out.tomes:
                job.status = ResearchJobStatus.FAILED
                job.error = out.reject_reason or "Ingest produced no tomes."
            else:
                job.status = ResearchJobStatus.COMPLETED
                job.error = None

            await self._jobs.update(job)

        except Exception as exc:
            is_storage = isinstance(exc, StorageError)
            logging.exception(
                "Research job %s failed%s",
                job_id,
                ": storage error" if is_storage else "",
            )
            job.status = ResearchJobStatus.FAILED
            job.error = f"Storage error: {exc}" if is_storage else str(exc)
            job.finished_at = datetime.now(UTC)
            try:
                await self._jobs.update(job)
            except StorageError:
                logging.exception("Research job %s failure write also failed (storage)", job_id)


async def plan_search_queries(
    topic: str,
    context: str | None,
    config: LibrarianConfig,
    *,
    max_queries: int,
) -> list[str]:
    """Produce search strings; prefer a small JSON call to Ollama when configured."""
    base = config.verification.ollama_base_url.rstrip("/")
    model = config.verification.claim_model
    ctx = f"\nContext: {context}" if context else ""
    prompt = (
        f"Generate {max_queries} distinct web search queries to research the topic below. "
        'Reply with JSON only: {{"queries": ["...", "..."]}}.\n\n'
        f"Topic: {topic}{ctx}"
    )
    url = f"{base}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    }
    queries: list[str] | None = None
    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
        message = data["choices"][0]["message"]["content"].strip()
        if message.startswith("```"):
            message = re.sub(r"^```(?:json)?\s*", "", message)
            message = re.sub(r"\s*```$", "", message)
        parsed = json.loads(message)
        raw = parsed.get("queries") if isinstance(parsed, dict) else None
        if isinstance(raw, list):
            queries = [str(q).strip() for q in raw if str(q).strip()]
    except Exception as exc:
        logging.debug("plan_search_queries LLM fallback: %s", exc)

    if not queries:
        queries = _default_queries(topic, context)
    return queries[:max_queries]


def _default_queries(topic: str, context: str | None) -> list[str]:
    t = topic.strip()
    out = [t, f"{t} overview", f"{t} explained", f"introduction to {t}", f"{t} facts"]
    if context and context.strip():
        out.insert(1, f"{t} {context.strip()[:120]}")
    return out
