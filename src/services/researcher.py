from __future__ import annotations

from src.config import ResearcherSettings
from src.models.tool_schemas import ResearchInput, ResearchOutput
from src.services.ingestor import Ingestor
from src.services.web_search import WebSearchClient
from src.storage.job_repository import JobRepository


class Researcher:
    """Autonomous sub-agent that searches the web, synthesises findings,
    and pipes results through the Ingestor to create new Tomes."""

    def __init__(
        self,
        settings: ResearcherSettings,
        web_search: WebSearchClient,
        ingestor: Ingestor,
        job_repo: JobRepository,
    ) -> None:
        self._settings = settings
        self._web_search = web_search
        self._ingestor = ingestor
        self._job_repo = job_repo

    async def research(self, params: ResearchInput) -> ResearchOutput:
        """Execute the full research pipeline: plan -> search -> extract -> synthesise -> ingest."""
        ...

    async def poll_job(self, job_id: str) -> ResearchOutput:
        """Check the status of an async research job and return results if complete."""
        ...

    async def _plan_queries(self, topic: str, context: str | None) -> list[str]:
        """Generate 3-6 focused search queries covering different facets of the topic."""
        ...

    async def _search_and_collect(
        self, queries: list[str], max_sources_per_query: int
    ) -> list[dict[str, str]]:
        """Issue queries, fetch pages, and return de-duplicated source documents."""
        ...

    async def _synthesise(
        self, sources: list[dict[str, str]], topic: str
    ) -> str:
        """Merge content across sources into a structured, logically organised text."""
        ...

    def _source_count_for_depth(self, depth: str) -> int:
        """Return the number of sources to consult based on depth setting."""
        ...
