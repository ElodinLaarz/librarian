import asyncio
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import UUID, uuid4

from mcp.server import FastMCP

from src.config import LibrarianConfig
from src.models.enums import ResearchJobStatus
from src.models.research_job import ResearchJob
from src.models.tool_schemas import (
    IngestInput,
    IngestOutput,
    ResearchInput,
    ResearchOutput,
    SearchInput,
    SearchOutput,
)
from src.services.embedding import (
    EmbeddingService,
    OllamaEmbeddingService,
    build_embedding_service,
)
from src.services.ingestor import IngestCallOptions, Ingestor
from src.services.researcher import Researcher
from src.services.verifier import Verifier
from src.services.web_search import build_web_search_client
from src.storage.filesystem.fs_research_job_repository import FsResearchJobRepository
from src.storage.filesystem.fs_tome_repository import FsTomeRepository
from src.storage.mongo.mongo_research_job_repository import MongoResearchJobRepository
from src.storage.mongo.mongo_tome_repository import MongoTomeRepository
from src.storage.research_job_repository import ResearchJobRepository
from src.storage.tome_repository import TomeRepository

config_path = Path(os.environ.get("LIBRARIAN_CONFIG", "librarian.config.yaml"))
config = LibrarianConfig.from_yaml(config_path)


class LibrarianServer:
    """Manages the lifecycle and state of the Librarian MCP server."""

    def __init__(self, config: LibrarianConfig) -> None:
        self.config = config
        self.ingestor: Ingestor | None = None
        self.tome_repo: TomeRepository | None = None
        self.job_repo: ResearchJobRepository | None = None
        self.researcher: Researcher | None = None
        self._embedding_service: EmbeddingService | None = None
        self._bg_tasks: set[asyncio.Task[None]] = set()
        self.mcp = FastMCP(
            "The Librarian",
            instructions=(
                "An intelligent knowledge management server. Use library_search to "
                "find information, library_ingest to store new knowledge, and "
                "library_research to gather information from the web when the library "
                "is thin on a topic."
            ),
            lifespan=self.lifespan,
            host=config.server.host,
            port=config.server.port,
            log_level=config.server.log_level.value.upper(),  # type: ignore[arg-type]
        )

        import logging

        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)

        self._setup_tools()

    def _track_background_task(self, task: asyncio.Task[None]) -> None:
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

    @asynccontextmanager
    async def lifespan(self, _server_mcp: FastMCP) -> AsyncIterator[None]:
        """Initialise and tear down services around the server lifetime."""
        self._embedding_service = await build_embedding_service(self.config.embedding)

        if self.config.database.uri.startswith("mongodb"):
            self.tome_repo = MongoTomeRepository(self.config.database, self._embedding_service)
            self.job_repo = MongoResearchJobRepository(self.config.database)
            await self.tome_repo.ensure_indexes()
        else:
            self.tome_repo = FsTomeRepository(self.config.database, self._embedding_service)
            self.job_repo = FsResearchJobRepository(self.config.database)

        web_client = build_web_search_client(self.config)
        verifier = Verifier(self.config, web_client)
        self.ingestor = Ingestor(self.config, self._embedding_service, verifier, self.tome_repo)
        self.researcher = Researcher(self.config, web_client, self.ingestor, self.job_repo)

        yield

        for t in list(self._bg_tasks):
            t.cancel()
        self._bg_tasks.clear()

        if self.job_repo:
            self.job_repo.close()
            self.job_repo = None
        if self.tome_repo:
            self.tome_repo.close()
        if isinstance(self._embedding_service, OllamaEmbeddingService):
            await self._embedding_service.aclose()
        self._embedding_service = None
        self.ingestor = None
        self.tome_repo = None
        self.researcher = None

    async def _research_poll(self, job_id: UUID) -> ResearchOutput:
        assert self.job_repo is not None
        job = await self.job_repo.get_by_id(job_id)
        if job is None:
            return ResearchOutput(
                job_id=str(job_id),
                status="not_found",
                error="Unknown job_id",
            )
        tomes = []
        if job.status == ResearchJobStatus.COMPLETED and self.tome_repo is not None:
            for tid in job.tome_ids:
                t = await self.tome_repo.get_by_id(tid)
                if t:
                    tomes.append(t.model_copy(update={"embedding": None}))
        return ResearchOutput(
            job_id=str(job.id),
            status=job.status.value,
            tome_ids=[str(x) for x in job.tome_ids],
            tomes=tomes,
            sources=job.sources,
            query_count=job.query_count,
            error=job.error,
        )

    def _setup_tools(self) -> None:
        """Register MCP tools."""

        @self.mcp.tool()
        async def library_search(params: SearchInput) -> SearchOutput:
            """Search the library for relevant Tomes using semantic vector search."""
            assert self.tome_repo is not None, "Server not initialised"

            results = await self.tome_repo.search(
                query=params.query,
                top_k=params.top_k,
                min_confidence=params.min_confidence,
                category=params.category,
            )

            scores = [s for _, s in results]

            # Strip embeddings and optionally summaries before returning to clients.
            update_data: dict[str, object] = {"embedding": None}
            if not params.include_summary:
                update_data["summary"] = ""
            tomes = [t.model_copy(update=update_data) for t, _ in results]

            return SearchOutput(
                tomes=tomes,
                scores=scores,
                query_id=uuid4().hex,
                from_cache=False,
            )

        @self.mcp.tool()
        async def library_ingest(params: IngestInput) -> IngestOutput:
            """Ingest new knowledge into the library. Validates, chunks, embeds, and stores it."""
            assert self.ingestor is not None, "Server not initialised"
            opts = IngestCallOptions(
                skip_verify=params.skip_verify,
                category_hint=params.category,
                tags_hint=params.tags,
                source_url=params.source_url,
            )
            return await self.ingestor.ingest(params.content, opts)

        @self.mcp.tool()
        async def library_research(params: ResearchInput) -> ResearchOutput:
            """Research a topic on the web and ingest findings as new Tomes."""
            assert self.job_repo is not None and self.researcher is not None, (
                "Server not initialised"
            )

            if params.job_id:
                job_id_str = params.job_id.strip()
                try:
                    jid = UUID(hex=job_id_str)
                except ValueError:
                    return ResearchOutput(
                        job_id=job_id_str,
                        status="invalid_job_id",
                        error="job_id must be a UUID hex string",
                    )
                return await self._research_poll(jid)

            if not params.topic or not params.topic.strip():
                return ResearchOutput(
                    job_id="",
                    status="invalid_input",
                    error="topic is required when job_id is not set",
                )

            job = ResearchJob(
                id=uuid4(),
                topic=params.topic.strip(),
                context=params.context.strip() if params.context else None,
                depth=params.depth,
                max_tomes=params.max_tomes,
                category=params.category,
            )
            await self.job_repo.insert(job)

            if params.async_:
                task = asyncio.create_task(self.researcher.run_job(job.id))
                self._track_background_task(task)
                return ResearchOutput(
                    job_id=str(job.id),
                    status=ResearchJobStatus.PENDING.value,
                )

            await self.researcher.run_job(job.id)
            return await self._research_poll(job.id)


_server = LibrarianServer(config)
mcp: FastMCP = _server.mcp  # type: ignore[has-type]
