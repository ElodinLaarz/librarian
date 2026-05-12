import asyncio
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TypeVar
from uuid import UUID, uuid4

from mcp.server import FastMCP

from src.config import LibrarianConfig
from src.models.enums import ResearchJobStatus
from src.models.research_job import ResearchJob
from src.models.tool_schemas import (
    GetInput,
    GetOutput,
    IngestInput,
    IngestOutput,
    ResearchInput,
    ResearchOutput,
    SearchInput,
    SearchOutput,
    TidyInput,
    TidyOutput,
)
from src.services.embedding import (
    EmbeddingService,
    OllamaEmbeddingService,
    build_embedding_service,
)
from src.services.ingestor import IngestCallOptions, Ingestor
from src.services.researcher import Researcher
from src.services.tidier import Tidier
from src.services.verifier import Verifier
from src.services.web_search import build_web_search_client
from src.storage.filesystem.fs_research_job_repository import FsResearchJobRepository
from src.storage.filesystem.fs_tome_repository import FsTomeRepository
from src.storage.mongo.mongo_research_job_repository import MongoResearchJobRepository
from src.storage.mongo.mongo_tome_repository import MongoTomeRepository
from src.storage.research_job_repository import ResearchJobRepository
from src.storage.tome_repository import TomeRepository

_T = TypeVar("_T")

DEFAULT_CONFIG_PATH = Path("librarian.config.yaml")
EXAMPLE_CONFIG_FILENAME = "librarian.config.example.yaml"


def load_config() -> LibrarianConfig:
    """Resolve the config path and load it, with a friendly error on first run.

    Kept separate from module import so that importing :mod:`src.server` is
    side-effect free; configuration errors only surface when callers (e.g.
    :func:`src.__main__.main`) actually invoke this function.

    Resolution order:
    1. ``LIBRARIAN_CONFIG`` env var (explicit, no fallback message).
    2. ``DEFAULT_CONFIG_PATH`` (``librarian.config.yaml``) in CWD.

    When the default path is missing and no env var is set, attempt to load
    so that environment-only configuration still works; if that fails, raise
    a friendly error pointing users at ``librarian.config.example.yaml``.
    """
    explicit = os.environ.get("LIBRARIAN_CONFIG")
    path = Path(explicit) if explicit else DEFAULT_CONFIG_PATH

    if explicit or path.exists():
        return LibrarianConfig.from_yaml(path)

    # Default path missing and no explicit env var. Allow env-only configuration
    # (e.g. LIBRARIAN_DATABASE_URI is set) by attempting the load; if validation
    # fails, surface a message that points at the example file.
    try:
        return LibrarianConfig.from_yaml(path)
    except ValueError as exc:
        raise ValueError(
            f"No config found at {path} and LIBRARIAN_CONFIG is not set. "
            f"Copy {EXAMPLE_CONFIG_FILENAME} to {DEFAULT_CONFIG_PATH} "
            f"(or set LIBRARIAN_CONFIG to point at your config file) and edit "
            f"as needed.\n\nUnderlying error:\n{exc}"
        ) from exc


class LibrarianServer:
    """Manages the lifecycle and state of the Librarian MCP server."""

    def __init__(self, config: LibrarianConfig) -> None:
        self.config = config
        self.ingestor: Ingestor | None = None
        self.tidier: Tidier | None = None
        self.tome_repo: TomeRepository | None = None
        self.job_repo: ResearchJobRepository | None = None
        self.researcher: Researcher | None = None
        self._embedding_service: EmbeddingService | None = None
        self._bg_tasks: set[asyncio.Task[None]] = set()
        self.mcp = FastMCP(
            "The Librarian",
            instructions=(
                "An intelligent knowledge management server. Use library_search to "
                "find information, library_get to fetch a single tome by ID, "
                "library_ingest to store new knowledge, and library_research to gather "
                "information from the web when the library is thin on a topic. Use "
                "library_tidy to consolidate duplicates."
            ),
            lifespan=self.lifespan,
            host=self.config.server.host,
            port=self.config.server.port,
            log_level=self.config.server.log_level.value.upper(),  # type: ignore[arg-type]
        )

        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)

        self._setup_tools()

    def _track_background_task(self, task: asyncio.Task[None]) -> None:
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

    def _require(self, value: _T | None, name: str) -> _T:
        """Return ``value`` if not ``None`` else raise a clear ``RuntimeError``.

        Tool handlers cannot rely on ``assert`` for this because ``assert``
        statements are stripped under ``python -O``, which would turn an
        un-initialised server into a silent ``AttributeError`` on ``None``
        instead of a clear, actionable error. Using a helper that returns the
        non-``None`` value also preserves type narrowing under mypy strict.
        """
        if value is None:
            raise RuntimeError(
                f"LibrarianServer not initialised — lifespan did not run (missing: {name})"
            )
        return value

    @asynccontextmanager
    async def lifespan(self, _server_mcp: FastMCP) -> AsyncIterator[None]:
        """Initialise and tear down services around the server lifetime."""
        self._embedding_service = await build_embedding_service(self.config.embedding)

        if self.config.database.uri.startswith("mongodb"):
            self.tome_repo = MongoTomeRepository(
                self.config.database,
                self._embedding_service,
                self.config.tidy,
            )
            self.job_repo = MongoResearchJobRepository(self.config.database)
            await self.tome_repo.ensure_indexes()
        else:
            self.tome_repo = FsTomeRepository(
                self.config.database,
                self._embedding_service,
                self.config.tidy,
            )
            self.job_repo = FsResearchJobRepository(self.config.database)

        web_client = build_web_search_client(self.config)
        verifier = Verifier(self.config, web_client)
        self.ingestor = Ingestor(self.config, self._embedding_service, verifier, self.tome_repo)
        self.researcher = Researcher(self.config, web_client, self.ingestor, self.job_repo)
        self.tidier = Tidier(self.ingestor, self.tome_repo, self.config.tidy)

        if not self.config.verification.enabled:
            logging.info("Verification disabled; skipping claim extraction.")
        elif self.config.verification.use_llm_claims:
            logging.info(
                "LLM claim extraction enabled, model=%s (ollama=%s). "
                "Run `ollama pull %s` if you have not already.",
                self.config.verification.claim_model,
                self.config.verification.ollama_base_url,
                self.config.verification.claim_model,
            )
        else:
            logging.info("LLM claim extraction disabled; using heuristic sentence split.")

        if self.config.tidy.enabled:
            task = asyncio.create_task(self._tidy_loop())
            self._track_background_task(task)

        yield

        for t in list(self._bg_tasks):
            t.cancel()
        self._bg_tasks.clear()

        if self.job_repo:
            self.job_repo.close()
            self.job_repo = None
        if self.tome_repo:
            self.tome_repo.close()
        if self.ingestor is not None:
            await self.ingestor.aclose()
        if isinstance(self._embedding_service, OllamaEmbeddingService):
            await self._embedding_service.aclose()
        self._embedding_service = None
        self.ingestor = None
        self.tidier = None
        self.tome_repo = None
        self.researcher = None

    async def _tidy_loop(self) -> None:
        """Background loop for library tidying."""
        while True:
            try:
                await asyncio.sleep(self.config.tidy.interval_seconds)
                if self.tidier:
                    logging.info("Starting background library tidy...")
                    report = await self.tidier.run_cleanup()
                    logging.info("Library tidy complete: %s", report)
            except asyncio.CancelledError:
                break
            except Exception:
                logging.error("Exception in background tidy loop", exc_info=True)
                await asyncio.sleep(60)  # Wait a bit before retrying after error

    async def _research_poll(self, job_id: UUID) -> ResearchOutput:
        job_repo = self._require(self.job_repo, "job_repo")
        job = await job_repo.get_by_id(job_id)
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
            tome_repo = self._require(self.tome_repo, "tome_repo")

            results = await tome_repo.search(
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
            ingestor = self._require(self.ingestor, "ingestor")
            opts = IngestCallOptions(
                skip_verify=params.skip_verify,
                category_hint=params.category,
                tags_hint=params.tags,
                source_url=params.source_url,
                force_format=params.force_format,
            )
            return await ingestor.ingest(params.content, opts)

        @self.mcp.tool()
        async def library_research(params: ResearchInput) -> ResearchOutput:
            """Research a topic on the web and ingest findings as new Tomes."""
            job_repo = self._require(self.job_repo, "job_repo")
            researcher = self._require(self.researcher, "researcher")

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
            await job_repo.insert(job)

            if params.async_:
                task = asyncio.create_task(researcher.run_job(job.id))
                self._track_background_task(task)
                return ResearchOutput(
                    job_id=str(job.id),
                    status=ResearchJobStatus.PENDING.value,
                )

            await researcher.run_job(job.id)
            return await self._research_poll(job.id)

        @self.mcp.tool()
        async def library_get(params: GetInput) -> GetOutput:
            """Fetch a single Tome by its UUID. Returns not_found if the ID is unknown."""
            tome_repo = self._require(self.tome_repo, "tome_repo")

            raw_id = params.tome_id.strip()
            try:
                # Use the default UUID constructor so we accept both hyphenated
                # canonical form (as serialised by Tome.id) and bare-hex form.
                tome_uuid = UUID(raw_id)
            except ValueError:
                return GetOutput(
                    tome=None,
                    status="invalid_tome_id",
                    error="tome_id must be a valid UUID string",
                )

            tome = await tome_repo.get_by_id(tome_uuid)
            if tome is None:
                return GetOutput(
                    tome=None,
                    status="not_found",
                    error=f"No tome found with id {raw_id}",
                )
            # Strip embedding payload before returning to clients (matches search/research).
            return GetOutput(
                tome=tome.model_copy(update={"embedding": None}),
                status="found",
            )

        @self.mcp.tool()
        async def library_tidy(params: TidyInput) -> TidyOutput:
            """Review the library tomes and remove duplicates by combining similar topics."""
            tidier = self._require(self.tidier, "tidier")
            report = await tidier.run_cleanup(
                limit=params.limit,
                threshold=params.threshold,
                skip_verify=params.skip_verify,
            )
            return TidyOutput(**report)
