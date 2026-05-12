import asyncio
import logging
import os
import re
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, TypeVar
from uuid import UUID, uuid4

from mcp.server import FastMCP
from motor.motor_asyncio import AsyncIOMotorClient

from src.config import LibrarianConfig
from src.models.enums import ResearchJobStatus
from src.models.research_job import ResearchJob
from src.models.tome import Tome
from src.models.tool_schemas import (
    DeleteInput,
    DeleteOutput,
    GetInput,
    GetOutput,
    IngestInput,
    IngestOutput,
    ListInput,
    ListOutput,
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
from src.storage.mongo.client import build_motor_client
from src.storage.mongo.mongo_research_job_repository import MongoResearchJobRepository
from src.storage.mongo.mongo_tome_repository import MongoTomeRepository
from src.storage.research_job_repository import ResearchJobRepository
from src.storage.tome_repository import TomeRepository

_T = TypeVar("_T")

DEFAULT_CONFIG_PATH = Path("librarian.config.yaml")
EXAMPLE_CONFIG_FILENAME = "librarian.config.example.yaml"

# Splits a search query into lexical terms used for snippet extraction.
_QUERY_TERM_RE = re.compile(r"\w+", re.UNICODE)
# Ellipsis appended when content is truncated. Kept as a constant so tests and
# callers agree on the exact marker.
_TRUNCATION_ELLIPSIS = "..."


def _truncate(text: str, max_chars: int) -> str:
    """Truncate ``text`` to ``max_chars`` total characters, appending an ellipsis.

    The returned string is guaranteed to be at most ``max_chars`` characters
    long. When ``max_chars`` is smaller than the ellipsis itself, the text is
    cut hard with no marker.
    """
    if len(text) <= max_chars:
        return text
    if max_chars <= len(_TRUNCATION_ELLIPSIS):
        return text[:max_chars]
    return text[: max_chars - len(_TRUNCATION_ELLIPSIS)] + _TRUNCATION_ELLIPSIS


def _compile_query_pattern(query: str) -> re.Pattern[str] | None:
    """Compile a case-insensitive regex matching any lexical token in ``query``.

    Returns ``None`` when the query yields no tokens. Tokenising once per
    `library_search` invocation (rather than once per hit) avoids redundant
    work, and using a single regex with ``re.IGNORECASE`` lets snippet
    extraction skip an expensive ``content.lower()`` copy on each hit — a
    real concern given that stored content can run to hundreds of KB.
    """
    terms = _QUERY_TERM_RE.findall(query)
    if not terms:
        return None
    return re.compile("|".join(re.escape(t) for t in terms), re.IGNORECASE)


def _extract_snippet(content: str, pattern: re.Pattern[str] | None, window: int) -> str:
    """Return a ``window``-char excerpt of ``content`` centred on the earliest
    case-insensitive match of ``pattern``.

    Falls back to the leading ``window`` characters when no term matches.
    Adds leading/trailing ellipses when the excerpt does not touch the
    respective edge of the source, so the agent knows the snippet is partial.
    """
    if window <= 0 or not content:
        return ""
    if len(content) <= window:
        return content

    match_idx = -1
    match_len = 0
    if pattern is not None:
        match = pattern.search(content)
        if match is not None:
            match_idx = match.start()
            match_len = match.end() - match.start()

    if match_idx == -1:
        # Leading window with trailing ellipsis (we know content is longer than window).
        return content[:window] + _TRUNCATION_ELLIPSIS

    centre = match_idx + match_len // 2
    half = window // 2
    start = max(0, centre - half)
    end = start + window
    if end > len(content):
        end = len(content)
        start = max(0, end - window)

    excerpt = content[start:end]
    if start > 0:
        excerpt = _TRUNCATION_ELLIPSIS + excerpt
    if end < len(content):
        excerpt = excerpt + _TRUNCATION_ELLIPSIS
    return excerpt


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
        # Shared Motor client owned by the lifespan and handed to every
        # Mongo-backed repo so the process maintains a single connection pool
        # (issue #25). ``None`` when the backend is filesystem.
        self._mongo_client: AsyncIOMotorClient[Mapping[str, Any]] | None = None
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

    async def _shutdown_background_tasks(self, timeout: float = 5.0) -> None:
        """Cancel tracked background tasks and wait for them to unwind.

        Issue #23: previously the lifespan handler only called ``cancel()``
        and cleared ``_bg_tasks``, never awaiting the tasks. That meant any
        in-flight work (HTTP requests, Mongo writes, partial ResearchJob
        records) was torn down mid-flight without giving the task a chance
        to clean up, run ``finally`` blocks, or propagate cancellation
        through pending I/O.

        This helper requests cancellation, then awaits all tasks with a
        timeout (so a malicious / stuck task cannot block server shutdown
        indefinitely). Exceptions surfaced during the unwind are logged but
        not re-raised so a single failing task does not abort the rest of
        the lifespan teardown (closing repositories, embedding services).
        """
        if not self._bg_tasks:
            return

        pending = list(self._bg_tasks)
        for t in pending:
            t.cancel()

        try:
            results: list[BaseException | None] = await asyncio.wait_for(
                asyncio.gather(*pending, return_exceptions=True),
                timeout=timeout,
            )
        except TimeoutError:
            logging.error(
                "Background tasks did not finish within %.1fs of shutdown; "
                "%d task(s) still pending — abandoning to avoid blocking shutdown.",
                timeout,
                sum(1 for t in pending if not t.done()),
            )
            # Preserve telemetry for tasks that DID complete (with or without
            # exception) before the timeout fired. Tasks still running are
            # represented as ``None`` since we have no result to inspect yet.
            results = [t.exception() if t.done() and not t.cancelled() else None for t in pending]

        for t, result in zip(pending, results, strict=False):
            if result is None or isinstance(result, asyncio.CancelledError):
                continue
            if isinstance(result, BaseException):
                logging.error(
                    "Background task %s raised during shutdown: %s",
                    t.get_name(),
                    result,
                    exc_info=result,
                )

        # Abandon all tracked tasks. Tasks that completed have already been
        # removed by the done_callback registered in ``_track_background_task``;
        # tasks that timed out are explicitly dropped here so they cannot leak
        # references to repositories/services that the rest of the lifespan
        # teardown is about to close.
        self._bg_tasks.clear()

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
            # Build the single Motor client up front so both repos share one
            # connection pool / TLS handshake (issue #25). Ownership lives on
            # the server; repos receive it via ``client=`` and must NOT close
            # it themselves.
            self._mongo_client = build_motor_client(self.config.database)
            self.tome_repo = MongoTomeRepository(
                self.config.database,
                self._embedding_service,
                self.config.tidy,
                client=self._mongo_client,
            )
            self.job_repo = MongoResearchJobRepository(
                self.config.database,
                client=self._mongo_client,
            )
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

        # Cancel + await in-flight background tasks so they get a chance to
        # unwind cleanly (issue #23) before we tear down their dependencies.
        await self._shutdown_background_tasks(timeout=5.0)

        if self.job_repo:
            self.job_repo.close()
            self.job_repo = None
        if self.tome_repo:
            self.tome_repo.close()
        # Repo .close() is a no-op for the shared client; the lifespan owns it
        # and must tear it down here so the connection pool actually drains.
        if self._mongo_client is not None:
            self._mongo_client.close()
            self._mongo_client = None
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
            """Search the library for relevant Tomes using semantic vector search.

            The returned ``content`` may be capped or snippet-ised to keep
            agent context windows under control (issue #48). The full content
            of any hit can be retrieved later via its ``tome_id``.
            """
            tome_repo = self._require(self.tome_repo, "tome_repo")

            results = await tome_repo.search(
                query=params.query,
                top_k=params.top_k,
                min_confidence=params.min_confidence,
                category=params.category,
            )

            scores = [s for _, s in results]
            search_cfg = self.config.search

            # Compile the query pattern once for all snippet extractions —
            # tokenising per-hit would re-run the same regex setup ``top_k``
            # times for no gain.
            query_pattern = _compile_query_pattern(params.query)

            # Resolve effective payload knobs: per-call overrides win, with
            # the server config defaults filling in when callers do not opt in.
            effective_max_chars = (
                params.content_max_chars
                if params.content_max_chars is not None
                else search_cfg.default_content_max_chars
            )
            effective_snippet_chars = (
                params.snippet_chars
                if params.snippet_chars is not None
                else search_cfg.default_snippet_chars
            )

            tomes: list[Tome] = []
            is_snippet: list[bool] = []
            tome_ids: list[str] = []
            for t, _ in results:
                tome_ids.append(str(t.id))
                modified = False

                # 1. Build the content payload according to caller preferences.
                if not params.include_content:
                    new_content = ""
                    modified = True
                elif effective_snippet_chars and effective_snippet_chars > 0:
                    snippet = _extract_snippet(t.content, query_pattern, effective_snippet_chars)
                    # Treat the hit as snippet-ised whenever we ran extraction
                    # against content longer than the requested window.
                    modified = len(t.content) > effective_snippet_chars
                    new_content = snippet
                else:
                    new_content = t.content

                # 2. Apply the absolute cap on top of whatever payload we have.
                if effective_max_chars is not None and len(new_content) > effective_max_chars:
                    new_content = _truncate(new_content, effective_max_chars)
                    modified = True

                update_data: dict[str, object] = {
                    "embedding": None,
                    "content": new_content,
                }
                if not params.include_summary:
                    update_data["summary"] = ""
                tomes.append(t.model_copy(update=update_data))
                is_snippet.append(modified)

            return SearchOutput(
                tomes=tomes,
                scores=scores,
                query_id=uuid4().hex,
                from_cache=False,
                tome_ids=tome_ids,
                is_snippet=is_snippet,
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

        @self.mcp.tool()
        async def library_delete(params: DeleteInput) -> DeleteOutput:
            """Permanently remove a tome from the library by its UUID id.

            Accepts either the 32-char hex form or the standard hyphenated
            form. Returns ``deleted=True`` on success. If the id is malformed
            the response carries ``error="invalid_id"``; if no tome matches
            the id it carries ``error="not_found"``. Validation problems are
            reported in the output rather than raised so MCP clients can
            recover.
            """
            tome_repo = self._require(self.tome_repo, "tome_repo")
            try:
                tid = UUID(params.tome_id)
            except (ValueError, AttributeError, TypeError):
                return DeleteOutput(deleted=False, error="invalid_id")

            deleted = await tome_repo.delete(tid)
            if not deleted:
                return DeleteOutput(deleted=False, error="not_found")
            return DeleteOutput(deleted=True)

        @self.mcp.tool()
        async def library_list(params: ListInput) -> ListOutput:
            """Browse / paginate the library.

            Returns a page of Tomes plus the total count matching the filter
            (ignoring limit/offset). Embeddings are stripped to keep the
            response payload small; pass ``include_summary=False`` to strip
            summaries as well when the agent only needs titles + IDs.
            """
            tome_repo = self._require(self.tome_repo, "tome_repo")

            research_job_uuid: UUID | None = None
            if params.research_job_id is not None:
                try:
                    research_job_uuid = UUID(hex=params.research_job_id.strip())
                except ValueError as exc:
                    raise ValueError("research_job_id must be a UUID hex string") from exc

            page, total = await asyncio.gather(
                tome_repo.list_all(
                    limit=params.limit,
                    offset=params.offset,
                    category=params.category,
                    min_confidence=params.min_confidence,
                    research_job_id=research_job_uuid,
                ),
                tome_repo.count(
                    category=params.category,
                    min_confidence=params.min_confidence,
                    research_job_id=research_job_uuid,
                ),
            )

            update_data: dict[str, object] = {"embedding": None}
            if not params.include_summary:
                update_data["summary"] = ""
            tomes = [t.model_copy(update=update_data) for t in page]

            return ListOutput(
                tomes=tomes,
                total=total,
                has_more=(params.offset + len(page)) < total,
            )
