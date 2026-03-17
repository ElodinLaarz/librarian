from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import uuid4

from mcp.server import FastMCP

from src.config import LibrarianConfig
from src.models.tool_schemas import (
    IngestInput,
    IngestOutput,
    SearchInput,
    SearchOutput,
)
from src.services.embedding import DummyEmbeddingService
from src.services.ingestor import Ingestor
from src.services.verifier import Verifier
from src.storage.filesystem.fs_tome_repository import FsTomeRepository
from src.storage.tome_repository import TomeRepository

config = LibrarianConfig()

_ingestor: Ingestor | None = None
_tome_repo: TomeRepository | None = None


@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[None]:
    """Initialise and tear down services around the server lifetime."""
    global _ingestor, _tome_repo

    _tome_repo = _build_tome_repository(config)
    embedding_service = DummyEmbeddingService(config.embedding)
    verifier = Verifier(config)
    _ingestor = Ingestor(config, embedding_service, verifier, _tome_repo)

    yield

    _ingestor = None
    _tome_repo = None


def _build_tome_repository(config: LibrarianConfig) -> TomeRepository:
    """Construct the active TomeRepository from config."""
    return FsTomeRepository(config.database)


mcp = FastMCP(
    "The Librarian",
    instructions=(
        "An intelligent knowledge management server. Use library_search to "
        "find information and library_ingest to store new knowledge."
    ),
    lifespan=lifespan,
)


@mcp.tool()
async def library_search(params: SearchInput) -> SearchOutput:
    """Search the library for relevant Tomes using semantic vector search."""
    assert _tome_repo is not None, "Server not initialised"

    results = await _tome_repo.search(
        query=params.query,
        top_k=params.top_k,
        min_confidence=params.min_confidence,
        category=params.category,
    )

    tomes = [tome for tome, _ in results]
    if not params.include_summary:
        tomes = [t.model_copy(update={"summary": ""}) for t in tomes]
    scores = [score for _, score in results]

    return SearchOutput(
        tomes=tomes,
        scores=scores,
        query_id=uuid4().hex,
        from_cache=False,
    )


@mcp.tool()
async def library_ingest(params: IngestInput) -> IngestOutput:
    """Ingest new knowledge into the library. Validates, chunks, embeds, and stores it."""
    assert _ingestor is not None, "Server not initialised"
    return await _ingestor.ingest(params.content)
