from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from src.config import LibrarianConfig
from src.models.tool_schemas import (
    IngestInput,
    IngestOutput,
    ResearchInput,
    ResearchOutput,
    SearchInput,
    SearchOutput,
)
from src.services.embedding import EmbeddingService
from src.services.ingestor import Ingestor
from src.services.researcher import Researcher
from src.services.search import SearchEngine
from src.services.verifier import Verifier
from src.services.web_search import WebSearchClient
from src.storage.database import DatabaseClient
from src.storage.job_repository import JobRepository
from src.storage.tome_repository import TomeRepository

mcp = FastMCP("The Librarian")


def _build_services(config: LibrarianConfig) -> dict:
    """Wire up all service dependencies from config. Returns a dict of named services."""
    ...


@mcp.tool()
async def library_search(
    query: str,
    top_k: int = 5,
    category: str | None = None,
    min_confidence: float = 0.5,
    include_summary: bool = False,
) -> SearchOutput:
    """Search the library for relevant Tomes using semantic vector search."""
    ...


@mcp.tool()
async def library_ingest(
    content: str,
    title: str | None = None,
    category: str | None = None,
    tags: list[str] | None = None,
    source_url: str | None = None,
    skip_verify: bool = False,
    allow_update: bool = True,
) -> IngestOutput:
    """Ingest new knowledge into the library. Validates, chunks, embeds, and stores it."""
    ...


@mcp.tool()
async def library_research(
    topic: str,
    context: str | None = None,
    depth: str = "standard",
    max_tomes: int = 10,
    category: str | None = None,
    async_mode: bool = False,
) -> ResearchOutput:
    """Dispatch a Researcher sub-agent to search the web and create new Tomes."""
    ...


async def start_server(config: LibrarianConfig) -> None:
    """Initialise services, connect to the database, and start the MCP server."""
    ...
