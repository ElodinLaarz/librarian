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
async def library_search(params: SearchInput) -> SearchOutput:
    """Search the library for relevant Tomes using semantic vector search."""
    ...


@mcp.tool()
async def library_ingest(params: IngestInput) -> IngestOutput:
    """Ingest new knowledge into the library. Validates, chunks, embeds, and stores it."""
    ...


@mcp.tool()
async def library_research(params: ResearchInput) -> ResearchOutput:
    """Dispatch a Researcher sub-agent to search the web and create new Tomes."""
    ...


async def start_server(config: LibrarianConfig) -> None:
    """Initialise services, connect to the database, and start the MCP server."""
    ...
