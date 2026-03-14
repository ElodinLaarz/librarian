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

mcp = FastMCP(
    "The Librarian",
    instructions=(
        "An intelligent knowledge management server. Use library_search to "
        "find information, library_ingest to store new knowledge, and "
        "library_research to discover knowledge from the web."
    ),
)


def _build_services(config: LibrarianConfig) -> dict[str, object]:
    """Wire up all service dependencies from config. Returns a dict of named services."""
    raise NotImplementedError


@mcp.tool()
async def library_search(params: SearchInput) -> SearchOutput:
    """Search the library for relevant Tomes using semantic vector search."""
    raise NotImplementedError


@mcp.tool()
async def library_ingest(params: IngestInput) -> IngestOutput:
    """Ingest new knowledge into the library. Validates, chunks, embeds, and stores it."""
    raise NotImplementedError


@mcp.tool()
async def library_research(params: ResearchInput) -> ResearchOutput:
    """Dispatch a Researcher sub-agent to search the web and create new Tomes."""
    raise NotImplementedError


async def start_server(config: LibrarianConfig) -> None:
    """Initialise services, connect to the database, and start the MCP server."""
    raise NotImplementedError


