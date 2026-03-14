from mcp.server import FastMCP

from src.config import LibrarianConfig
from src.models import SearchInput
from src.models.tool_schemas import (
    IngestInput,
    IngestOutput,
    SearchOutput,
)

mcp = FastMCP(
    "The Librarian",
    instructions=(
        "An intelligent knowledge management server. Use library_search to "
        "find information and library_ingest to store new knowledge."
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


async def start_server(config: LibrarianConfig) -> None:
    """Initialise services, connect to the database, and start the MCP server."""
    raise NotImplementedError
