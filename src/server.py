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
        "find information, library_ingest to store new knowledge, and "
        "library_research to discover knowledge from the web."
    ),
)


from src.services.embedding import DummyEmbeddingService
from src.services.ingestor import Ingestor
from src.services.verifier import Verifier
from src.services.web_search import WebSearchClient
from src.storage.filesystem.fs_tome_repository import FsTomeRepository

# Global services dictionary populated by start_server
_services: dict[str, object] = {}

def _build_services(config: LibrarianConfig) -> dict[str, object]:
    """Wire up all service dependencies from config. Returns a dict of named services."""
    embedding = DummyEmbeddingService(config.embedding)
    # Use a dummy web search client for the verifier since we are returning constants
    web_search = WebSearchClient(config.verification)
    verifier = Verifier(config.verification, web_search)
    repo = FsTomeRepository(config.database)
    ingestor = Ingestor(config, embedding, verifier, repo)

    return {
        "embedding": embedding,
        "verifier": verifier,
        "repo": repo,
        "ingestor": ingestor,
    }

@mcp.tool()
async def library_search(params: SearchInput) -> SearchOutput:
    """Search the library for relevant Tomes using semantic vector search."""
    repo: FsTomeRepository = _services["repo"]  # type: ignore
    tomes_with_scores = await repo.search(
        params.query,
        top_k=params.top_k,
        min_confidence=params.min_confidence
    )

    return SearchOutput(
        tomes=[t for t, _ in tomes_with_scores],
        scores=[s for _, s in tomes_with_scores],
        query_id="dummy-query-id",  # Can be generated
        from_cache=False
    )

@mcp.tool()
async def library_ingest(params: IngestInput) -> IngestOutput:
    """Ingest new knowledge into the library. Validates, chunks, embeds, and stores it."""
    ingestor: Ingestor = _services["ingestor"]  # type: ignore
    return await ingestor.ingest(params.content)

async def start_server(config: LibrarianConfig) -> None:
    """Initialise services, connect to the database, and start the MCP server."""
    global _services
    _services = _build_services(config)

    # Initialize the embedding service
    embedding: DummyEmbeddingService = _services["embedding"]  # type: ignore
    await embedding.initialize()

    print("Starting Librarian MCP Server...")
    mcp.run()
