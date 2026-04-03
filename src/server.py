import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
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
from src.storage.mongo import MongoTomeRepository
from src.storage.tome_repository import TomeRepository

config_path = Path(os.environ.get("LIBRARIAN_CONFIG", "config.yml"))
config = LibrarianConfig.from_yaml(config_path)


class LibrarianServer:
    """Manages the lifecycle and state of the Librarian MCP server."""

    def __init__(self, config: LibrarianConfig) -> None:
        self.config = config
        self.ingestor: Ingestor | None = None
        self.tome_repo: TomeRepository | None = None
        self.mcp = FastMCP(
            "The Librarian",
            instructions=(
                "An intelligent knowledge management server. Use library_search to "
                "find information and library_ingest to store new knowledge."
            ),
            lifespan=self.lifespan,
        )
        self._setup_tools()

    @asynccontextmanager
    async def lifespan(self, server: FastMCP) -> AsyncIterator[None]:
        """Initialise and tear down services around the server lifetime."""
        embedding_service = DummyEmbeddingService(self.config.embedding)
        self.tome_repo = MongoTomeRepository(self.config.database, embedding_service)
        verifier = Verifier(self.config)
        self.ingestor = Ingestor(self.config, embedding_service, verifier, self.tome_repo)

        yield

        if self.tome_repo:
            self.tome_repo.close()
        self.ingestor = None
        self.tome_repo = None

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
            return await self.ingestor.ingest(params.content)


_server = LibrarianServer(config)
mcp: FastMCP = _server.mcp  # type: ignore[has-type]
