"""Live-MongoDB end-to-end test of the full production server path.

Unlike ``tests/test_e2e_ingest_search_research.py`` (which hand-wires
filesystem repositories and stub embeddings), this test enters the **real**
``LibrarianServer.lifespan`` against a live MongoDB Atlas cluster with real
sentence-transformers embeddings. That means it exercises the production
wiring exactly as ``python -m src`` would: ``build_embedding_service``,
``build_motor_client``, and ``MongoTomeRepository.ensure_indexes`` all run on a
FRESH, uniquely-named database, and search goes through real Atlas vector
search.

The point of the test is to prove that a fresh clone against a fresh database
"just works": every index — including the Atlas vector index — is created
automatically at startup, ingest stores tomes, and a PARAPHRASED query
retrieves them via semantic vector similarity (not substring matching).

Skip conditions (both evaluated at collection time, each with a clear reason):

1. No live MongoDB reachable at ``LIBRARIAN_TEST_MONGO_URI`` (default
   ``mongodb://localhost:27017/?directConnection=true``), detected with the
   same 500ms-ping guard used by ``tests/test_mongo_repository.py``.
2. ``sentence_transformers`` is not importable (the optional heavy extra is
   not installed). Detected via ``importlib.util.find_spec`` so the module
   never imports sentence-transformers at collection time.

In CI's ``test`` job both conditions are satisfied (a
``mongodb/mongodb-atlas-local`` service container is running and the
``sentence-transformers`` extra is installed), so this test executes there and
skips cleanly everywhere else.
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
from typing import Any
from uuid import uuid4

import pytest
from pymongo import MongoClient
from pymongo.errors import PyMongoError

from src.config import (
    DatabaseSettings,
    EmbeddingSettings,
    IngestSettings,
    LibrarianConfig,
    TidySettings,
    VerificationSettings,
)
from src.models.tool_schemas import (
    DeleteInput,
    GetInput,
    IngestInput,
    ListInput,
    SearchInput,
    TidyInput,
    UpdateInput,
)
from src.server import LibrarianServer

# ── skip guards ─────────────────────────────────────────────────────────────

DEFAULT_TEST_MONGO_URI = "mongodb://localhost:27017/?directConnection=true"
TEST_MONGO_URI = os.environ.get("LIBRARIAN_TEST_MONGO_URI", DEFAULT_TEST_MONGO_URI)

# The verified sentence-transformers MiniLM model. ``SentenceTransformerEmbedding
# Service`` passes ``settings.model_name`` straight to ``SentenceTransformer(...)``;
# this fully-qualified id loads ``sentence-transformers/all-MiniLM-L6-v2`` at 384
# dimensions (matching the README stack table and design.md).
MINILM_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MINILM_DIMENSIONS = 384

# Fresh Atlas search indexes are eventually-consistent: after ensure_indexes
# creates them, queries succeed but return EMPTY (not errors) until the index
# finishes building. Poll on this cadence/budget until a hit appears.
_SEARCH_POLL_INTERVAL_S = 2.0
_SEARCH_POLL_DEADLINE_S = 120.0


def _mongo_available(uri: str) -> bool:
    """Ping the configured Mongo URI with a short timeout.

    Returns True iff a ``ping`` command succeeds. Any pymongo / network failure
    is interpreted as "no live Mongo" and triggers a clean skip. Mirrors the
    guard in ``tests/test_mongo_repository.py``.
    """
    client: MongoClient | None = None
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=500)
        client.admin.command("ping")
        return True
    except (PyMongoError, OSError):
        return False
    finally:
        if client is not None:
            client.close()


# ``find_spec`` never imports the package, so collection stays cheap and does
# not require the heavy sentence-transformers/torch stack to be present.
_SENTENCE_TRANSFORMERS_AVAILABLE = importlib.util.find_spec("sentence_transformers") is not None

pytestmark = [
    pytest.mark.skipif(
        not _mongo_available(TEST_MONGO_URI),
        reason=(
            f"No live MongoDB reachable at {TEST_MONGO_URI!r}. "
            "Set LIBRARIAN_TEST_MONGO_URI to a running Atlas-capable instance to "
            "enable the live-Mongo e2e test."
        ),
    ),
    pytest.mark.skipif(
        not _SENTENCE_TRANSFORMERS_AVAILABLE,
        reason=(
            "sentence_transformers is not installed. Install the extra "
            "(`uv sync --extra sentence-transformers`) to enable the live-Mongo e2e test."
        ),
    ),
]


# ── helpers ─────────────────────────────────────────────────────────────────


def _make_live_config(database: str) -> LibrarianConfig:
    """Build a production-shaped config: live Mongo + real MiniLM embeddings.

    Every LLM-bound feature is disabled so the test needs no Ollama / web-search
    provider, but storage, embedding, indexing, and search all use the real
    production code paths.
    """
    return LibrarianConfig(
        database=DatabaseSettings(uri=TEST_MONGO_URI, database=database, tls=False),
        embedding=EmbeddingSettings(
            provider="sentence-transformers",
            model_name=MINILM_MODEL_NAME,
            dimensions=MINILM_DIMENSIONS,
        ),
        ingest=IngestSettings(
            use_llm_chunking=False,
            use_llm_summary=False,
            use_llm_classification=False,
        ),
        verification=VerificationSettings(enabled=False),
        tidy=TidySettings(enabled=False),
    )


def _tool(server: LibrarianServer, name: str) -> Any:
    """Look up a registered MCP tool handler by name."""
    return next(t.fn for t in server.mcp._tool_manager.list_tools() if t.name == name)


# ── main E2E walk ───────────────────────────────────────────────────────────


async def test_e2e_live_mongo_full_production_path() -> None:
    """Drive the whole tool surface through the real lifespan on a fresh DB.

    Enters ``LibrarianServer.lifespan`` — no hand-wiring — so the test exercises
    ``build_embedding_service`` (real MiniLM), ``build_motor_client``, and
    ``ensure_indexes`` (auto-creates every index, including the Atlas vector
    index) on a brand-new, uniquely-named database. The unique database is
    dropped in a ``finally`` so a failing assertion never strands a database or
    its Atlas search indexes.
    """
    database = f"e2e_lifespan_{uuid4().hex[:8]}"
    config = _make_live_config(database)
    server = LibrarianServer(config)

    async with server.lifespan(server.mcp):
        try:
            library_ingest = _tool(server, "library_ingest")
            library_search = _tool(server, "library_search")
            library_get = _tool(server, "library_get")
            library_update = _tool(server, "library_update")
            library_list = _tool(server, "library_list")
            library_tidy = _tool(server, "library_tidy")
            library_delete = _tool(server, "library_delete")

            # ── Step 1: ingest ───────────────────────────────────────────
            ingest_out = await library_ingest(
                IngestInput(
                    content=(
                        "The mitochondrion is the powerhouse of the cell. "
                        "It produces ATP via oxidative phosphorylation. "
                        "Mitochondria are double-membrane-bound organelles found in most "
                        "eukaryotic cells. They serve as the energy production centers, "
                        "converting nutrients into adenosine triphosphate (ATP) through the "
                        "process of cellular respiration. The inner mitochondrial membrane "
                        "contains the electron transport chain and ATP synthase, which are "
                        "crucial for oxidative phosphorylation. This process involves the "
                        "transfer of electrons through a series of protein complexes, creating "
                        "a proton gradient that drives ATP synthesis. Cells with high energy "
                        "demands, such as muscle cells and neurons, contain numerous "
                        "mitochondria to meet their substantial ATP requirements."
                    ),
                    skip_verify=True,
                    category="science",
                    tags=["biology", "cell"],
                )
            )
            assert ingest_out.tomes, f"ingest produced no tomes: {ingest_out!r}"
            ingested_ids = {t.id for t in ingest_out.tomes}
            target_id = next(iter(ingested_ids))

            # ── Step 2: search (paraphrased query, poll until index ready) ─
            # A paraphrase with no verbatim overlap forces genuine semantic
            # vector retrieval rather than lexical/substring matching.
            search_out = None
            deadline = asyncio.get_running_loop().time() + _SEARCH_POLL_DEADLINE_S
            while True:
                search_out = await library_search(
                    SearchInput(
                        query="how do cells generate energy from nutrients",
                        min_confidence=0.0,
                    )
                )
                returned_ids = {t.id for t in search_out.tomes}
                if ingested_ids & returned_ids:
                    break
                if asyncio.get_running_loop().time() > deadline:
                    pytest.fail(
                        "Atlas vector search never returned an ingested tome within "
                        f"{_SEARCH_POLL_DEADLINE_S:.0f}s. Last result: "
                        f"tome_ids={search_out.tome_ids!r} scores={search_out.scores!r}"
                    )
                await asyncio.sleep(_SEARCH_POLL_INTERVAL_S)
            assert len(search_out.scores) == len(search_out.tomes)

            # ── Step 3: get ──────────────────────────────────────────────
            get_out = await library_get(GetInput(tome_id=str(target_id)))
            assert get_out.status == "found", f"unexpected get status: {get_out!r}"
            assert get_out.tome is not None

            # ── Step 4: update + verify the change is persisted ──────────
            update_out = await library_update(
                UpdateInput(
                    tome_id=str(target_id),
                    category="cellular-biology",
                    tags=["mitochondria", "energy"],
                )
            )
            assert update_out.status == "updated", f"unexpected update status: {update_out!r}"
            after_update = await library_get(GetInput(tome_id=str(target_id)))
            assert after_update.status == "found"
            assert after_update.tome is not None
            assert after_update.tome.category == "cellular-biology"
            assert after_update.tome.tags == ["mitochondria", "energy"]

            # ── Step 5: list ─────────────────────────────────────────────
            list_out = await library_list(ListInput())
            assert list_out.total >= 1
            listed_ids = {t.id for t in list_out.tomes}
            assert target_id in listed_ids, (
                f"ingested tome {target_id} not present in list results {listed_ids}"
            )

            # ── Step 6: tidy (numeric report, must not raise) ────────────
            tidy_out = await library_tidy(TidyInput())
            assert isinstance(tidy_out.scanned, int)
            assert tidy_out.scanned >= 0

            # ── Step 7: delete + verify it is gone ───────────────────────
            delete_out = await library_delete(DeleteInput(tome_id=str(target_id)))
            assert delete_out.deleted is True, f"unexpected delete result: {delete_out!r}"
            after_delete = await library_get(GetInput(tome_id=str(target_id)))
            assert after_delete.status == "not_found"
        finally:
            # Leak-proof teardown: drop the unique test database (and its Atlas
            # search indexes) via the lifespan-owned Motor client, even if an
            # assertion above failed. The lifespan's own __aexit__ handles
            # client / repo / background-task cleanup.
            client = server._mongo_client
            assert client is not None, "lifespan did not build a Mongo client"
            await client.drop_database(database)
