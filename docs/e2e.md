# End-to-End Testing

This runbook covers running the Librarian against a real backend and running the
live end-to-end test (`tests/test_e2e_live_mongo.py`), which drives the full
production path — real `LibrarianServer.lifespan` wiring, real
sentence-transformers embeddings, and real Atlas vector search — on a fresh
database.

## Fresh-clone Docker path (MongoDB Atlas + vector search)

This is the production-shaped path: a live Atlas-capable MongoDB plus local
sentence-transformers embeddings.

1. Start the infrastructure (the compose file includes a
   `mongodb/mongodb-atlas-local` service):

   ```bash
   docker compose up -d
   ```

1. Install dependencies, including the heavy `sentence-transformers` extra
   (~870MB, pulls in PyTorch):

   ```bash
   uv sync --extra dev --extra sentence-transformers
   ```

1. Point the server at Mongo. Either set the URI directly:

   ```bash
   export LIBRARIAN_DATABASE_URI=mongodb://localhost:27017
   export LIBRARIAN_EMBEDDING_PROVIDER=sentence-transformers
   export LIBRARIAN_EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
   export LIBRARIAN_EMBEDDING_DIMENSIONS=384
   ```

   or copy the example config and edit it:

   ```bash
   cp librarian.config.example.yaml librarian.config.yaml
   ```

1. Run the server:

   ```bash
   uv run python -m src
   ```

All MongoDB indexes — **including the Atlas vector search index** — are created
automatically at startup by `MongoTomeRepository.ensure_indexes` (invoked from
the lifespan). There is **no manual index setup**: point the server at a fresh,
empty database and the standard secondary indexes plus the `vectors` and
`default` Atlas search indexes are provisioned on first boot. Fresh Atlas search
indexes are eventually-consistent, so queries may return empty results for a
short window after startup while the index finishes building.

## No-Docker local path (filesystem backend)

For a quick local run without Docker or Mongo, use the filesystem backend. Any
`database.uri` that does **not** start with `mongodb` selects the filesystem
repositories (`FsTomeRepository` / `FsResearchJobRepository`):

```bash
uv sync --extra dev --extra sentence-transformers
export LIBRARIAN_DATABASE_URI=/tmp/librarian-data
export LIBRARIAN_EMBEDDING_PROVIDER=sentence-transformers
export LIBRARIAN_EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
export LIBRARIAN_EMBEDDING_DIMENSIONS=384
uv run python -m src
```

This gives you real semantic embeddings with zero external services, at the cost
of Atlas vector search (search falls back to the filesystem implementation).

## Running the live-Mongo e2e test locally

`tests/test_e2e_live_mongo.py` skips unless **both** of the following hold:

- a live MongoDB is reachable at `LIBRARIAN_TEST_MONGO_URI` (default
  `mongodb://localhost:27017/?directConnection=true`), and
- the `sentence-transformers` extra is installed.

To run it locally against the Docker Atlas-local container:

```bash
docker compose up -d
uv sync --extra dev --extra sentence-transformers
export LIBRARIAN_TEST_MONGO_URI=mongodb://localhost:27017/?directConnection=true
uv run pytest tests/test_e2e_live_mongo.py -q
```

The test creates a uniquely-named database, runs the full ingest → search →
get → update → list → tidy → delete cycle through the registered MCP tool
handlers, and drops that database in a `finally` block so nothing is left
behind. The first run downloads the ~90MB MiniLM model into
`~/.cache/huggingface`.

This test runs automatically in CI's `test` job, which provides a
`mongodb/mongodb-atlas-local` service container and installs the
`sentence-transformers` extra. Everywhere else (no Mongo and/or no
sentence-transformers) it skips cleanly.
