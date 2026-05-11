# librarian

An MCP server that gives AI agents a persistent, searchable knowledge base. The Librarian stores verified, bite-sized knowledge documents called **Tomes** — single-topic documents with dense semantic embeddings — and exposes them via three MCP tools.

## Tools

| Tool | Status | Description |
| --- | --- | --- |
| `library.search` | Implemented | Hybrid vector + lexical search over stored tomes, with optional category and confidence filtering |
| `library.ingest` | Implemented | Splits content into shards, generates embeddings, deduplicates, and stores tomes; optional `skip_verify`, `category`, `tags`, `source_url` |
| `library.research` | Implemented | Plans queries (Ollama optional), searches the web, fetches pages, ingests findings; `async: true` returns a `job_id` for polling |

### Usage pattern

Call `library.search` first. If results are sparse or low-confidence, call `library.research` to populate the library on that topic. Call `library.ingest` whenever the agent learns something worth persisting.

## Data Model

The core unit is a **Tome**: a compact, single-topic document (100–400 words) with a title, content, one-to-two sentence summary, category, tags, source provenance, a `confidence` score (0.0–1.0), and a dense embedding vector. Tomes are intentionally small to keep retrieval precise.

## Verification

Before storing, a Verifier cross-references key factual claims against web search results:

- **confidence > 0.7** — stored as-is
- **confidence 0.3–0.7** — stored with a low-confidence flag
- **confidence < 0.3** — rejected

Set `skip_verify: true` on ingest to bypass (useful for notes or fictional content). With no search API key, verification is skipped and a synthetic confidence of `0.6` is assigned. To enable live checks and `library.research`, set `LIBRARIAN_WEB_SEARCH_PROVIDER` (`brave` / `serper` / `tavily`) and an API key (`LIBRARIAN_WEB_SEARCH_API_KEY`, or `BRAVE_API_KEY` / `SERPER_API_KEY` / `TAVILY_API_KEY`). Claim extraction uses Ollama’s OpenAI-compatible chat JSON when reachable (`LIBRARIAN_VERIFICATION__OLLAMA_BASE_URL`, `LIBRARIAN_VERIFICATION__CLAIM_MODEL`); otherwise text is split into sentence-like claims.

## LLM Fact Extraction (opt-in, changes content semantics)

The default ingestion path **splits** prose with LangChain's
`RecursiveCharacterTextSplitter` — the stored content is byte-equal to slices
of the source. Setting `ingest.use_llm_chunking: true` (env:
`LIBRARIAN_INGEST_USE_LLM_CHUNKING=true`) replaces that path with **LLM fact
extraction** (`Ingestor._extract_facts_llm`), which is **not** chunking:

- The prompt asks the model to "decompose into atomic factual statements".
  Stored content is **generated text**, not slices of the source, so wording
  and meaning may differ from the original.
- The source is **silently truncated** at `ingest.llm_extraction_max_chars`
  (default `8000`) before the prompt is built. Anything past that limit is
  discarded; large documents lose their tail unless the LLM call fails and
  the deterministic splitter is used as a fallback.
- The ingestor logs a `WARNING` on startup whenever this flag is enabled.

Default is `false`. Enable only when you explicitly want LLM-rewritten facts
and have a model you trust for paraphrase.

## Stack

| Component | Technology |
| --- | --- |
| MCP Framework | FastMCP (Python) |
| Database | MongoDB 7.x via `motor` (async) |
| Vector Search | Atlas Vector Search (cosine similarity) |
| Embedding (default) | `nomic-embed-text` (768 dims) via Ollama |
| Embedding (alt) | `sentence-transformers/all-MiniLM-L6-v2` (384 dims) |
| Sharding (default) | LangChain `RecursiveCharacterTextSplitter` (deterministic split) |
| Fact extraction (opt-in) | Ollama LLM, `ingest.use_llm_chunking=true` — see caveat below |
| Claim extraction | Ollama OpenAI-compatible JSON (optional) or heuristic sentences |
| Web search | Brave / Serper / Tavily (`build_web_search_client`) |
| HTML extraction | `trafilatura` |
| Config | Pydantic Settings + `librarian.config.yaml` |

## Breaking Change: Embedding Model (May 2026)

The default embedding model has moved from `all-MiniLM-L6-v2` (384 dimensions) to `nomic-embed-text` (768 dimensions). This provides better retrieval quality but **will cause dimension mismatch errors** if you have an existing MongoDB Atlas Vector Search index.

### Migration Options

1. **Re-ingest (Recommended):** Drop your `tomes` collection and re-ingest your data to take advantage of the better model.
1. **Maintain Compatibility:** If you want to keep your existing 384-dimensional data, set the following environment variables (or update `librarian.config.yaml`):
   ```bash
   LIBRARIAN_EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
   LIBRARIAN_EMBEDDING_DIMENSIONS=384
   LIBRARIAN_EMBEDDING_PROVIDER=sentence-transformers
   ```

## Quick Start

**1. Start infrastructure:**

```bash
docker compose up -d
docker compose exec ollama ollama pull nomic-embed-text
# Required for default LLM claim extraction (override via
# LIBRARIAN_VERIFICATION__CLAIM_MODEL / LIBRARIAN_INGEST__EXTRACTION_MODEL):
docker compose exec ollama ollama pull gemma2:2b
```

**2. Install Python deps:**

```bash
uv sync
# Optional: install sentence-transformers (~870MB, includes PyTorch)
uv sync --extra sentence-transformers
```

**3. Configure (optional):**

Copy the example config to the default location (the server looks for
`librarian.config.yaml` in the current working directory by default):

```bash
cp librarian.config.example.yaml librarian.config.yaml
# then edit as needed
```

Minimal config:

```yaml
# librarian.config.yaml
database:
  uri: mongodb://localhost:27017
embedding:
  provider: sentence-transformers   # or "dummy" for testing
server:
  host: 0.0.0.0
  port: 8000
```

To point at a different path, set `LIBRARIAN_CONFIG=/path/to/config.yaml`.
Environment variables override YAML using each section's prefix:

```bash
LIBRARIAN_DATABASE_URI=mongodb://localhost:27017
LIBRARIAN_EMBEDDING_PROVIDER=sentence-transformers
```

**4. Run:**

```bash
python -m src
# Server starts on http://localhost:8000
```

## Setup guides (Cursor, Claude, Gemini / Antigravity)

Step-by-step guides and helper scripts live under **[docs/setup/](docs/setup/README.md)**. Quick start:

```bash
chmod +x scripts/*.sh
./scripts/dev-up.sh              # config + docker stack + HTTP MCP snippets
./scripts/mcp-config-cursor.sh   # or the Claude / HTTP scripts listed in docs/setup
```

## MCP Client Configuration

The server supports three transports selectable via `--transport` flag or the `LIBRARIAN_SERVER_TRANSPORT` env var (or `server.transport` in `librarian.config.yaml`):

| Transport | When to use |
| --- | --- |
| `stdio` (default) | Local clients that spawn the process directly (Claude Code, Claude Desktop, Cursor) |
| `sse` | HTTP clients that connect to a running server (Gemini, Antigravity, remote setups) |
| `streamable-http` | Newer HTTP clients that prefer the streamable-http MCP transport |

### Claude Code / Claude Desktop

Stdio — Claude spawns the process directly:

```json
{
  "mcpServers": {
    "librarian": {
      "command": "uv",
      "args": ["run", "python", "-m", "src"],
      "cwd": "/path/to/librarian"
    }
  }
}
```

### Cursor

Add to `.cursor/mcp.json` (project) or `~/.cursor/mcp.json` (global):

```json
{
  "mcpServers": {
    "librarian": {
      "command": "uv",
      "args": ["run", "python", "-m", "src"],
      "cwd": "/path/to/librarian"
    }
  }
}
```

### Gemini CLI / Antigravity

Gemini CLI and Antigravity can connect via **`stdio`** (recommended) or **`sse`**.

**Stdio setup (Gemini CLI):**

```bash
gemini mcp add librarian uv -e LIBRARIAN_CONFIG=$(pwd)/librarian.config.yaml -- run python -m src
```

**SSE setup (HTTP-based):**

Start the server with SSE transport first:

```bash
python -m src --transport sse
# or: LIBRARIAN_SERVER_TRANSPORT=sse python -m src
# or: docker compose up librarian
```

Then point the client at the SSE endpoint:

```json
{
  "mcpServers": {
    "librarian": {
      "url": "http://localhost:8000/sse"
    }
  }
}
```

## Development

Install dev tools (pytest, pytest-asyncio, ruff, mypy, mdformat, pre-commit):

```bash
uv sync --extra dev
```

Then:

```bash
# Run tests (no external deps required)
uv run pytest

# Lint + format
uv run ruff check . && uv run ruff format .

# Type check
uv run mypy src

# Markdown formatting check
git ls-files -z '*.md' | xargs -0 -r uv run mdformat --check
```

Tests use in-memory stubs for all services — no MongoDB or embedding model needed for the unit test suite. Integration tests in `tests/test_mongo_repository.py` require a running MongoDB instance; when one is not reachable those tests skip automatically (a fast ping with `serverSelectionTimeoutMS=500` decides).

### Test environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `LIBRARIAN_TEST_MONGO_URI` | `mongodb://localhost:27017/?directConnection=true` | Mongo URI used by `tests/test_mongo_repository.py`. Set to point at a non-default instance, or leave unset and the live-Mongo tests will skip cleanly when no Mongo is running. |
