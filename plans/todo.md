# Implementation Todo List: The Librarian MCP Server

This document breaks down the design document into a comprehensive, step-by-step todo list. Each major phase is broken down into specific, actionable tasks sized appropriately for an L3 (junior) Software Engineer.

## Phase 1: Core Infrastructure

**Objective:** Set up the foundational project structure, database connection, and embedding utilities.

### P1.1: Project Scaffold & Data Models

- [x] **Initialize Project:** Create a new Python project. Set up a virtual environment (using `uv`, `poetry`, or `venv`) and initialize a `pyproject.toml` or `requirements.txt`.
- [x] **Install Dependencies:** Add core dependencies: `fastmcp`, `motor` (for async MongoDB), `sentence-transformers`, `pydantic`, and `pydantic-settings`.
- [x] **Define Data Models:** Create a `models.py` file. Using `pydantic.BaseModel`, implement the exact schemas defined in section 4 for `Tome` and `ResearchJob`. Ensure proper types (e.g., `datetime` for ISODate, `list[float]` for embeddings).
- [x] **Configuration Management:** Create a `config.py` file using `pydantic-settings`. Define a `Settings` class that matches the structure in section 8 (`librarian.config.yaml`). Ensure it can read from both a YAML file and environment variables.

### P1.2: MongoDB Setup & Indexes

- [x] **Docker Compose Setup:** Create a `docker-compose.yml` file with a `mongodb/mongodb-atlas-local` service to support vector search locally (refer to Appendix A).
- [x] **Database Connection:** Create a `db.py` module. Write an asynchronous function using `motor` to connect to MongoDB based on the URI in the settings and return the database instance.
- [x] **Index Creation Script:** ~~Write a standalone Python script (`scripts/setup_db.py`)~~ — implemented differently: `MongoTomeRepository.ensure_indexes()` creates the collection and every index automatically at server startup (invoked from the lifespan) and waits for the Atlas search indexes to become queryable. No standalone script exists or is needed.
- [x] **Apply Indexes:** Done via `ensure_indexes()`, with two deliberate deviations from section 4.2:
  - Vector Index on `embedding` (Atlas Search index `vectors`, cosine, dims from the configured embedding model, with a startup dimension-mismatch check). ✔
  - ~~Text Index on `title + content + tags`~~ — replaced by a dynamic-mapping Atlas lexical Search index (`default`) covering title/content/summary/tags.
  - ~~Compound Index on `{ category: 1, created_at: -1 }`~~ — separate standard indexes on `created_at` (desc) and `category` instead.
  - Standard Index on `research_job_id` in the `tomes` collection. ✔

### P1.3: Embedding Service

- [x] **Service Interface:** Create an `embeddings.py` module. Define an abstract base class `EmbeddingProvider` with an async method `embed_text(text: str) -> list[float]`.
- [x] **Ollama Provider:** Implement `OllamaEmbeddingProvider`. Use `httpx` or `aiohttp` to make an async call to a local Ollama instance (`/api/embeddings` endpoint) requesting the `nomic-embed-text` model.
- [x] **Sentence-Transformers Provider:** Implement `SentenceTransformerProvider` using the `sentence-transformers` library (e.g., `all-MiniLM-L6-v2`).
- [x] **LRU Cache:** Wrap the embedding calls with an in-memory Least Recently Used (LRU) cache. The cache key should be the SHA-256 hash of the input text, and the size should be configurable via settings.

______________________________________________________________________

## Phase 2: Search & Ingest

**Objective:** Implement the core tools for adding knowledge to the library and searching it.

### P2.1: Implement `library.search`

- [x] **FastMCP Tool Setup:** In `main.py`, initialize the FastMCP server and register a tool named `library.search`.
- [x] **Vector Search Query:** Implement the logic to take the user's `query`, get its vector embedding using the `EmbeddingService`, and perform an aggregation pipeline on the `tomes` collection using `$vectorSearch`.
- [x] **Apply Filters:** Update the aggregation pipeline to respect the `category` and `min_confidence` optional parameters.
- [x] **Format Output:** Map the MongoDB results to a dictionary matching the Output Schema in section 5.1, handling the `include_summary` flag (if true, omit the full `content`).

### P2.2: Implement `library.ingest` (Basic)

- [x] **FastMCP Tool Setup:** Register the `library.ingest` tool in `main.py` with the required input parameters.
- [x] **Text Chunking:** Integrate `langchain-text-splitters` (specifically `RecursiveCharacterTextSplitter`). Write a function to split the incoming `content` into chunks of roughly 400 words, respecting sentence boundaries.
- [x] **Auto-classification & Tagging:** Write a placeholder utility function that assigns a default `category` and extracts simple tags from the text (this can be improved later with NLP/LLM extraction).
- [x] **Deduplication Logic:** For each chunk, generate an embedding. Query MongoDB for existing tomes with a cosine similarity > 0.95. If found, log a skip/merge message.
- [x] **Storage:** Construct `Tome` Pydantic models for the new chunks and insert them into the `tomes` collection. Return the required output schema.

### P2.3: Verification Pipeline

- [x] **Claim Extraction:** Create a prompt and utility function that uses a local LLM (via Ollama or an external API) to extract 3–7 key factual claims from a text chunk.
- [x] **Web Search Verification:** Write logic to take each extracted claim, search the web (using a dummy client or actual Brave API if available), and score the claim as `supported`, `contradicted`, or `unverifiable`.
- [x] **Confidence Calculation:** Compute an aggregate confidence score (0.0 - 1.0) based on the claim scores.
- [x] **Integrate Verifier:** Wire the Verifier into the `library.ingest` flow before chunking/embedding. Reject content if confidence < 0.3. Add an offline bypass mode that assigns a `0.6` score.

______________________________________________________________________

## Phase 3: Researcher

**Objective:** Build the autonomous agent component that fetches new knowledge from the web.

### P3.1: Search API Client

- [x] **Client Interface:** Create a `web_search.py` module with an abstract `WebSearchClient`.
- [x] **Brave Search Implementation:** Implement a concrete client for the Brave Search API. It should take a query and return a list of URLs and text snippets. Ensure it handles rate limits and API errors gracefully.
- [x] **Content Extraction:** Integrate the `trafilatura` library. Write an async function that takes a URL, downloads the HTML, and uses `trafilatura` to extract the main readable text, stripping out navigation and ads.

### P3.2: Implement `library.research` (Synchronous Flow)

- [x] **FastMCP Tool Setup:** Register the `library.research` tool in `main.py`.
- [x] **Query Planning:** Write a function that uses an LLM to generate 3–6 focused search queries based on the provided `topic` and `context`.
- [x] **Web Fetching Pipeline:** For each query, call the `WebSearchClient`, de-duplicate the resulting URLs, and use the content extraction function to fetch the text for the top N pages.
- [x] **Synthesis:** Write an LLM prompt/function to merge the extracted texts, identifying consensus facts and structuring them into a coherent summary.
- [x] **Ingest Integration:** Pass the synthesized text directly into the `library.ingest` pipeline. Return the resulting `tome_ids` and summaries.

### P3.3: Async Research Jobs

- [x] **Job Tracking:** Implement the logic to create a `ResearchJob` document in MongoDB with a `pending` status when `library.research` is called.
- [x] **Background Execution:** Modify the `library.research` endpoint to accept an `async` flag. If true, use `asyncio.create_task()` (or a simple background worker) to run the research flow without blocking.
- [x] **State Updates:** Ensure the background task updates the `ResearchJob` document's `status` to `running`, and then to `completed` or `failed` at the end, populating the `tome_ids` or `error` fields accordingly.

______________________________________________________________________

## Phase 4: Polish & Testing

**Objective:** Ensure the system is robust, thoroughly tested, and easy to deploy.

### P4.1: Unit & Integration Tests

- [x] **Embedding Tests:** Write `pytest` unit tests for the LRU cache and the embedding providers (mocking the HTTP calls).
- [x] **Ingest Chunking Tests:** Write unit tests verifying that long text is correctly split into \<400 word chunks without breaking sentences.
- [x] **Search Integration Tests:** Write `pytest-asyncio` tests that insert known `Tome` records into a local test database and verify that `library.search` returns the correct semantic matches.

### P4.2: Migration Utility (still open)

- [ ] **CLI Setup:** Create a command-line script (e.g., using the `argparse` or `typer` library) with a command like `python -m scripts.migrate_index`.
- [ ] **Migration Logic:** Implement logic that iterates over all documents in the `tomes` collection, re-calculates their embeddings using the currently configured `EmbeddingService`, and updates the documents. This is necessary for when a user changes the embedding model.

> Status: not built. Startup now fails fast when the existing Atlas vector
> index's dimensions disagree with the configured model, and the README
> documents the interim migration paths (drop + re-ingest, or pin the old
> model via config). This utility remains the missing piece for in-place
> migration.

### P4.3: Final Documentation & Deployment

- [x] **Complete Docker Compose:** Finalize the `docker-compose.yml` adding the `librarian` service itself, ensuring it connects properly to `mongo` and `ollama`.
- [x] **README:** Write a comprehensive `README.md` containing:
  - What The Librarian is.
  - Quick-start instructions (using Docker Compose).
  - The full tool surface (grew from the 3 planned tools to 8: search, ingest, research, get, update, delete, tidy, list).
  - Configuration environment variables.
