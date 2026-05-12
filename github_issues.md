# Comparison and GitHub Issues Proposal

This document compares the initial independent `gap_analysis.md` with the provided `ARCHITECTURE_AUDIT.md` and synthesizes the findings into actionable GitHub Issues.

## 1. Comparison of Findings

### Areas of Alignment

- **Ingestor Resharding Failure (Data Integrity):** Both reports identified that the ingestor swallows deletion errors during resharding, leading to potential data duplication and index drift.
- **Test Suite Deficiencies:** Both noted the presence of `test_placeholder.py` and the lack of robust end-to-end or fault-injection testing.

### Findings Unique to `gap_analysis.md`

- **Unmocked MongoDB Tests:** Specifically identified that `test_mongo_repository.py` fails fatally without a live MongoDB instance on port 27017, rather than skipping or mocking.
- **Dependency Management Issues:** Highlighted the fragmented use of `[project.optional-dependencies]` vs `[dependency-groups]` in `pyproject.toml` breaking `uv sync`.
- **Pydantic V1 Deprecation:** Caught the `langchain_core` deprecation warning regarding Pydantic V1 incompatibility with Python 3.14+.
- **Missing Error Taxonomy:** Noted the lack of a unified `StorageError` abstraction in the storage layer.

### Findings Unique to `ARCHITECTURE_AUDIT.md`

The audit provided a much deeper dive into runtime behavior, concurrency, and security, revealing significant flaws:

- **Severity High (🔴):**
  - Module-level config loading crashes the app at import time (A1).
  - The Verifier's heuristic is severely biased towards "supported" because it just checks for word overlap with search snippets (D1).
  - The filesystem backend does an O(N) brute-force load on every search/ingest (I1).
- **Concurrency & Performance (🟡):**
  - `trafilatura.extract` is blocking the async event loop (E1).
  - Web searches run sequentially instead of in parallel (E2).
  - Background tasks are cancelled during shutdown but never awaited, risking half-written state (A3, L2).
- **Data Integrity & DB (🟡):**
  - LLM "chunking" hallucinates/rewrites text instead of chunking it, and truncates at 8000 chars (C1, C2).
  - Vector embedding dimensions aren't validated against the DB config (F2).
  - Silent swallows on MongoDB index creation and filesystem read failures (H1, I4).
  - Two Motor clients instantiated instead of one (H2).
- **Security & Config (🟡/🔵):**
  - No SSRF guard on web fetch (M2).
  - No content-size enforcement per shard (M1).
  - Custom `.env` parser is buggy; should use `python-dotenv` (B2).
  - Default model is `gemma4:e2b` (likely a typo) (B5).

______________________________________________________________________

## 2. Proposed GitHub Issues Framework

The findings have been categorized into actionable epics/issues.

### Category 1: Critical Correctness & Data Integrity

**Issue 1.1: Fix Verifier heuristic bias**

- **Description:** The current verification heuristic simply checks for word overlap between the claim and the search snippet. Since the search snippet usually contains the claim words, almost everything is marked as "SUPPORTED".
- **Action:** Replace the overlap heuristic with a real LLM evaluation prompt, or at minimum, adjust the logic to require semantic agreement, not just keyword overlap.

**Issue 1.2: Fix LLM "chunking" hallucination and truncation**

- **Description:** The LLM prompt asks to "decompose text into atomic factual statements," causing it to rewrite the source material instead of chunking it. It also silently truncates inputs over 8000 characters.
- **Action:** Rename the feature to `extract_facts` and document the behavior. For actual chunking, default to the `RecursiveCharacterTextSplitter`.

**Issue 1.3: Prevent data duplication during ingestor reshard failures**

- **Description:** If `_dedup_and_store` fails to delete old duplicates after inserting replacements, it swallows the error, leading to duplicate data in the index.
- **Action:** Implement `return_exceptions=True` in the deletion gather and rollback the new inserts if deletion fails, or use a distributed transaction if supported.

**Issue 1.4: Validate embedding dimensions against Database configuration**

- **Description:** If the ST/Ollama model outputs 768 dimensions but the config defaults to 384, MongoDB silent-fails on vector insertion.
- **Action:** Dynamically read the dimension size from the `EmbeddingService` and use it to configure the MongoDB Atlas index. Fail fast if they mismatch.

### Category 2: Server Architecture & Concurrency

**Issue 2.1: Defer configuration loading until application startup (Remove module-level config)**

- **Description:** `server.py` loads the config at import time. A missing env var crashes the process before FastMCP can boot or handle errors.
- **Action:** Move configuration parsing into the `lifespan` context manager or the `__main__.py` entry point.

**Issue 2.2: Fix event loop blocking in Researcher**

- **Description:** `trafilatura.extract(html)` and `splitter.split_text()` are sync CPU-bound tasks running directly on the asyncio event loop, stalling the server.
- **Action:** Wrap these calls in `asyncio.to_thread()`.

**Issue 2.3: Parallelize web searches**

- **Description:** The researcher issues web queries sequentially.
- **Action:** Use `asyncio.gather` with a bounded semaphore to run searches in parallel.

**Issue 2.4: Graceful shutdown of background tasks**

- **Description:** `LibrarianServer.lifespan` cancels background tasks but does not await them, potentially leaving MongoDB writes in a corrupted half-state.
- **Action:** Add `await asyncio.gather(*tasks, return_exceptions=True)` during shutdown.

### Category 3: Database & Storage Reliability

**Issue 3.1: Share a single `AsyncIOMotorClient`**

- **Description:** `MongoTomeRepository` and `MongoResearchJobRepository` instantiate their own Motor clients, doubling the connection pool.
- **Action:** Instantiate the client at the server/DI level and pass it to both repositories. Add a `serverSelectionTimeoutMS=5000` to prevent indefinite hangs.

**Issue 3.2: Surface MongoDB index creation failures**

- **Description:** `ensure_indexes` swallows index creation errors, resulting in a server that boots but cannot search.
- **Action:** Log the error as critical and potentially fail startup or surface via a health check endpoint.

**Issue 3.3: Rethink the Filesystem backend for scale**

- **Description:** The FS backend does an O(N) read of every file on every search/ingest, making it unusable beyond a trivial number of tomes.
- **Action:** Document FS as a test-only backend, or implement an in-memory index that persists to a single JSON file.

### Category 4: Security, Config & Dependencies

**Issue 4.1: Remove buggy custom `.env` parser**

- **Description:** `config.py` contains a custom dotenv parser with bugs regarding quote handling.
- **Action:** Remove it and rely on `python-dotenv` (already a transitive dependency).

**Issue 4.2: Implement SSRF guards and payload size limits**

- **Description:** The researcher will fetch arbitrary URLs (including localhost/internal IPs) and the ingestor accepts arbitrarily large shard arrays.
- **Action:** Implement an IP blocklist for the web fetcher, and cap the maximum number of shards processed in a single ingest call.

**Issue 4.3: Fix local Dev Environment and Testing Setup**

- **Description:** `uv sync` fails to install dev tools due to split `[project.optional-dependencies]` and `[dependency-groups]`. Tests fail without a live Mongo instance.
- **Action:** Consolidate dev dependencies into PEP-735 groups. Add `pytest.mark.skipif` to Mongo tests if no connection is available. Remove `test_placeholder.py`.

**Issue 4.4: Fix missing Ollama pull in docker-compose**

- **Description:** A fresh `docker-compose up` fails because the `nomic-embed-text` model isn't pulled automatically inside the Ollama container.
- **Action:** Add an init container or a startup script to the compose file to run `ollama pull nomic-embed-text`.

### Category 5: Usefulness & Data Lifecycle (New Use Cases)

*This category focuses on gaps that prevent Librarian from being useful in scenarios like an **AI Coding Assistant Knowledge Base**, a **Long-Term Personal Memory (CRM)**, or an **Autonomous Support Agent**.*

**Issue 5.1: No Explicit Data Update/Delete/Supersede Mechanisms**

- **Description:** Agents have no MCP tools to explicitly say "Delete Tome X" or "Update Fact Y". The `_dedup_and_store` resharding is the *only* update mechanism, and it relies entirely on high vector similarity (near duplicates). If a fact fundamentally changes (e.g., "The API is REST" -> "The API is GraphQL"), the vectors might be far apart, resulting in *both* facts coexisting. Long-term memory gets polluted with contradictory facts.
- **Action:** Add `library_update` and `library_delete` MCP tools. Consider adding an explicit `supersedes` parameter to `library_ingest`.

**Issue 5.2: Lack of Temporal Resolution / Recency Weighting**

- **Description:** `Tome` has `created_at`, but `library_search` returns results sorted purely by RRF (Reciprocal Rank Fusion) of vector and lexical scores. It does not prioritize newer facts if they conflict with older facts.
- **Action:** Add an `updated_at` field and allow agents to pass a `recency_bias` parameter to `library_search` to automatically uprank recent information when resolving conflicts.

**Issue 5.3: Blind Metadata and "Write-Only" Tagging**

- **Description:** Agents can pass `category` and `tags` to `library_ingest`, and can filter by `category` in `library_search`. However, there is no tool to list available categories or tags. The agent is effectively guessing string literals for categories, rendering the tagging system nearly useless for structured exploration. Furthermore, `library_search` doesn't even accept `tags` as a filter in its schema.
- **Action:** Implement a `library_list_metadata` tool that returns all distinct categories and tags currently in the repository. Add tag filtering to `library_search`.

**Issue 5.4: Code and Syntax Destruction During Chunking**

- **Description:** `RecursiveCharacterTextSplitter` and the LLM "chunker" both destroy the syntactic integrity of code blocks. A 100-line Python script or YAML config will be arbitrarily sliced or hallucinated into "atomic facts," rendering the code useless for copy-pasting by a Coding Assistant.
- **Action:** Implement a `MarkdownSplitter` or syntax-aware chunker that preserves code fences and formatting when `source_type` implies structured data.

**Issue 5.5: Context Window Saturation (Over-fetching)**

- **Description:** `SearchOutput` returns the full `Tome` object, including `content` (which can be up to 400 words). If an agent queries with `top_k=20`, this can flood the context window with ~8000 words. While `SearchInput` has an `include_summary` flag, the `SearchOutput` schema unconditionally returns full Tomes.
- **Action:** Enforce the `include_summary` parameter in `library_search`: when `True` (or by default to save tokens), omit the full `content` field from the returned JSON and only return the `summary` to save context space, requiring the agent to request specific Tomes by ID if it wants the full content.

**Issue 5.6: Missing Provenance / Threading for Conversational Memory**

- **Description:** When ingesting agent input (`source_type=AGENT_INPUT`), there's no way to link multiple tomes together to represent a contiguous "conversation" or "document". They are treated as isolated facts.
- **Action:** Add a `context_id` or `thread_id` to the Tome model to allow retrieving facts grouped by their original ingest session or source conversation.
