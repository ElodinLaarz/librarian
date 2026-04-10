# The Librarian

### MCP Server Design Document

*An Intelligent Knowledge Management MCP*

| | |
| ------------ | ---------------------------- |
| **Version** | 1.0 |
| **Date** | March 2026 |
| **Protocol** | Model Context Protocol (MCP) |
| **Status** | Implemented |

______________________________________________________________________

## Table of Contents

1. [Executive Summary](#1-executive-summary)
1. [Goals & Non-Goals](#2-goals--non-goals)
1. [System Architecture](#3-system-architecture)
1. [Data Model](#4-data-model)
1. [Tool Specifications](#5-tool-specifications)
1. [Embedding Strategy](#6-embedding-strategy)
1. [Verification Pipeline](#7-verification-pipeline)
1. [Configuration Reference](#8-configuration-reference)
1. [Implementation Plan](#9-implementation-plan)
1. [Technology Stack](#10-technology-stack)
1. [Open Questions & Future Work](#11-open-questions--future-work)

- [Appendix A: Quick-Start (Docker Compose)](#appendix-a-quick-start-docker-compose)

______________________________________________________________________

## 1. Executive Summary

The Librarian is an MCP (Model Context Protocol) server that provides AI agents with a persistent, searchable, and self-expanding knowledge base. Rather than relying solely on training data or ad-hoc web searches, agents equipped with The Librarian gain access to a curated store of verified, bite-sized knowledge documents called **Tomes** — each focused on a single topic or fact, semantically indexed and retrievable at query time.

The system exposes three primary tools to any MCP-compatible agent:

| Tool | Purpose | Returns |
| ------------------ | -------------------------------------------------------------------------- | --------------------------------- |
| `library.search` | Retrieve relevant Tomes using semantic vector search | Ranked list of Tome documents |
| `library.ingest` | Feed new knowledge into the library; validates, chunks, and stores it | Ingested Tome IDs and summary |
| `library.research` | Dispatch a Researcher agent to collect information on a topic from the web | Newly created Tomes for the topic |

The underlying store is MongoDB (with Atlas-compatible Vector Search), paired with a locally running embedding model for offline, low-latency semantic similarity. Internet connectivity is used optionally during ingestion (for truth-checking) and during research dispatches.

______________________________________________________________________

## 2. Goals & Non-Goals

### 2.1 Goals

- Provide agents with persistent, long-term memory that survives session boundaries.
- Enable semantic (meaning-based) retrieval — not just keyword matching.
- Enforce knowledge quality by checking ingested claims against external sources before storage.
- Keep documents (Tomes) small and single-purpose so retrieval stays precise.
- Allow autonomous knowledge expansion: if the library lacks information, a Researcher sub-agent can go find it.
- Remain embeddable in any MCP-compatible environment with minimal configuration.
- Support fully offline operation for search and retrieval (internet only required for research/truth-checking).

### 2.2 Non-Goals

- The Librarian is not a general-purpose database — it stores knowledge, not structured application data.
- It is not a web browser or scraping engine; the Researcher uses internet search APIs rather than arbitrary browsing.
- It does not perform reasoning or inference — it provides raw knowledge to agents who do the reasoning.
- It is not a replacement for an agent's context window — it is a retrieval aid, not a memory dump.
- Real-time news feeds or streaming data are out of scope for v1.

______________________________________________________________________

## 3. System Architecture

The Librarian is structured as three loosely coupled layers: the MCP Interface Layer, the Librarian Core (business logic), and the Storage & Embedding Layer. External agents interact only through the MCP interface; all knowledge management happens internally.

```
MCP CLIENT (Agent)
| library.search  |  library.ingest  |  library.research |
                          v
+---------------------------------------------------+
|            MCP INTERFACE LAYER (FastMCP)           |
|   Tool Router  |  Input Validation  |  Auth         |
+---------------------------------------------------+
        v                    v                    v
+------------+  +------------+  +--------------------+
| SEARCH     |  | INGESTOR   |  | RESEARCHER         |
| ENGINE     |  | + VERIFIER |  | (sub-agent)        |
+------------+  +------------+  +--------------------+
        v                    v                    v
+---------------------------------------------------+
|  STORAGE & EMBEDDING LAYER                        |
|  MongoDB (Tomes collection)  |  Local Embeddings   |
|  Vector Search Index         |  Embedding Cache    |
+---------------------------------------------------+
```

### 3.1 Component Descriptions

**MCP Interface Layer** — Built with FastMCP (Python), this layer handles all communication with MCP clients. It registers the three tools, validates inputs, enforces rate limits, and routes requests to the appropriate internal service.

**Search Engine** — Takes a natural-language query, converts it to a vector embedding via the local embedding model, and performs a vector similarity search against the MongoDB Tomes collection. Results are re-ranked by a combination of vector similarity score and recency.

**Ingestor + Verifier** — Receives raw text or structured knowledge from the agent. Processes in four stages: (1) truthiness checking via internet search cross-reference, (2) chunking into small single-concept Tomes, (3) embedding generation, and (4) storage in MongoDB with full metadata.

**Researcher (Sub-Agent)** — A lightweight autonomous agent that accepts a topic and optional context string. It constructs targeted search queries, retrieves web results, synthesises findings, and pipes them through the Ingestor. Returns IDs and summaries of newly created Tomes.

**Storage & Embedding Layer** — MongoDB stores all Tomes as structured BSON documents. A dedicated vector index enables ANN search. The local embedding model (default: `nomic-embed-text` via Ollama, or `sentence-transformers/all-MiniLM-L6-v2` for pure Python) runs in-process with an LRU embedding cache.

______________________________________________________________________

## 4. Data Model

### 4.1 Tome Document Schema

Every piece of knowledge stored in The Librarian is a **Tome** — a compact, single-topic document stored in the `tomes` collection. Tomes are intentionally small (100–400 words) to keep retrieval precise.

| Field | Type | Required | Description |
| --------------- | ---------- | -------- | ----------------------------------------------------- |
| `_id` | ObjectId | Yes | MongoDB auto-generated unique identifier |
| `title` | string | Yes | Short descriptive title (max 120 chars) |
| `content` | string | Yes | Full text body (100–400 words recommended) |
| `summary` | string | Yes | One-to-two sentence summary for quick scanning |
| `category` | string | Yes | High-level domain category (e.g. `science`, `code`) |
| `tags` | string[] | No | Freeform topic tags for keyword filtering |
| `source_url` | string | No | Origin URL if sourced from the web |
| `source_type` | string | Yes | One of: `agent_input`, `researcher`, `manual` |
| `confidence` | number | Yes | Truthiness confidence score 0.0–1.0 from the Verifier |
| `embedding` | number[] | Yes | Dense vector embedding of the content (e.g. 768 dims) |
| `created_at` | ISODate | Yes | Timestamp when the Tome was first created |
| `updated_at` | ISODate | Yes | Timestamp of last modification |
| `version` | number | Yes | Incremented each time content is updated |
| `superseded_by` | ObjectId | No | Points to successor if this Tome was replaced |
| `research_job` | string | No | ID of the Researcher job that produced this Tome |

### 4.2 MongoDB Indexes

- **Vector Index** on `embedding` — ANN search via cosine similarity (primary retrieval mechanism).
- **Text Index** on `title + content + tags` — enables BM25 keyword pre-filtering to narrow the ANN candidate pool.
- **Compound Index** on `{ category: 1, created_at: -1 }` — supports category-scoped searches and time-sorted queries.
- **Index** on `research_job` — fast lookup of all Tomes from a given research run.

### 4.3 ResearchJob Document Schema

Tracks the state and output of each Researcher dispatch, stored in the `research_jobs` collection.

| Field | Type | Required | Description |
| ------------- | ------------ | -------- | --------------------------------------------------- |
| `_id` | ObjectId | Yes | Unique job identifier |
| `topic` | string | Yes | The topic string passed to the Researcher |
| `context` | string | No | Optional agent-supplied context |
| `status` | string | Yes | One of: `pending`, `running`, `completed`, `failed` |
| `queries` | string[] | No | Search queries issued during the run |
| `tome_ids` | ObjectId[] | No | IDs of Tomes created by this job |
| `error` | string | No | Error message if status is `failed` |
| `started_at` | ISODate | Yes | When the job began executing |
| `finished_at` | ISODate | No | When the job completed or failed |

______________________________________________________________________

## 5. Tool Specifications

### 5.1 `library.search` — Search the Library

Converts a natural-language query to a vector embedding and retrieves the most semantically relevant Tomes. This should be the agent's first call whenever it needs factual information.

> **Note:** If zero results are returned with confidence > 0.5, the agent should consider calling `library.research` to populate the library on that topic.

**Input Parameters**

| Parameter | Type | Description |
| ----------------- | ------- | ------------------------------------------------------------------ |
| `query` | string | Natural language question or topic (required). Max 2000 chars. |
| `top_k` | number | Number of results to return. Default 5, max 20. |
| `category` | string | Optional: restrict results to a specific category. |
| `min_confidence` | number | Minimum confidence score. Default 0.5. |
| `include_summary` | boolean | Return only summary field rather than full content. Default false. |

**Output Schema**

| Field | Type | Description |
| ------------ | ---------- | -------------------------------------------------------------- |
| `tomes` | Tome[] | Matching Tome documents, sorted by similarity score descending |
| `scores` | number[] | Cosine similarity scores corresponding to each Tome |
| `query_id` | string | Unique identifier for this search request |
| `from_cache` | boolean | Whether the result was served from the embedding cache |

**Processing Flow**

1. **Embed Query** — Run the query string through the local embedding model to produce a dense vector.
1. **Vector Search** — Query MongoDB using `$vectorSearch`. Apply optional category and confidence pre-filters.
1. **Re-rank** — Optionally boost results with recent `updated_at` timestamps or higher confidence scores.
1. **Return** — Serialize matched Tome objects (full or summary-only) alongside similarity scores.

______________________________________________________________________

### 5.2 `library.ingest` — Ingest Knowledge

Accepts a raw knowledge payload, validates its truthfulness against web sources, decomposes it into one or more focused Tomes, embeds each, and persists them. Call this whenever the agent learns something new that should be remembered.

> **Warning:** Content that fails verification (confidence < 0.3) will be rejected by default. Set `skip_verify: true` to bypass — useful for agent-generated internal notes or fictional world-building where web truth-checking is not applicable.

**Input Parameters**

| Parameter | Type | Description |
| -------------- | ---------- | ------------------------------------------------------------------------------------ |
| `content` | string | The raw knowledge text to ingest (required). Chunked at ~400 words. |
| `title` | string | Optional title (auto-generated if omitted). |
| `category` | string | Domain category hint. Auto-classified if omitted. |
| `tags` | string[] | Optional topic tags. Merged with auto-detected tags. |
| `source_url` | string | URL where this content originated, if known. |
| `skip_verify` | boolean | Bypass the Verifier step. Default false. |
| `allow_update` | boolean | Update an existing very-similar Tome rather than creating a duplicate. Default true. |

**Output Schema**

| Field | Type | Description |
| --------------- | ---------- | ------------------------------------------ |
| `tome_ids` | string[] | IDs of all Tomes created or updated |
| `tomes` | Tome[] | Full Tome objects created during this call |
| `confidence` | number | Overall verification confidence (0.0–1.0) |
| `chunks` | number | Number of Tomes the input was split into |
| `status` | string | One of: `stored`, `rejected`, `partial` |
| `reject_reason` | string | Explanation if status is `rejected` |

**Processing Flow**

1. **Pre-flight Check** — Validate input length and required fields. Sanitise HTML/markdown if present.
1. **Verify** — Extract key factual claims. Search the web for corroborating or conflicting sources. Compute confidence score.
1. **Chunk** — Split content into single-topic chunks of ~400 words using sentence-boundary splitting. Assign a generated title and summary to each.
1. **Classify & Tag** — Auto-classify into a category if not provided. Extract topic tags using a lightweight keyword extractor.
1. **Embed** — Generate a dense vector embedding for each chunk.
1. **Dedup & Store** — Search for existing Tomes with high cosine similarity (> 0.95). Merge or skip duplicates. Write to MongoDB.

______________________________________________________________________

### 5.3 `library.research` — Research a Topic

Dispatches a Researcher sub-agent to independently search the web, synthesise findings, and ingest the results as new Tomes. Use this when the library lacks information or the returned Tomes are insufficient.

> **Note:** When `async: true`, the agent should poll `library.research` with the returned `job_id` to check status. The library can still be searched normally while a research job is running.

**Input Parameters**

| Parameter | Type | Description |
| ----------- | ------- | ------------------------------------------------------------------------------------ |
| `topic` | string | Core subject to research (required). Be specific. |
| `context` | string | Optional context about what the agent needs to know. |
| `depth` | string | One of: `shallow` (3 sources), `standard` (6 sources, default), `deep` (12 sources). |
| `max_tomes` | number | Maximum Tomes to create per run. Default 10. |
| `category` | string | Optional category to assign to all produced Tomes. |
| `async` | boolean | Return a `job_id` immediately without waiting. Default false. |

**Output Schema**

| Field | Type | Description |
| ------------- | ---------- | --------------------------------------------------- |
| `job_id` | string | ID of the ResearchJob record created |
| `tome_ids` | string[] | IDs of all Tomes created |
| `tomes` | Tome[] | Full Tome objects produced, ready for immediate use |
| `sources` | string[] | URLs of web pages consulted |
| `query_count` | number | Number of individual search queries issued |
| `status` | string | `completed` or `failed` |

**Researcher Sub-Agent Flow**

1. **Query Planning** — Generate 3–6 focused search queries from the topic and context. Prioritise different facets (definition, examples, caveats, recent developments).
1. **Web Search** — Issue queries to the configured search API (Brave / Serper / Tavily). Collect URLs and snippets. De-duplicate by domain.
1. **Content Extraction** — Fetch and parse HTML from top-ranked URLs. Extract main body text, stripping navigation and boilerplate.
1. **Synthesis** — Merge content across sources. Identify consensus facts, note disagreements, structure into logical sub-topics.
1. **Ingest** — Pipe synthesised content through the standard `library.ingest` pipeline: verify, chunk, classify, embed, store.
1. **Job Update** — Mark the ResearchJob as completed. Return all produced Tomes and metadata.

______________________________________________________________________

## 6. Embedding Strategy

| Model | Dimensions | Mode | Best For | Latency (CPU) |
| ------------------------------------------- | ---------- | -------------- | -------------------------------------------------- | ------------- |
| `nomic-embed-text` (Ollama) | 768 | Local (Ollama) | Balanced quality & speed; recommended default | ~30ms |
| `all-MiniLM-L6-v2` (sentence-transformers) | 384 | Local (Python) | Pure Python; no Ollama required; lighter | ~15ms |
| `all-mpnet-base-v2` (sentence-transformers) | 768 | Local (Python) | Higher quality; slower; better for dense knowledge | ~60ms |

### 6.1 Embedding Cache

To avoid re-embedding identical or near-identical texts, The Librarian maintains an in-memory LRU cache keyed on SHA-256 hash of the input text. Cache size defaults to 10,000 entries and is configurable. On restart, the cache is cold but rebuilds quickly from query traffic.

### 6.2 Dimensionality & Index Configuration

The MongoDB vector index must be configured to match the chosen model's output dimensions. The Librarian checks this at startup and will refuse to start if there is a mismatch. A migration utility (`librarian migrate-index`) is provided to re-embed the full Tomes collection when switching models.

______________________________________________________________________

## 7. Verification Pipeline

The Verifier is the quality control layer — its job is to estimate the truthfulness of incoming content before storage, preventing the propagation of hallucinated or incorrect information across sessions.

### 7.1 Verification Algorithm

1. Extract 3–7 key factual claims from the content using a zero-shot claim extraction prompt.
1. For each claim, construct a targeted search query and retrieve the top 3 web results.
1. Score each claim as `supported`, `contradicted`, or `unverifiable` based on snippet analysis.
1. Compute an aggregate confidence score: supported claims add weight, contradicted claims subtract, unverifiable are neutral.
1. If confidence > 0.7: store as-is. Between 0.3–0.7: store with a low-confidence flag. Below 0.3: reject with contradicting evidence.

### 7.2 Offline Mode

When configured without internet access, the Verifier is automatically skipped and all content is assigned a synthetic confidence score of `0.6`. A metadata flag `offline_verified: true` is stored on each Tome. Agents can filter these out when precision is critical.

> **Note:** The Verifier uses the same search API as the Researcher. If no API key is configured, verification is silently skipped with a 0.6 score and a note in the Tome metadata.

______________________________________________________________________

## 8. Configuration Reference

The Librarian is configured via `librarian.config.yaml` or environment variables. Environment variables take precedence.

```yaml
# librarian.config.yaml

database:
  uri: mongodb://localhost:27017          # or Atlas connection string
  database: librarian
  tomes_collection: tomes
  jobs_collection: research_jobs

embedding:
  provider: ollama                        # ollama | sentence-transformers | dummy
  model_name: nomic-embed-text            # model name per provider
  dimensions: 768                         # must match vector index
  cache_size: 10000                       # LRU cache entries

search:
  default_top_k: 5
  max_top_k: 20
  min_confidence: 0.5
  use_keyword_prefilter: true             # BM25 pre-filter before ANN

verification:
  enabled: true
  reject_threshold: 0.3
  store_threshold: 0.7

web_search:
  provider: brave                         # brave | serper | tavily
  api_key: ${LIBRARIAN_WEB_SEARCH_API_KEY}

researcher:
  shallow_queries: 3
  standard_queries: 5
  deep_queries: 8

server:
  host: 0.0.0.0
  port: 8000
  log_level: info
```

______________________________________________________________________

## 9. Implementation Plan

### Phase 1 — Core Infrastructure

- **P1.1 Project Scaffold** — Set up Python project with FastMCP, motor (async MongoDB driver), and sentence-transformers. Define Tome Pydantic model.
- **P1.2 MongoDB Setup** — Provision local MongoDB with Atlas-compatible vector index. Write index creation script and migration utility.
- **P1.3 Embedding Service** — Implement `EmbeddingService` abstraction with drivers for Ollama and sentence-transformers. Add LRU cache.

### Phase 2 — Search & Ingest

- **P2.1 `library.search`** — Implement vector search with optional category filter and keyword pre-filter. Wire into MCP tool.
- **P2.2 `library.ingest` (basic)** — Implement chunking, auto-tagging, embedding, dedup, and storage. Skip verification initially.
- **P2.3 Verifier** — Implement claim extraction and web search scoring. Integrate into ingest pipeline.

### Phase 3 — Researcher

- **P3.1 Search API Client** — Abstract Brave/Serper/Tavily behind a common interface. Implement rate limiting.
- **P3.2 `library.research` (sync)** — Implement full Researcher flow: query plan, search, extract, synthesise, ingest.
- **P3.3 Async Research Jobs** — Add `async: true` support with background task queue, job polling, and ResearchJob lifecycle.

### Phase 4 — Polish & Testing

- Integration tests for all three MCP tools using a local MongoDB test instance.
- Load testing: measure latency under 1,000 and 10,000 Tomes in the collection.
- Embedding model swap test: verify `migrate-index` utility preserves all Tomes correctly.
- Write comprehensive README with Docker Compose setup for zero-config local deployment.
- Publish as an installable MCP server to the MCP registry.

______________________________________________________________________

## 10. Technology Stack

| Component | Technology | Rationale |
| ------------------- | ----------------------------------- | ------------------------------------------------------------------ |
| MCP Framework | FastMCP (Python) | Native Python MCP server framework; minimal boilerplate |
| Database | MongoDB 7.x (motor async driver) | Flexible document model; built-in vector search via Atlas |
| Vector Index | Atlas Vector Search / mongot | ANN search on embedding field; cosine similarity |
| Embedding (default) | nomic-embed-text via Ollama | High-quality, locally run, zero API-cost embeddings |
| Embedding (alt) | sentence-transformers (HuggingFace) | Pure Python fallback; no Ollama daemon required |
| Chunking | LangChain RecursiveTextSplitter | Battle-tested sentence-boundary chunking with configurable overlap |
| Claim Extraction | Instructor + local LLM (Ollama) | Structured output for claim extraction during verification |
| Web Search | Brave Search API / Tavily / Serper | Configurable; Brave preferred for privacy & cost |
| HTML Extraction | trafilatura | High-quality boilerplate removal for web content |
| Config Management | Pydantic Settings + YAML | Type-safe config with env-var override support |
| Containerisation | Docker + Docker Compose | One-command local setup including MongoDB and Ollama |
| Testing | pytest + pytest-asyncio | Async-native testing for motor and FastMCP interactions |

______________________________________________________________________

## 11. Open Questions & Future Work

### 11.1 Open Questions

- **Multi-tenancy:** Should the Librarian support separate knowledge namespaces per user or agent, or is a single shared library the right default?
- **Tome expiry:** Should Tomes have a configurable TTL? Knowledge can become stale, especially in fast-moving domains.
- **Conflict resolution:** When two contradictory Tomes exist, how should search resolve the conflict? Return both and let the agent decide, or flag conflicts explicitly?
- **Cross-library federation:** Should The Librarian support querying against multiple library instances in a single search call?
- **Security model:** Should MCP tool calls require an API key? How should multi-agent environments handle write permissions to the shared library?

### 11.2 Future Work (v2+)

- Webhook callbacks for async research job completion.
- Knowledge graph layer: extract entity relationships from Tomes and store them in a graph for structured traversal queries.
- Active forgetting: identify low-confidence or low-retrieval Tomes and schedule them for re-verification or deletion.
- Tome versioning UI: a lightweight web dashboard for browsing, editing, and auditing the knowledge store.
- Streaming research results: return Tomes to the agent incrementally as the Researcher discovers them.
- Plugin search APIs: community-contributed drivers for DuckDuckGo, Google Custom Search, arXiv, PubMed, etc.

______________________________________________________________________

## Appendix A: Quick-Start (Docker Compose)

```yaml
version: '3.8'
services:
  mongo:
    image: mongodb/mongodb-atlas-local:latest
    ports: ['27017:27017']
    volumes: ['mongo_data:/data/db']

  ollama:
    image: ollama/ollama:latest
    ports: ['11434:11434']
    volumes: ['ollama_data:/root/.ollama']
    command: serve

  librarian:
    build: .
    ports: ['8000:8000']
    environment:
      LIBRARIAN_DATABASE_URI: mongodb://mongo:27017
      LIBRARIAN_EMBEDDING_OLLAMA_URL: http://ollama:11434
      LIBRARIAN_WEB_SEARCH_API_KEY: ${LIBRARIAN_WEB_SEARCH_API_KEY}
    depends_on: [mongo, ollama]

volumes:
  mongo_data:
  ollama_data:
```

After starting, pull the embedding model once:

```bash
docker compose exec ollama ollama pull nomic-embed-text
```

Then point any MCP client at `http://localhost:8000` to start using all three tools.
