# The Librarian — Low-Level Design

### API Contracts & User Journeys

| | |
| ----------- | ------------ |
| **Version** | 0.1 (Draft) |
| **Date** | March 2026 |
| **Status** | Design Phase |

______________________________________________________________________

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
1. [API 1 — Calling Agent ↔ Librarian MCP Agent](#2-api-1--calling-agent--librarian-mcp-agent)
1. [API 2 — Librarian MCP Agent ↔ Tome Repository Agent](#3-api-2--librarian-mcp-agent--tome-repository-agent)
1. [User Journeys](#4-user-journeys)
1. [Error Taxonomy](#5-error-taxonomy)

______________________________________________________________________

## 1. Architecture Overview

The system exposes two distinct API boundaries. The **MCP API** (API 1) is the external surface that any MCP-compatible calling agent communicates with. The **Internal RPC API** (API 2) is the in-process contract between the Librarian orchestrator and the Tome Repository Agent, which owns all persistence decisions.

```
┌──────────────────────────────────┐
│  CALLING AGENT                   │
│  (Claude, Gemini, gemini-cli…)   │
└──────────────┬───────────────────┘
               │
               │  ◄── API 1: MCP Protocol (stdio / SSE) ──►
               │  tools: library.search / library.ingest / library.research
               │
               ▼
┌──────────────────────────────────┐
│  LIBRARIAN MCP AGENT             │
│  Orchestrates search, ingest,    │
│  and research; calls Repo Agent  │
└──────────────┬───────────────────┘
               │
               │  ◄── API 2: Internal Agent RPC ──►
               │  namespaces: tome.* / job.*
               │
               ▼
┌──────────────────────────────────┐
│  TOME REPOSITORY AGENT           │
│  Owns persistence decisions,     │
│  dedup, versioning, MongoDB      │
└──────────────────────────────────┘
```

The Calling Agent never communicates with the Tome Repository Agent directly. All persistence is brokered through the Librarian MCP Agent.

______________________________________________________________________

## 2. API 1 — Calling Agent ↔ Librarian MCP Agent

All three tools follow the standard MCP tool-call envelope. Errors are returned as a JSON error payload (see [§5 Error Taxonomy](#5-error-taxonomy)).

______________________________________________________________________

### 2.1 `library.search`

Converts a natural-language query to a vector embedding and retrieves the most semantically relevant Tomes from the library.

#### Input — `SearchInput`

| Field | Type | Required | Constraints | Default |
| ----------------- | ----------- | -------- | --------------------- | ------- |
| `query` | `str` | Yes | max length 2000 chars | — |
| `top_k` | `int` | No | 1 ≤ value ≤ 20 | `5` |
| `category` | `str\|null` | No | — | `null` |
| `min_confidence` | `float` | No | 0.0 ≤ value ≤ 1.0 | `0.5` |
| `include_summary` | `bool` | No | — | `false` |

#### Output — `SearchOutput`

| Field | Type | Description |
| ------------ | --------- | -------------------------------------------------------------------- |
| `tomes` | `Tome[]` | Matching Tome documents, sorted by similarity score descending |
| `scores` | `float[]` | Cosine similarity scores corresponding to each Tome (parallel array) |
| `query_id` | `str` | Unique identifier for this search request |
| `from_cache` | `bool` | Whether the embedding was served from the LRU cache |

#### Errors

| Code | When raised |
| ------------------- | -------------------------------------------- |
| `EMBED_UNAVAILABLE` | Embedding model is unreachable |
| `NO_RESULTS` | Zero tomes exceed `min_confidence` threshold |

______________________________________________________________________

### 2.2 `library.ingest`

Validates, chunks, embeds, and stores a raw knowledge payload as one or more Tomes.

#### Input — `IngestInput`

| Field | Type | Required | Constraints | Default |
| -------------- | ----------- | -------- | ----------- | ------- |
| `content` | `str` | Yes | — | — |
| `title` | `str\|null` | No | — | `null` |
| `category` | `str\|null` | No | — | `null` |
| `tags` | `str[]` | No | — | `[]` |
| `source_url` | `str\|null` | No | — | `null` |
| `skip_verify` | `bool` | No | — | `false` |
| `allow_update` | `bool` | No | — | `true` |

#### Output — `IngestOutput`

| Field | Type | Description |
| --------------- | -------------- | ------------------------------------------------------------------- |
| `tome_ids` | `str[]` | IDs of all Tomes created or updated during this call |
| `tomes` | `Tome[]` | Full Tome objects as they were stored |
| `confidence` | `float` | Aggregate verification confidence score (0.0–1.0) |
| `chunks` | `int` | Number of chunks the input was split into |
| `status` | `IngestStatus` | One of: `stored`, `rejected`, `partial` |
| `reject_reason` | `str\|null` | Human-readable rejection explanation; non-null when status=rejected |

#### Errors

| Code | When raised |
| ------------------- | ------------------------------------------ |
| `VERIFY_FAILED` | Confidence below reject threshold (< 0.3) |
| `CONTENT_TOO_SHORT` | Input is too short to chunk meaningfully |
| `EMBED_UNAVAILABLE` | Embedding model is unreachable |

______________________________________________________________________

### 2.3 `library.research`

Dispatches a Researcher to search the web, synthesise findings, and ingest results as new Tomes. Supports both synchronous and asynchronous operation.

#### Input — `ResearchInput` (initial call)

| Field | Type | Required | Constraints | Default |
| ----------- | --------------- | -------- | ---------------------------------------- | ---------- |
| `topic` | `str` | Yes | — | — |
| `context` | `str\|null` | No | — | `null` |
| `depth` | `ResearchDepth` | No | `shallow`, `standard`, `deep` | `standard` |
| `max_tomes` | `int` | No | ≥ 1 | `10` |
| `category` | `str\|null` | No | — | `null` |
| `async` | `bool` | No | (field alias; Python name: `async_mode`) | `false` |

#### Input — polling call (async only)

Pass `job_id` as `topic` or use the same `ResearchInput` with a recognised job ID string. The Librarian detects an existing job ID and returns current status instead of starting a new job.

#### Output — `ResearchOutput`

| Field | Type | Description |
| ------------- | ----------- | ------------------------------------------------------------ |
| `job_id` | `str` | ID of the ResearchJob record |
| `tome_ids` | `str[]` | IDs of all Tomes created; empty while job is pending/running |
| `tomes` | `Tome[]` | Full Tome objects; empty while job is pending/running |
| `sources` | `str[]` | URLs of web pages consulted |
| `query_count` | `int` | Number of individual search queries issued |
| `status` | `JobStatus` | One of: `pending`, `running`, `completed`, `failed` |

#### Errors

| Code | When raised |
| ------------------------ | ------------------------------------------ |
| `SEARCH_API_UNAVAILABLE` | Web search API key missing or rate-limited |
| `JOB_NOT_FOUND` | Polling an unknown `job_id` |

______________________________________________________________________

### 2.4 Shared Type — `Tome`

All tool outputs that include `Tome` objects use this schema (sourced from `src/models/tome.py`):

| Field | Type | Description |
| --------------- | ------------ | ----------------------------------------------------- |
| `id` | `str` | UUID hex string, auto-generated |
| `title` | `str` | Short descriptive title (max 120 chars) |
| `content` | `str` | Full text body |
| `summary` | `str` | One-to-two sentence summary |
| `category` | `str` | High-level domain category |
| `tags` | `str[]` | Freeform topic tags |
| `source_url` | `str\|null` | Origin URL if sourced from the web |
| `source_type` | `SourceType` | One of: `agent_input`, `researcher`, `manual` |
| `confidence` | `float` | Truthiness confidence score (0.0–1.0) |
| `embedding` | `float[]` | Dense vector embedding of the content |
| `created_at` | `datetime` | ISO 8601 UTC timestamp of creation |
| `updated_at` | `datetime` | ISO 8601 UTC timestamp of last modification |
| `version` | `int` | Incremented on each content update; starts at 1 |
| `superseded_by` | `str\|null` | ID of successor Tome if this one has been replaced |
| `research_job` | `str\|null` | ID of the ResearchJob that produced this Tome, if any |

______________________________________________________________________

## 3. API 2 — Librarian MCP Agent ↔ Tome Repository Agent

The Tome Repository Agent exposes two operation namespaces via in-process RPC. The Librarian is the only caller; no external agent calls these directly.

______________________________________________________________________

### 3.1 Tome Namespace

| Operation | Inputs | Returns | Notes |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- | --------------------------------------------------------------------- |
| `tome.store` | `content`, `title`, `summary`, `category`, `tags`, `source_url`, `source_type`, `confidence`, `embedding`, `research_job?` | `Tome` (with `id`, `created_at`, `updated_at`, `version=1`) | Agent decides whether to version or insert fresh |
| `tome.get` | `tome_id: str` | `Tome \| None` | |
| `tome.update` | `tome_id: str`, `patch: TomePatch` | `Tome` | Bumps `version`, sets `updated_at` |
| `tome.supersede` | `old_id: str`, `new_id: str` | `void` | Sets `superseded_by` on the old Tome |
| `tome.search` | `embedding: float[]`, `top_k: int`, `category?: str`, `min_confidence?: float` | `SearchResult[]` | Executes `$vectorSearch`; each result = Tome + score |
| `tome.find_near_duplicates` | `embedding: float[]`, `threshold: float` (default 0.95), `limit: int` (default 5) | `DuplicateMatch[]` | Each match = Tome + similarity score; called by ingestor before store |
| `tome.find_by_job` | `job_id: str` | `Tome[]` | Returns all Tomes produced by a given research job |

#### `TomePatch` fields (all optional)

`content`, `title`, `summary`, `category`, `tags`, `source_url`, `confidence`, `embedding`

______________________________________________________________________

### 3.2 Job Namespace

| Operation | Inputs | Returns | Notes |
| ----------------- | ----------------------------------------------------- | --------------------- | -------------------------------------------------- |
| `job.create` | `topic: str`, `context?: str`, `depth: ResearchDepth` | `ResearchJob` | Initial status = `pending` |
| `job.get` | `job_id: str` | `ResearchJob \| None` | |
| `job.set_running` | `job_id: str` | `void` | Sets `status=running`, records `started_at` |
| `job.add_queries` | `job_id: str`, `queries: str[]` | `void` | Appends to the `queries` list |
| `job.complete` | `job_id: str`, `tome_ids: str[]`, `queries: str[]` | `void` | Sets `status=completed`, `finished_at`, `tome_ids` |
| `job.fail` | `job_id: str`, `error: str` | `void` | Sets `status=failed`, `finished_at`, `error` |

#### `ResearchJob` shape (sourced from `src/models/research_job.py`)

| Field | Type | Description |
| ------------- | ---------------- | ---------------------------------------------- |
| `id` | `str` | UUID hex string, auto-generated |
| `topic` | `str` | Topic string passed to the Researcher |
| `context` | `str\|null` | Optional agent-supplied context |
| `status` | `JobStatus` | `pending`, `running`, `completed`, or `failed` |
| `queries` | `str[]` | Search queries issued during the run |
| `tome_ids` | `str[]` | IDs of Tomes created by this job |
| `error` | `str\|null` | Error message if status is `failed` |
| `started_at` | `datetime` | When the job began executing |
| `finished_at` | `datetime\|null` | When the job completed or failed |

______________________________________________________________________

## 4. User Journeys

Each journey shows the full call chain across both API boundaries from the moment the Calling Agent issues a request to the moment it receives a response.

______________________________________________________________________

### Journey A — Knowledge Retrieval (cache hit)

The agent knows what it wants and the library already has it.

```
Calling Agent → library.search(query="how does X work", top_k=5)
  Librarian:
    embed query → LRU cache hit → from_cache=true
    → tome.search(embedding, top_k=5, min_confidence=0.5)
      Repo Agent → $vectorSearch → returns 3 Tomes with scores
    re-rank by score + recency
← Calling Agent ← SearchOutput{tomes: [t1,t2,t3], scores: [...], query_id: "q1", from_cache: true}
```

______________________________________________________________________

### Journey B — Knowledge Retrieval (empty result → research trigger)

The library has nothing on the topic; the agent decides to commission research.

```
Calling Agent → library.search(query="obscure topic")
  Librarian → embed query → tome.search(embedding, top_k=5)
    Repo Agent → $vectorSearch → returns [] (no results above threshold)
← Calling Agent ← SearchOutput{tomes: [], scores: [], query_id: "q2", from_cache: false}

Calling Agent → library.research(topic="obscure topic", depth=standard)
  [see Journey E — synchronous research]
← Calling Agent ← ResearchOutput{job_id: "j1", tomes: [...], status: "completed"}

Calling Agent → library.search(query="obscure topic")
  ← now returns populated results (Journey A path)
```

______________________________________________________________________

### Journey C — Knowledge Ingestion (happy path)

New content is verified, chunked, and stored as two fresh Tomes.

```
Calling Agent → library.ingest(content="...", source_url="https://...")
  Librarian:
    pre-flight: validate length and fields
    Verifier:
      extract 5 factual claims from content
      web search each claim → scores → confidence=0.82
    Chunker:
      split into 2 chunks at sentence boundaries
      generate title + summary for each
    Embedder: embed chunk 1, embed chunk 2

    For chunk 1:
      → tome.find_near_duplicates(embedding_1, threshold=0.95)
        Repo Agent → $vectorSearch → [] (no duplicates)
      → tome.store(chunk_1_data)
        Repo Agent → insert → Tome{id:"t1", version:1, ...}

    For chunk 2:
      → tome.find_near_duplicates(embedding_2, threshold=0.95)
        Repo Agent → $vectorSearch → [] (no duplicates)
      → tome.store(chunk_2_data)
        Repo Agent → insert → Tome{id:"t2", version:1, ...}

← Calling Agent ← IngestOutput{
    tome_ids: ["t1","t2"],
    tomes: [Tome_t1, Tome_t2],
    confidence: 0.82,
    chunks: 2,
    status: "stored",
    reject_reason: null
  }
```

______________________________________________________________________

### Journey D — Knowledge Ingestion (duplicate detected)

Content is very similar to an existing Tome; `allow_update=true` causes an update instead of insert.

```
Calling Agent → library.ingest(content="similar content", allow_update=true)
  Librarian:
    Verifier → confidence=0.75
    Chunker → 1 chunk
    Embedder → embedding

    → tome.find_near_duplicates(embedding, threshold=0.95, limit=5)
      Repo Agent → $vectorSearch → [{tome: Tome{id:"t_existing"}, similarity: 0.97}]

    similarity 0.97 > threshold 0.95 → merge/update path:
      → tome.update("t_existing", patch={content, confidence, updated_at})
        Repo Agent → bump version to 2, set updated_at → Tome{id:"t_existing", version:2}

← Calling Agent ← IngestOutput{
    tome_ids: ["t_existing"],
    tomes: [Tome_t_existing_v2],
    confidence: 0.75,
    chunks: 1,
    status: "stored",
    reject_reason: null
  }
```

______________________________________________________________________

### Journey E — Research (synchronous)

Full synchronous research flow. Blocks until all web fetching and ingestion is complete.

```
Calling Agent → library.research(topic="topic X", depth=standard)
  Librarian:
    → job.create("topic X", depth=standard)
      Repo Agent → insert ResearchJob{id:"j1", status:"pending"}
    → job.set_running("j1")
      Repo Agent → status="running", started_at=now

    Researcher sub-agent:
      plan 6 targeted search queries for "topic X"
      → job.add_queries("j1", ["query1", ..., "query6"])
        Repo Agent → append to queries[]

      WebSearch API: fetch results for all 6 queries
      trafilatura: extract body text from top URLs
      synthesise content across sources into N logical sub-topics

      ingest pipeline for each sub-topic (Journey C):
        → tome.find_near_duplicates(embedding) → []
        → tome.store(chunk_data) → Tome{id:"tN"}

    → job.complete("j1", tome_ids=["t3","t4","t5"], queries=["query1",...])
      Repo Agent → status="completed", finished_at=now, tome_ids=[...]

← Calling Agent ← ResearchOutput{
    job_id: "j1",
    tome_ids: ["t3","t4","t5"],
    tomes: [Tome_t3, Tome_t4, Tome_t5],
    sources: ["https://...", ...],
    query_count: 6,
    status: "completed"
  }
```

______________________________________________________________________

### Journey F — Research (asynchronous)

Job is kicked off immediately; the agent polls separately for status.

```
Calling Agent → library.research(topic="topic X", async=true)
  Librarian:
    → job.create("topic X", depth=standard)
      Repo Agent → ResearchJob{id:"j2", status:"pending"}
    starts Journey E in background (non-blocking)
← Calling Agent ← ResearchOutput{
    job_id: "j2",
    tome_ids: [],
    tomes: [],
    sources: [],
    query_count: 0,
    status: "pending"
  }

... agent does other work ...

Calling Agent → library.research(topic="j2")   ← polling by job_id
  Librarian:
    detect "j2" matches an existing job ID
    → job.get("j2")
      Repo Agent → ResearchJob{id:"j2", status:"running", ...}
← Calling Agent ← ResearchOutput{job_id:"j2", status:"running", tomes:[], ...}

... later ...

Calling Agent → library.research(topic="j2")   ← final poll
  Librarian → job.get("j2") → status:"completed"
  Librarian → tome.find_by_job("j2") → [Tome_t3, Tome_t4, Tome_t5]
← Calling Agent ← ResearchOutput{
    job_id: "j2",
    tome_ids: ["t3","t4","t5"],
    tomes: [Tome_t3, Tome_t4, Tome_t5],
    sources: ["https://...", ...],
    query_count: 6,
    status: "completed"
  }
```

______________________________________________________________________

## 5. Error Taxonomy

All errors follow this envelope regardless of which API boundary they cross:

```json
{
  "error": {
    "code": "VERIFY_FAILED",
    "message": "Human-readable description of the failure.",
    "details": {}
  }
}
```

| Code | Layer | Meaning |
| ------------------------ | ---------- | -------------------------------------------------------------- |
| `EMBED_UNAVAILABLE` | Librarian | Embedding model is unreachable (Ollama down, model not loaded) |
| `SEARCH_API_UNAVAILABLE` | Librarian | Web search API key is missing, invalid, or rate-limited |
| `VERIFY_FAILED` | Librarian | Verification confidence is below `reject_threshold` (< 0.3) |
| `CONTENT_TOO_SHORT` | Librarian | Input is too short to chunk into at least one meaningful Tome |
| `JOB_NOT_FOUND` | Librarian | Polling request references an unknown `job_id` |
| `NO_RESULTS` | Librarian | Search returned zero results above `min_confidence` threshold |
| `DB_UNAVAILABLE` | Repo Agent | MongoDB is unreachable or the connection timed out |
| `DUPLICATE_CONFLICT` | Repo Agent | Near-duplicate detected and `allow_update=false` |
| `EMBEDDING_MISMATCH` | Repo Agent | Stored vector dimensions don't match the configured index |

`details` is an optional free-form object; callers should not depend on its structure remaining stable across versions.
