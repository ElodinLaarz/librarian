# The Librarian — Low-Level Design

### API Contracts & User Journeys

| | |
| ----------- | ------------ |
| **Version** | 0.2 (Draft) |
| **Date** | March 2026 |
| **Status** | Design Phase |

______________________________________________________________________

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
1. [API 1 — MCP Tools](#2-api-1--mcp-tools)
1. [Service Layer](#3-service-layer)
1. [API 2 — Repository Layer](#4-api-2--repository-layer)
1. [User Journeys](#5-user-journeys)
1. [Error Taxonomy](#6-error-taxonomy)
1. [Beyond v1 — Research Tool](#7-beyond-v1--research-tool)

______________________________________________________________________

## 1. Architecture Overview

```
┌──────────────────────────────────┐
│  CALLING AGENT                   │
│  (Claude, Gemini, gemini-cli…)   │
└──────────────┬───────────────────┘
               │
               │  ◄── API 1: MCP Protocol (stdio / SSE) ──►
               │  library_search / library_ingest
               │
               ▼
┌──────────────────────────────────┐
│  LIBRARIAN MCP SERVER            │
│  server.py — tool routing,       │
│  input validation, wiring        │
└──────┬──────────────┬────────────┘
       │              │
       ▼              ▼
┌────────────┐  ┌─────────────────────────────────┐
│  Search    │  │  Ingestor                        │
│  Engine    │  │  ┌──────────┐  ┌──────────────┐ │
│            │  │  │ Verifier │  │ Embedding    │ │
│ Embedding  │  │  │          │  │ Service      │ │
│ Service    │  │  └──────────┘  └──────────────┘ │
└──────┬─────┘  └──────────────────┬──────────────┘
       │                           │
       │  ◄── API 2: Repository Layer ──►
       │                           │
       ▼                           ▼
┌──────────────────────────────────┐
│  TomeRepository                  │
│  (abstract; MongoDB impl)        │
└──────────────────────────────────┘
```

The Calling Agent communicates only with the MCP server. All persistence goes
through `TomeRepository`. `EmbeddingService` is shared between `SearchEngine`
and `Ingestor`.

______________________________________________________________________

## 2. API 1 — MCP Tools

Both tools follow the standard MCP tool-call envelope. Errors use the envelope
in [§6 Error Taxonomy](#6-error-taxonomy).

______________________________________________________________________

### 2.1 `library_search`

Converts a natural-language query to a vector embedding and retrieves the most
semantically relevant Tomes.

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
| `tomes` | `Tome[]` | Matching Tome documents, sorted by re-ranked score descending |
| `scores` | `float[]` | Final scores corresponding to each Tome (parallel array) |
| `query_id` | `str` | Unique identifier for this search request |
| `from_cache` | `bool` | Whether the query embedding was served from the LRU cache |

#### Errors

| Code | When raised |
| ------------------- | -------------------------------------------- |
| `EMBED_UNAVAILABLE` | Embedding model unreachable |
| `NO_RESULTS` | Zero tomes exceed `min_confidence` threshold |

______________________________________________________________________

### 2.2 `library_ingest`

Validates, chunks, embeds, and stores a raw knowledge payload as one or more
Tomes.

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
| `tome_ids` | `str[]` | IDs of all Tomes created or updated |
| `tomes` | `Tome[]` | Full Tome objects as stored |
| `confidence` | `float` | Aggregate verification confidence (0.0–1.0) |
| `chunks` | `int` | Number of chunks the input was split into |
| `status` | `IngestStatus` | One of: `stored`, `rejected`, `partial` |
| `reject_reason` | `str\|null` | Rejection explanation; non-null when `status=rejected` |

#### Errors

| Code | When raised |
| ------------------- | ---------------------------------------- |
| `VERIFY_FAILED` | Confidence below reject threshold (< 0.3) |
| `CONTENT_TOO_SHORT` | Input too short to chunk meaningfully |
| `EMBED_UNAVAILABLE` | Embedding model unreachable |

______________________________________________________________________

### 2.3 Shared Type — `Tome`

All tool outputs that include `Tome` objects use this schema (source:
`src/models/tome.py`):

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

## 3. Service Layer

The service layer sits between the MCP server and the repository. It owns all
business logic; the MCP server only does routing and validation.

______________________________________________________________________

### 3.1 `SearchEngine`

Source: `src/services/search.py`

Dependencies: `SearchSettings`, `EmbeddingService`, `TomeRepository`

| Method | Signature | Returns | Notes |
| --------------- | -------------------------------------------- | --------------------- | ------------------------------------------ |
| `search` | `(params: SearchInput)` | `SearchOutput` | Full pipeline: embed → search → re-rank |
| `_embed_query` | `(query: str)` | `(list[float], bool)` | Returns `(vector, from_cache)` |
| `_rerank` | `(results, recency_boost: float=0.1)` | `results` | Blends similarity, recency, confidence |

**Pipeline:**

1. `_embed_query` — pass query through `EmbeddingService.embed()`; SHA-256 cache
   hit returns immediately with `from_cache=True`
1. `TomeRepository.vector_search` — ANN search with optional `category` and
   `min_confidence` filters
1. `_rerank` — blend raw cosine score with `updated_at` recency and
   `confidence`; re-sort descending
1. Serialize `tomes` and parallel `scores`; generate `query_id`; return
   `SearchOutput`

______________________________________________________________________

### 3.2 `EmbeddingService`

Source: `src/services/embedding.py`

Abstract base; concrete implementations cover Ollama, sentence-transformers,
OpenAI. Shared by `SearchEngine` and `Ingestor`.

| Method | Signature | Returns | Notes |
| -------------- | ---------------------- | ------------- | ----------------------------------------- |
| `initialize` | `()` | `None` | Load model; warm up provider connection |
| `embed` | `(text: str)` | `list[float]` | Single text; returns from LRU cache if hit |
| `embed_batch` | `(texts: list[str])` | `list[list[float]]` | Batch embed for throughput |
| `_cache_key` | `(text: str)` | `str` | SHA-256 hash of text for cache lookup |
| `dimensions` | property | `int` | Output vector dimensionality |

Cache is an in-memory LRU keyed on `_cache_key(text)`. Cache size is
configurable (`embedding.cache_size`). Cold on restart.

______________________________________________________________________

### 3.3 `Ingestor`

Source: `src/services/ingestor.py`

Dependencies: `LibrarianConfig`, `EmbeddingService`, `Verifier`,
`TomeRepository`

| Method | Signature | Returns | Notes |
| ----------------------- | -------------------------------------------- | ------------- | --------------------------------------------------- |
| `ingest` | `(params: IngestInput)` | `IngestOutput` | Full pipeline: validate → verify → chunk → embed → dedup → store |
| `_validate` | `(params: IngestInput)` | `None` | Length check, HTML sanitisation; raises on failure |
| `_chunk` | `(content: str)` | `list[str]` | LLM-driven decomposition into self-contained atomic facts |
| `_classify_and_tag` | `(chunk: str, category_hint: str\|None)` | `(str, list[str])` | Returns `(category, tags)` |
| `_generate_title_and_summary` | `(chunk: str)` | `(str, str)` | Returns `(title, summary)` |
| `_dedup_and_store` | `(tome: Tome, allow_update: bool)` | `str` | Near-dup check → merge or insert; returns Tome ID |

**Pipeline:**

1. `_validate` — enforce minimum length, sanitise HTML/markdown
1. `Verifier.verify` — unless `skip_verify=True`; reject if
   `confidence < reject_threshold`
1. `_chunk` — LLM prompt decomposes content into atomic, self-contained facts;
   each chunk must stand alone without requiring surrounding context. Word count
   is a soft upper bound (~400 words), not the splitting criterion. A single
   short input may produce one chunk; a dense multi-topic input may produce many.
1. For each chunk:
   1. `_classify_and_tag` — category classification + keyword tag extraction
   1. `_generate_title_and_summary` — LLM-generated title and summary
   1. `EmbeddingService.embed` — produce chunk vector
   1. `_dedup_and_store` — call `TomeRepository.find_near_duplicates`; if
      similarity ≥ 0.95 and `allow_update=True` → `update`, else → `insert`

______________________________________________________________________

### 3.4 `Verifier`

Source: `src/services/verifier.py`

Dependencies: `VerificationSettings`, `WebSearchClient`

| Method | Signature | Returns | Notes |
| ---------------------- | ------------------------------ | -------------------- | -------------------------------------------- |
| `verify` | `(content: str)` | `VerificationResult` | Full pipeline; returns offline result if unavailable |
| `_extract_claims` | `(content: str)` | `list[str]` | 3–7 claims via zero-shot prompt |
| `_check_claim` | `(claim: str)` | `ClaimResult` | Web search + snippet scoring per claim |
| `_aggregate_confidence`| `(results: list[ClaimResult])` | `float` | Weighted score across all claims |
| `_make_offline_result` | `()` | `VerificationResult` | Synthetic 0.6 confidence when search unavailable |

**Types:**

```python
@dataclass
class ClaimResult:
    claim: str
    verdict: VerificationVerdict  # supported | contradicted | unverifiable
    evidence: str

@dataclass
class VerificationResult:
    confidence: float  # 0.0–1.0
    claims: list[ClaimResult]
    skipped: bool
```

**Confidence thresholds** (from `VerificationSettings`):

| Range | Action |
| ----------- | ---------------------------------------- |
| ≥ 0.7 | Store with full confidence |
| 0.3 – 0.7 | Store with low-confidence flag |
| < 0.3 | Reject; raise `VERIFY_FAILED` |

When `skip_verify=True` or no search API key is configured,
`_make_offline_result` returns `confidence=0.6` and `skipped=True`.

______________________________________________________________________

## 4. API 2 — Repository Layer

Defined in `src/storage/`. Abstract base classes; concrete implementations bind
to MongoDB. Services are the only callers.

______________________________________________________________________

### 4.1 `TomeRepository`

Source: `src/storage/tome_repository.py`

| Method | Signature | Returns | Notes |
| ----------------------- | ------------------------------------------------------------------------------ | -------------------------- | ----------------------------------------------------- |
| `insert` | `(tome: Tome)` | `str` | Inserts a new Tome; returns its ID |
| `get_by_id` | `(tome_id: str)` | `Tome \| None` | |
| `update` | `(tome_id: str, updates: dict[str, Any])` | `bool` | `True` if a document was modified |
| `supersede` | `(old_id: str, new_id: str)` | `None` | Sets `superseded_by` on the old Tome |
| `vector_search` | `(embedding: float[], top_k: int, category?: str, min_confidence: float=0.5)` | `list[tuple[Tome, float]]` | ANN search via `$vectorSearch`; sorted by similarity |
| `find_near_duplicates` | `(embedding: float[], threshold: float=0.95)` | `list[Tome]` | Tomes with cosine similarity above threshold |
| `find_by_research_job` | `(job_id: str)` | `list[Tome]` | All Tomes from a given research job (v2) |

______________________________________________________________________

## 5. User Journeys

Each journey shows the full call chain across both API boundaries.

______________________________________________________________________

### Journey A — Search (cache hit)

```
Agent → library_search(query="how does X work", top_k=5)
  SearchEngine.search():
    _embed_query("how does X work")
      EmbeddingService.embed() → LRU cache hit → from_cache=True
    tome_repo.vector_search(embedding, top_k=5, min_confidence=0.5)
      MongoDB $vectorSearch → [(Tome_t1, 0.91), (Tome_t2, 0.87), (Tome_t3, 0.81)]
    _rerank(results, recency_boost=0.1) → re-sorted list
← Agent ← SearchOutput{
    tomes: [t1, t2, t3],
    scores: [0.93, 0.88, 0.82],
    query_id: "q1",
    from_cache: true
  }
```

______________________________________________________________________

### Journey B — Search (no results)

```
Agent → library_search(query="obscure topic")
  SearchEngine.search():
    _embed_query("obscure topic") → cache miss → EmbeddingService.embed()
    tome_repo.vector_search(embedding, top_k=5, min_confidence=0.5)
      MongoDB $vectorSearch → [] (nothing above threshold)
← Agent ← SearchOutput{tomes: [], scores: [], query_id: "q2", from_cache: false}
```

The agent may call `library_ingest` with curated content on the topic, or
`library_research` (v2) to auto-populate.

______________________________________________________________________

### Journey C — Ingest (happy path, multi-fact decomposition)

Input is a dense article covering several distinct facts. `_chunk` decomposes
it into three self-contained atomic facts rather than splitting at a word-count
boundary.

```
Agent → library_ingest(content="<article covering facts A, B, C>", source_url="https://...")
  Ingestor.ingest():
    _validate() → OK
    Verifier.verify(content):
      _extract_claims() → 5 claims
      _check_claim() × 5 → [supported, supported, unverifiable, supported, supported]
      _aggregate_confidence() → 0.82
      → VerificationResult{confidence: 0.82, skipped: false}
    0.82 ≥ 0.3 → proceed
    _chunk(content):
      LLM prompt: "decompose into atomic, self-contained facts"
      → ["Fact A (standalone)", "Fact B (standalone)", "Fact C (standalone)"]
      ← 3 chunks, each independently meaningful

    chunk "Fact A":
      _classify_and_tag() → ("science", ["tag_a"])
      _generate_title_and_summary() → ("Title A", "Summary A")
      EmbeddingService.embed() → embedding_a
      _dedup_and_store(allow_update=true):
        tome_repo.find_near_duplicates(embedding_a) → []
        tome_repo.insert(Tome{...}) → "t1"

    chunk "Fact B":
      _classify_and_tag() → ("science", ["tag_b"])
      _generate_title_and_summary() → ("Title B", "Summary B")
      EmbeddingService.embed() → embedding_b
      _dedup_and_store(allow_update=true):
        tome_repo.find_near_duplicates(embedding_b) → []
        tome_repo.insert(Tome{...}) → "t2"

    chunk "Fact C":
      _classify_and_tag() → ("science", ["tag_c"])
      _generate_title_and_summary() → ("Title C", "Summary C")
      EmbeddingService.embed() → embedding_c
      _dedup_and_store(allow_update=true):
        tome_repo.find_near_duplicates(embedding_c) → []
        tome_repo.insert(Tome{...}) → "t3"

← Agent ← IngestOutput{
    tome_ids: ["t1", "t2", "t3"],
    tomes: [Tome_t1, Tome_t2, Tome_t3],
    confidence: 0.82,
    chunks: 3,
    status: "stored",
    reject_reason: null
  }
```

______________________________________________________________________

### Journey D — Ingest (near-duplicate update)

```
Agent → library_ingest(content="similar to existing", allow_update=true)
  Ingestor.ingest():
    _validate() → OK
    Verifier.verify() → VerificationResult{confidence: 0.75}
    _chunk() → ["chunk_1"]
    _classify_and_tag() → ("science", ["tag_a"])
    _generate_title_and_summary() → ("Title A revised", "Summary A revised")
    EmbeddingService.embed("chunk_1") → embedding
    _dedup_and_store(tome, allow_update=true):
      tome_repo.find_near_duplicates(embedding, threshold=0.95)
        → [Tome{id:"t_existing"}]  ← similarity 0.97
      similarity 0.97 ≥ 0.95 and allow_update=True → update path:
        tome_repo.update("t_existing", {content, title, summary, confidence, updated_at})
          → True  (version bumped to 2 by MongoDB write)

← Agent ← IngestOutput{
    tome_ids: ["t_existing"],
    tomes: [Tome{id:"t_existing", version:2}],
    confidence: 0.75,
    chunks: 1,
    status: "stored",
    reject_reason: null
  }
```

______________________________________________________________________

### Journey E — Ingest (rejected)

```
Agent → library_ingest(content="The moon is made of cheese...")
  Ingestor.ingest():
    _validate() → OK
    Verifier.verify():
      _extract_claims() → 3 claims
      _check_claim() × 3 → [contradicted, contradicted, contradicted]
      _aggregate_confidence() → 0.11
      → VerificationResult{confidence: 0.11}
    0.11 < 0.3 → raise VERIFY_FAILED

← Agent ← IngestOutput{
    tome_ids: [],
    tomes: [],
    confidence: 0.11,
    chunks: 0,
    status: "rejected",
    reject_reason: "Confidence 0.11 below reject threshold 0.3"
  }
```

______________________________________________________________________

## 6. Error Taxonomy

All errors use this envelope:

```json
{
  "error": {
    "code": "VERIFY_FAILED",
    "message": "Human-readable description.",
    "details": {}
  }
}
```

| Code | Layer | Meaning |
| -------------------- | ----------- | --------------------------------------------------------------- |
| `EMBED_UNAVAILABLE` | SearchEngine / Ingestor | Embedding model unreachable (Ollama down, model not loaded) |
| `VERIFY_FAILED` | Verifier | Confidence below `reject_threshold` (< 0.3) |
| `CONTENT_TOO_SHORT` | Ingestor | Input too short to produce at least one chunk |
| `NO_RESULTS` | SearchEngine | Zero results above `min_confidence` threshold |
| `DB_UNAVAILABLE` | TomeRepository | MongoDB unreachable or connection timed out |
| `DUPLICATE_CONFLICT` | TomeRepository | Near-duplicate found and `allow_update=false` |
| `EMBEDDING_MISMATCH` | TomeRepository | Stored vector dimensions don't match the configured index |

`details` is an optional free-form object; its structure is not stable across
versions.

______________________________________________________________________

## 7. Beyond v1 — Research Tool

`library_research` and its supporting infrastructure are deferred to v2. The
contracts below are provisional; they will be fleshed out once the core search
and ingest tools are complete and stable.

______________________________________________________________________

### 7.1 `library_research` (provisional)

Dispatches a Researcher to search the web, synthesise findings, and pipe results
through the ingest pipeline as new Tomes.

#### Input — `ResearchInput`

| Field | Type | Required | Default | Notes |
| ----------- | --------------- | -------- | ---------- | ----------------------------------------- |
| `topic` | `str` | Yes | — | |
| `context` | `str\|null` | No | `null` | |
| `depth` | `ResearchDepth` | No | `standard` | `shallow` / `standard` / `deep` |
| `max_tomes` | `int` | No | `10` | |
| `category` | `str\|null` | No | `null` | |
| `async` | `bool` | No | `false` | Python field name: `async_mode` |

Polling: pass a known `job_id` as `topic`; server detects and returns job
status instead of starting a new job.

#### Output — `ResearchOutput`

| Field | Type | Description |
| ------------- | ----------- | ------------------------------------------------------------ |
| `job_id` | `str` | ID of the ResearchJob record |
| `tome_ids` | `str[]` | IDs of Tomes created; empty while pending/running |
| `tomes` | `Tome[]` | Full Tome objects; empty while pending/running |
| `sources` | `str[]` | URLs consulted |
| `query_count` | `int` | Search queries issued |
| `status` | `JobStatus` | `pending` / `running` / `completed` / `failed` |

______________________________________________________________________

### 7.2 `JobRepository` (provisional)

Source: `src/storage/job_repository.py`

| Method | Signature | Returns | Notes |
| --------------- | ------------------------------------------------------------ | --------------------- | ------------------------------------------ |
| `create` | `(job: ResearchJob)` | `str` | Returns job ID |
| `get_by_id` | `(job_id: str)` | `ResearchJob \| None` | |
| `set_running` | `(job_id: str)` | `None` | Sets `status=running`, `started_at` |
| `add_queries` | `(job_id: str, queries: list[str])` | `None` | Appends to `queries` list |
| `set_completed` | `(job_id: str, tome_ids: list[str], finished_at: datetime)` | `None` | Sets `status=completed`, `finished_at` |
| `set_failed` | `(job_id: str, error: str, finished_at: datetime)` | `None` | Sets `status=failed`, `finished_at` |

______________________________________________________________________

### 7.3 Additional errors (v2)

| Code | Layer | Meaning |
| ------------------------ | ------------ | ------------------------------------------ |
| `SEARCH_API_UNAVAILABLE` | Researcher | Web search API key missing or rate-limited |
| `JOB_NOT_FOUND` | MCP server | Polling an unknown `job_id` |
