# The Librarian — Low-Level Design

### API Contracts & User Journeys

| | |
| ----------- | ------------ |
| **Version** | 0.3 (Draft) |
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
└──────────┬───────────────────────┘
           │
           ├── library_search ──► TomeRepository.search(query)
           │
           └── library_ingest ──► Ingestor
                                    ├── EmbeddingService
                                    ├── Verifier
                                    └── TomeRepository
                                          │
                                          ▼
                                  ┌───────────────────┐
                                  │  FsTomeRepository  │
                                  │  ~/.librarian_mcp/ │
                                  │  tomes/<uuid>.json │
                                  └───────────────────┘
```

Search is routed directly to `TomeRepository.search`; no separate search
service. The concrete storage implementation is `FsTomeRepository` (filesystem
JSON). A MongoDB implementation is planned for a later phase.

______________________________________________________________________

## 2. API 1 — MCP Tools

Both tools follow the standard MCP tool-call envelope. Errors use the envelope
in [§6 Error Taxonomy](#6-error-taxonomy).

______________________________________________________________________

### 2.1 `library_search`

Retrieves the most semantically relevant Tomes for a natural-language query.

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
| ------------ | --------- | ---------------------------------------------------------------- |
| `tomes` | `Tome[]` | Matching Tome documents, sorted by score descending |
| `scores` | `float[]` | Similarity scores corresponding to each Tome (parallel array) |
| `query_id` | `str` | Unique identifier for this search request |
| `from_cache` | `bool` | Whether the query embedding was served from the LRU cache |

#### Errors

| Code | When raised |
| ------------------- | -------------------------------------------- |
| `EMBED_UNAVAILABLE` | Embedding model unreachable |
| `NO_RESULTS` | Zero tomes exceed `min_confidence` threshold |

______________________________________________________________________

### 2.2 `library_ingest`

Validates, embeds, deduplicates, and stores raw text as one or more Tomes.

#### Input — `IngestInput`

| Field | Type | Required | Notes |
| --------- | ----- | -------- | ------ |
| `content` | `str` | Yes | Raw knowledge text |

#### Output — `IngestOutput`

| Field | Type | Description |
| --------------- | -------------- | ------------------------------------------------------- |
| `tomes` | `Tome[]` | Full Tome objects as stored |
| `status` | `IngestStatus` | One of: `stored`, `rejected`, `partial` |
| `reject_reason` | `str\|null` | Rejection explanation; non-null when `status=rejected` |

#### Errors

| Code | When raised |
| ------------------- | ------------------------------------------ |
| `VERIFY_FAILED` | Confidence below reject threshold (< 0.3) |
| `EMBED_UNAVAILABLE` | Embedding model unreachable |

______________________________________________________________________

### 2.3 Shared Type — `Tome`

Source: `src/models/tome.py`

| Field | Type | Description |
| ------------- | -------------------- | ---------------------------------------- |
| `id` | `UUID` | Auto-generated UUID |
| `title` | `str` | Short descriptive title (max 120 chars) |
| `content` | `str` | Full text body |
| `summary` | `str` | One-to-two sentence summary |
| `category` | `str` | High-level domain category |
| `tags` | `str[]` | Freeform topic tags |
| `source_url` | `str\|null` | Origin URL if sourced from the web |
| `source_type` | `SourceType` | `agent_input`, `researcher`, or `manual` |
| `confidence` | `float` | Truthiness confidence score (0.0–1.0) |
| `embedding` | `NDArray[np.float32]` | Dense vector embedding of the content |
| `created_at` | `datetime` | UTC timestamp of creation |

______________________________________________________________________

## 3. Service Layer

______________________________________________________________________

### 3.1 `EmbeddingService`

Source: `src/services/embedding.py`

Abstract base; concrete implementations cover Ollama, sentence-transformers,
OpenAI. Used by `Ingestor` to embed content before storage and dedup.

| Method | Signature | Returns | Notes |
| ------------ | ------------- | ----------- | ------------------------------------------ |
| `initialize` | `()` | `None` | Load model; warm up provider connection |
| `embed` | `(text: str)` | `np.ndarray` | Single text; returns from LRU cache if hit |

Cache is an in-memory LRU keyed on SHA-256 of input text. Cache size is
configurable (`embedding.cache_size`). Cold on restart.

______________________________________________________________________

### 3.2 `Ingestor`

Source: `src/services/ingestor.py`

Dependencies: `LibrarianConfig`, `EmbeddingService`, `Verifier`,
`TomeRepository`

| Method | Signature | Returns | Notes |
| ----------------------------- | ----------------------------------------- | ----------- | ---------------------------------------------------------------- |
| `ingest` | `(blob: str)` | `list[Tome]` | Full pipeline: classify+summarize+embed (concurrent) → validate → dedup → store |
| `_validate` | `(tome: Tome)` | `None` | Post-construction checks; raises on failure |
| `_classify_and_tag` | `(chunk: str, category_hint: str\|None)` | `(str, list[str])` | Returns `(category, tags)` |
| `_generate_title_and_summary` | `(chunk: str)` | `(str, str)` | Returns `(title, summary)` |
| `_dedup_and_store` | `(tome: Tome)` | `list[UUID]` | Near-dup check → reshard or insert; returns stored Tome IDs |

**Pipeline:**

1. `_classify_and_tag`, `_generate_title_and_summary`, and
   `EmbeddingService.embed` run **concurrently** via `asyncio.gather`
1. A `Tome` object is constructed from the results
1. `_validate` — post-construction checks on the assembled Tome
1. `Verifier.verify` — reject if `confidence < reject_threshold`
1. `_dedup_and_store`:
   - Call `TomeRepository.find_near_duplicates(tome)`
   - **No duplicates** → `TomeRepository.insert(tome)`
   - **Duplicates found** → reshard: combine new content with all overlapping
     Tomes' content, re-run the pipeline on the combined text, delete old
     Tomes, insert fresh ones

> **Note:** Chunking of long inputs into multiple atomic facts is handled
> within `_dedup_and_store`'s reshard path. A single input blob always enters
> the pipeline as one unit; resharding is what produces multiple Tomes when
> content overlaps with existing knowledge.

______________________________________________________________________

### 3.3 `Verifier`

Source: `src/services/verifier.py`

Dependencies: `VerificationSettings`, `WebSearchClient`

| Method | Signature | Returns | Notes |
| ----------------------- | ------------------------------ | -------------------- | ---------------------------------------------------- |
| `verify` | `(content: str)` | `VerificationResult` | Full pipeline; returns offline result if unavailable |
| `_extract_claims` | `(content: str)` | `list[str]` | 3–7 claims via zero-shot prompt |
| `_check_claim` | `(claim: str)` | `ClaimResult` | Web search + snippet scoring per claim |
| `_aggregate_confidence` | `(results: list[ClaimResult])` | `float` | Weighted score across all claims |
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
| ----------- | ------------------------------------- |
| ≥ 0.7 | Store with full confidence |
| 0.3 – 0.7 | Store with low-confidence flag |
| < 0.3 | Reject; raise `VERIFY_FAILED` |

When no search API key is configured, `_make_offline_result` returns
`confidence=0.6` and `skipped=True`.

______________________________________________________________________

## 4. API 2 — Repository Layer

Source: `src/storage/`. Abstract base class with a concrete filesystem
implementation. Services are the only callers.

______________________________________________________________________

### 4.1 `TomeRepository`

Source: `src/storage/tome_repository.py`

| Method | Signature | Returns | Notes |
| --------------------- | ------------------------------------------------- | -------------------------- | --------------------------------------------------- |
| `insert` | `(tome: Tome)` | `UUID` | Inserts a new Tome; returns its ID |
| `delete` | `(tome_id: UUID)` | `bool` | Removes a Tome; `True` if found and deleted |
| `get_by_id` | `(tome_id: UUID)` | `Tome \| None` | |
| `search` | `(query: str, top_k: int, min_confidence: float)` | `list[tuple[Tome, float]]` | Full-text or vector search; sorted by score |
| `find_near_duplicates` | `(tome: Tome)` | `list[Tome]` | Tomes with high embedding similarity to the input |

______________________________________________________________________

### 4.2 `FsTomeRepository`

Source: `src/storage/filesystem/fs_tome_repository.py`

Current concrete implementation. Stores each Tome as a JSON file under
`~/.librarian_mcp/tomes/<uuid>.json` (path configurable via
`DatabaseSettings.uri`).

| Method | Behaviour |
| --------------------- | ----------------------------------------------------------------- |
| `insert` | Writes `<uuid>.json` via `Tome.model_dump_json()` |
| `delete` | Unlinks `<uuid>.json`; returns `False` if file not found |
| `get_by_id` | Reads and deserialises `<uuid>.json` |
| `search` | Brute-force scan of all `.json` files; filters by `min_confidence`; placeholder similarity score |
| `find_near_duplicates` | Scans all files; placeholder comparison by title equality |

> **Note:** `search` and `find_near_duplicates` in `FsTomeRepository` use
> placeholder logic. A production implementation (MongoDB or similar) will
> replace these with proper vector similarity search against stored embeddings.

______________________________________________________________________

## 5. User Journeys

______________________________________________________________________

### Journey A — Search (results found)

```
Agent → library_search(query="how does X work", top_k=5)
  server:
    tome_repo.search("how does X work", top_k=5, min_confidence=0.5)
      FsTomeRepository: scan *.json, filter confidence ≥ 0.5, return top 5
← Agent ← SearchOutput{
    tomes: [t1, t2, t3],
    scores: [0.93, 0.88, 0.82],
    query_id: "q1",
    from_cache: false
  }
```

______________________________________________________________________

### Journey B — Search (no results)

```
Agent → library_search(query="obscure topic")
  server:
    tome_repo.search("obscure topic", top_k=5, min_confidence=0.5)
      FsTomeRepository: scan *.json → [] (nothing above threshold)
← Agent ← SearchOutput{tomes: [], scores: [], query_id: "q2", from_cache: false}
```

The agent may call `library_ingest` with curated content on the topic, or
`library_research` (v2) to auto-populate.

______________________________________________________________________

### Journey C — Ingest (new content, no duplicates)

```
Agent → library_ingest(content="<article covering a single fact>")
  Ingestor.ingest(blob):
    concurrently:
      _classify_and_tag(blob)        → ("science", ["tag_a", "tag_b"])
      _generate_title_and_summary(blob) → ("Title A", "Summary A")
      EmbeddingService.embed(blob)   → embedding (NDArray[float32])
    Tome{id: uuid4(), content, title, summary, category, tags, embedding, ...}
    _validate(tome) → OK
    Verifier.verify(blob):
      _extract_claims() → 4 claims
      _check_claim() × 4 → [supported, supported, unverifiable, supported]
      _aggregate_confidence() → 0.82
    0.82 ≥ 0.3 → proceed
    _dedup_and_store(tome):
      tome_repo.find_near_duplicates(tome) → []
      tome_repo.insert(tome) → uuid_t1
← Agent ← IngestOutput{
    tomes: [Tome_t1],
    status: "stored",
    reject_reason: null
  }
```

______________________________________________________________________

### Journey D — Ingest (duplicate → reshard)

New content overlaps with an existing Tome. The old Tome is deleted and the
combined knowledge is re-ingested as fresh atomic facts.

```
Agent → library_ingest(content="updated fact about X with new detail")
  Ingestor.ingest(blob):
    concurrently:
      _classify_and_tag(blob)           → ("science", ["tag_a"])
      _generate_title_and_summary(blob) → ("Title A v2", "Summary A v2")
      EmbeddingService.embed(blob)      → embedding_new
    Tome{id: uuid_new, ...}
    _validate(tome) → OK
    Verifier.verify(blob) → VerificationResult{confidence: 0.75}
    _dedup_and_store(tome):
      tome_repo.find_near_duplicates(tome)
        → [Tome{id: uuid_t1, content: "original fact about X"}]

      reshard:
        combined = "original fact about X\n\nupdated fact about X with new detail"
        ingest(combined) [recursive, internal]:
          → Tome_ra ("Resharded fact A") → uuid_ra
          → Tome_rb ("Resharded fact B") → uuid_rb
        tome_repo.delete(uuid_t1)  ← old Tome permanently removed

← Agent ← IngestOutput{
    tomes: [Tome_ra, Tome_rb],
    status: "stored",
    reject_reason: null
  }
```

______________________________________________________________________

### Journey E — Ingest (rejected by verifier)

```
Agent → library_ingest(content="The moon is made of cheese...")
  Ingestor.ingest(blob):
    concurrently: classify, summarize, embed
    _validate(tome) → OK
    Verifier.verify(blob):
      _extract_claims() → 3 claims
      _check_claim() × 3 → [contradicted, contradicted, contradicted]
      _aggregate_confidence() → 0.11
    0.11 < 0.3 → reject
← Agent ← IngestOutput{
    tomes: [],
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
| ------------------- | --------------- | --------------------------------------------------------------- |
| `EMBED_UNAVAILABLE` | Ingestor | Embedding model unreachable |
| `VERIFY_FAILED` | Verifier | Confidence below `reject_threshold` (< 0.3) |
| `NO_RESULTS` | MCP server | Zero results above `min_confidence` threshold |
| `DB_UNAVAILABLE` | TomeRepository | Storage backend unreachable or unreadable |
| `EMBEDDING_MISMATCH` | TomeRepository | Stored vector dimensions don't match the configured model |

`details` is an optional free-form object; its structure is not stable across
versions.

______________________________________________________________________

## 7. Beyond v1 — Research Tool

`library_research` and its supporting infrastructure are deferred to v2. The
contracts below are provisional; they will be fleshed out once the core search
and ingest tools are complete and stable.

______________________________________________________________________

### 7.1 `library_research` (provisional)

Dispatches a Researcher to search the web, synthesise findings, and pipe
results through the ingest pipeline as new Tomes.

#### Input — `ResearchInput`

| Field | Type | Required | Default | Notes |
| ----------- | --------------- | -------- | ---------- | -------------------------------- |
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
| ------------- | ----------- | ------------------------------------------- |
| `job_id` | `str` | ID of the ResearchJob record |
| `tome_ids` | `UUID[]` | IDs of Tomes created; empty while in-flight |
| `tomes` | `Tome[]` | Full Tome objects; empty while in-flight |
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
