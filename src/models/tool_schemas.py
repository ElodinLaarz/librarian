from typing import Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from src.models.enums import IngestStatus, ResearchDepth
from src.models.tome import Tome

# ── library.search ──────────────────────────────────────────────────


class SearchInput(BaseModel):
    query: str = Field(..., max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    category: str | None = None
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    include_summary: bool = False
    # Payload-shaping knobs (issue #48). Defaults preserve backward compatibility:
    # full content is returned unless the caller (or the server's
    # ``search.default_content_max_chars`` / ``search.default_snippet_chars``)
    # opts in.
    include_content: bool = Field(
        default=True,
        description=(
            "When False, ``content`` is replaced with an empty string. Use this "
            "to retrieve only metadata; full content can still be fetched later "
            "using the returned ``tome_ids``."
        ),
    )
    content_max_chars: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Truncate each hit's ``content`` to this many characters and append "
            "``...`` when cut. Overrides ``search.default_content_max_chars``."
        ),
    )
    snippet_chars: int | None = Field(
        default=None,
        ge=0,
        description=(
            "If > 0, return only a window of this many characters around the "
            "first lexical match of any query term (falls back to the leading "
            "window when no term matches). Overrides "
            "``search.default_snippet_chars``."
        ),
    )


class SearchOutput(BaseModel):
    tomes: list[Tome]
    scores: list[float]
    query_id: str
    from_cache: bool
    # 1:1 with ``tomes``. Exposed so that callers can request the full content
    # of an individual hit (via a future ``library_get`` tool) when the
    # returned snippet/truncated payload is insufficient.
    tome_ids: list[str] = Field(default_factory=list)
    # 1:1 with ``tomes``. ``True`` when the corresponding tome's ``content``
    # was truncated, snippeted, or dropped — i.e. the payload is no longer the
    # full stored content.
    is_snippet: list[bool] = Field(default_factory=list)


# ── library.ingest ──────────────────────────────────────────────────


class IngestInput(BaseModel):
    content: str = Field(..., max_length=500_000)
    skip_verify: bool = False
    category: str | None = None
    tags: list[str] | None = None
    source_url: str | None = None
    thread_id: str | None = Field(
        default=None,
        description="Thread UUID (hex string) for grouping related tomes in conversation.",
    )
    force_format: Literal["text", "code", "python", "yaml", "json", "markdown"] | None = Field(
        default=None,
        description=(
            "Override automatic format detection. When set, the chunker uses a "
            "syntax-aware splitter for the given format instead of the default "
            "LLM/recursive text splitter."
        ),
    )


class IngestOutput(BaseModel):
    tomes: list[Tome]
    status: IngestStatus
    reject_reason: str | None = None


# ── library.research ────────────────────────────────────────────────


class ResearchInput(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    job_id: str | None = Field(
        default=None,
        description="When set, return status/results for this job instead of starting a new run.",
    )
    topic: str | None = Field(default=None, max_length=2000)
    context: str | None = Field(default=None, max_length=8000)
    depth: ResearchDepth = ResearchDepth.STANDARD
    max_tomes: int = Field(default=10, ge=1, le=50)
    category: str | None = None
    async_: bool = Field(
        default=False,
        validation_alias=AliasChoices("async", "async_"),
    )


class ResearchOutput(BaseModel):
    job_id: str
    status: str
    tome_ids: list[str] = Field(default_factory=list)
    tomes: list[Tome] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    query_count: int = 0
    error: str | None = None


# ── library.get ─────────────────────────────────────────────────────


class GetInput(BaseModel):
    tome_id: str = Field(
        ...,
        max_length=200,
        description=(
            "UUID string identifying the Tome to retrieve. Accepts the "
            "canonical hyphenated form (e.g. as returned by library_search) "
            "as well as bare hex."
        ),
    )


class GetOutput(BaseModel):
    tome: Tome | None = None
    status: Literal["found", "not_found", "invalid_tome_id"]
    error: str | None = None


# ── library.update ──────────────────────────────────────────────────


class UpdateInput(BaseModel):
    tome_id: str = Field(
        ...,
        max_length=200,
        description=(
            "UUID string identifying the Tome to update. Accepts the "
            "canonical hyphenated form (e.g. as returned by library_search) "
            "as well as bare hex."
        ),
    )
    content: str | None = Field(default=None, max_length=500_000)
    category: str | None = None
    tags: list[str] | None = None
    source_url: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class UpdateOutput(BaseModel):
    tome: Tome | None = None
    status: Literal["updated", "not_found", "invalid_id"]
    error: str | None = None


# ── library.tidy ──────────────────────────────────────────────────


class TidyInput(BaseModel):
    limit: int = Field(default=1000, ge=1, le=10000, description="Max groups to process.")
    threshold: float = Field(
        default=0.95,
        ge=0.5,
        le=1.0,
        description="Cosine similarity threshold for semantic duplicate checks.",
    )
    skip_verify: bool = Field(
        default=True,
        description="Whether to skip verification when rebuilding consolidated tomes.",
    )


class TidyOutput(BaseModel):
    scanned: int
    groups_found: int
    groups_consolidated: int
    tomes_removed: int
    failed_groups: int
    skipped_groups: int
    elapsed_ms: int


# ── library.delete ──────────────────────────────────────────────────


class DeleteInput(BaseModel):
    tome_id: str = Field(
        ...,
        description="UUID string (hex or hyphenated) identifying the tome to remove.",
        max_length=64,
    )


class DeleteOutput(BaseModel):
    deleted: bool
    error: str | None = None


# ── library.list ────────────────────────────────────────────────────


class ListInput(BaseModel):
    """Filters + pagination for ``library_list``.

    ``research_job_id`` and ``thread_id`` are strings for MCP-client convenience
    (clients can pass the value they got back from ``library_research`` or
    ``library_ingest`` without round-tripping through UUID parsing); invalid
    values produce a clear error at the tool boundary.
    """

    category: str | None = None
    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    research_job_id: str | None = Field(
        default=None,
        description="Filter by research job. Must be a UUID hex string.",
    )
    thread_id: str | None = Field(
        default=None,
        description="Filter by thread. Must be a UUID hex string.",
    )
    limit: int = Field(default=50, ge=1, le=500)
    offset: int = Field(default=0, ge=0)
    include_summary: bool = True


class ListOutput(BaseModel):
    """A page of tomes plus the total number matching the filter.

    ``total`` is the count *before* limit/offset are applied so clients can
    compute the number of pages remaining. ``has_more`` is a convenience flag
    derived from ``offset + len(tomes) < total``.
    """

    tomes: list[Tome]
    total: int
    has_more: bool
