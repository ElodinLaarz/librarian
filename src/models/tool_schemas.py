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


class SearchOutput(BaseModel):
    tomes: list[Tome]
    scores: list[float]
    query_id: str
    from_cache: bool


# ── library.ingest ──────────────────────────────────────────────────


class IngestInput(BaseModel):
    content: str = Field(..., max_length=500_000)
    skip_verify: bool = False
    category: str | None = None
    tags: list[str] | None = None
    source_url: str | None = None
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
