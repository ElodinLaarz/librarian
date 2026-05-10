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
    force_format: (
        Literal["text", "code", "python", "yaml", "json", "markdown"] | None
    ) = Field(
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
