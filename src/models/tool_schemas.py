from __future__ import annotations

from pydantic import BaseModel, Field

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
    content: str
    title: str | None = None
    category: str | None = None
    tags: list[str] = Field(default_factory=list)
    source_url: str | None = None
    skip_verify: bool = False
    allow_update: bool = True


class IngestOutput(BaseModel):
    tome_ids: list[str]
    tomes: list[Tome]
    confidence: float
    chunks: int
    status: Literal["stored", "rejected", "partial"]
    reject_reason: str | None = None


# ── library.research ────────────────────────────────────────────────

class ResearchInput(BaseModel):
    topic: str
    context: str | None = None
    depth: Literal["shallow", "standard", "deep"] = Field(default="standard")
    max_tomes: int = Field(default=10, ge=1)
    category: str | None = None
    async_mode: bool = Field(default=False, alias="async")


class ResearchOutput(BaseModel):
    job_id: str
    tome_ids: list[str]
    tomes: list[Tome]
    sources: list[str]
    query_count: int
    status: str  # "completed" | "failed"
