from pydantic import BaseModel, Field

from src.models.enums import IngestStatus
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


class IngestOutput(BaseModel):
    tomes: list[Tome]
    status: IngestStatus
    reject_reason: str | None = None
