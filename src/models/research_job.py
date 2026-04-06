from datetime import UTC, datetime
from uuid import UUID

from pydantic import BaseModel, Field

from src.models.enums import ResearchDepth, ResearchJobStatus


class ResearchJob(BaseModel):
    """Tracks a `library_research` run persisted in MongoDB."""

    id: UUID
    topic: str
    context: str | None = None
    depth: ResearchDepth = ResearchDepth.STANDARD
    max_tomes: int = 10
    category: str | None = None
    status: ResearchJobStatus = ResearchJobStatus.PENDING
    queries: list[str] = Field(default_factory=list)
    tome_ids: list[UUID] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    query_count: int = 0
    error: str | None = None
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None
