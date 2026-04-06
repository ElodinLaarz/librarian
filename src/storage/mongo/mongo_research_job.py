from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from src.models.enums import ResearchDepth, ResearchJobStatus
from src.models.research_job import ResearchJob


class MongoResearchJob(BaseModel):
    """BSON-shaped document for the research_jobs collection."""

    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

    id: UUID = Field(alias="_id")
    topic: str
    context: str | None = None
    depth: ResearchDepth = ResearchDepth.STANDARD
    max_tomes: int = 10
    category: str | None = None
    status: ResearchJobStatus
    queries: list[str] = Field(default_factory=list)
    tome_ids: list[UUID] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    query_count: int = 0
    error: str | None = None
    started_at: datetime
    finished_at: datetime | None = None

    @classmethod
    def from_domain(cls, job: ResearchJob) -> MongoResearchJob:
        return cls(
            _id=job.id,
            topic=job.topic,
            context=job.context,
            depth=job.depth,
            max_tomes=job.max_tomes,
            category=job.category,
            status=job.status,
            queries=job.queries,
            tome_ids=job.tome_ids,
            sources=job.sources,
            query_count=job.query_count,
            error=job.error,
            started_at=job.started_at,
            finished_at=job.finished_at,
        )

    def to_domain(self) -> ResearchJob:
        return ResearchJob(
            id=self.id,
            topic=self.topic,
            context=self.context,
            depth=self.depth,
            max_tomes=self.max_tomes,
            category=self.category,
            status=self.status,
            queries=self.queries,
            tome_ids=self.tome_ids,
            sources=self.sources,
            query_count=self.query_count,
            error=self.error,
            started_at=self.started_at,
            finished_at=self.finished_at,
        )
