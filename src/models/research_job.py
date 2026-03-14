from datetime import datetime
from uuid import uuid4

from pydantic import BaseModel, Field

from src.models.enums import JobStatus


class ResearchJob(BaseModel):
    """Tracks the state and output of a Researcher dispatch."""

    id: str = Field(default_factory=lambda: uuid4().hex)
    topic: str
    context: str | None = None
    status: JobStatus = JobStatus.PENDING
    queries: list[str] = Field(default_factory=list)
    tome_ids: list[str] = Field(default_factory=list)
    error: str | None = None
    started_at: datetime = Field(default_factory=datetime.utcnow)
    finished_at: datetime | None = None
