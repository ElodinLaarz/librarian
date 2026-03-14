from datetime import datetime
from uuid import uuid4

from pydantic import BaseModel, Field

from src.models.enums import SourceType


class Tome(BaseModel):
    """A compact, single-topic knowledge document stored in the library."""

    id: str = Field(default_factory=lambda: uuid4().hex)
    title: str = Field(..., max_length=120)
    content: str
    summary: str
    category: str
    tags: list[str] = Field(default_factory=list)
    source_url: str | None = None
    source_type: SourceType
    confidence: float = Field(..., ge=0.0, le=1.0)
    embedding: list[float]
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    version: int = 1
    superseded_by: str | None = None
    research_job: str | None = None
