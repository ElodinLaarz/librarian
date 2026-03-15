from datetime import UTC, datetime
from uuid import UUID

from pydantic import BaseModel, Field

from src.models.enums import SourceType


class Tome(BaseModel):
    """A compact, single-topic knowledge document stored in the library."""

    id: UUID
    title: str = Field(..., max_length=120)
    content: str
    summary: str
    category: str
    tags: list[str] = Field(default_factory=list)
    source_url: str | None = None
    source_type: SourceType
    confidence: float = Field(..., ge=0.0, le=1.0)
    embedding: list[float] # This might become some numpy thing later.
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
