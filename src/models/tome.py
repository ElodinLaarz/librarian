from datetime import UTC, datetime
from uuid import UUID

import numpy as np
from pydantic import BaseModel, Field

from src.models.enums import SourceType
from numpydantic import NDArray, Shape


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
    embedding: NDArray[Shape["* x"], np.float32]
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
