from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class ResearchJob(BaseModel):
    """Tracks the state and output of a Researcher dispatch."""

    id: str = Field(default_factory=lambda: uuid4().hex)
    topic: str
    context: str | None = None
    status: Literal["pending", "running", "completed", "failed"] = "pending"
    queries: list[str] = Field(default_factory=list)
    tome_ids: list[str] = Field(default_factory=list)
    error: str | None = None
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None
