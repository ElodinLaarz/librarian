from datetime import UTC, datetime
from typing import Annotated
from uuid import UUID

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator
from pydantic.json_schema import SkipJsonSchema

from src import constants
from src.models.enums import SourceType


class Tome(BaseModel):
    """A compact, single-topic knowledge document stored in the library.
    This is the data model understood by "The Librarian" and public interfaces, although
    the actual storage format may be different."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: UUID
    title: str = Field(..., max_length=constants.TITLE_MAX_LENGTH)
    content: str
    summary: str
    category: str
    tags: list[str] = Field(default_factory=list)
    source_url: str | None = None
    source_type: SourceType
    confidence: float = Field(..., ge=0.0, le=1.0)

    embedding: Annotated[NDArray[np.float32], SkipJsonSchema()] = Field(exclude=True)

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_serializer("embedding")
    def serialize_embedding(self, v: NDArray[np.float32]) -> list[float]:
        return v.tolist()  # type: ignore[no-any-return]

    @field_validator("embedding", mode="before")
    @classmethod
    def validate_embedding(cls, v: object) -> NDArray[np.float32]:
        if isinstance(v, list):
            return np.array(v, dtype=np.float32)
        if not isinstance(v, np.ndarray):
            raise TypeError("embedding must be a numpy.ndarray or list of floats")
        if v.dtype != np.float32:
            raise ValueError("embedding must have dtype float32")
        return v
