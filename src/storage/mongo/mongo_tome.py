from __future__ import annotations

from datetime import datetime
from uuid import UUID

import numpy as np
from bson.binary import Binary, BinaryVector, BinaryVectorDtype
from pydantic import BaseModel, ConfigDict, Field

from src.models.enums import SourceType
from src.models.tome import Tome


class MongoTome(BaseModel):
    """Pydantic model for MongoDB storage of a Tome.

    Handles mapping between Tome.id <-> MongoTome._id and
    numpy.ndarray <-> BSON Binary (subtype 16).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

    id: UUID = Field(alias="_id")
    title: str
    content: str
    summary: str
    category: str
    tags: list[str] = Field(default_factory=list)
    source_url: str | None = None
    source_type: SourceType
    confidence: float
    embedding: Binary | BinaryVector
    created_at: datetime

    @classmethod
    def from_tome(cls, tome: Tome) -> MongoTome:
        """Convert a domain Tome to a MongoTome."""
        return cls(
            _id=tome.id,
            title=tome.title,
            content=tome.content,
            summary=tome.summary,
            category=tome.category,
            tags=tome.tags,
            source_url=tome.source_url,
            source_type=tome.source_type,
            confidence=tome.confidence,
            embedding=Binary.from_vector(
                tome.embedding.astype(np.float32).tolist(), BinaryVectorDtype.FLOAT32
            ),
            created_at=tome.created_at,
        )

    def to_tome(self) -> Tome:
        """Convert a MongoTome back to a domain Tome."""
        return Tome(
            id=self.id,
            title=self.title,
            content=self.content,
            summary=self.summary,
            category=self.category,
            tags=self.tags,
            source_url=self.source_url,
            source_type=self.source_type,
            confidence=self.confidence,
            embedding=np.array(
                self.embedding.as_vector().data
                if isinstance(self.embedding, Binary)
                else self.embedding.data,
                dtype=np.float32,
            ),
            created_at=self.created_at,
        )
