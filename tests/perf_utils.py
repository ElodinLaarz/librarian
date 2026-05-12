from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import numpy as np
from numpy.typing import NDArray

from src.config import TidySettings
from src.models.enums import SourceType
from src.models.tome import Tome
from src.services.duplicate_detection import build_duplicate_groups
from src.storage.tome_repository import DuplicateScanResult, TomeRepository


def make_synthetic_tome(content: str, embedding: NDArray[np.floating[Any]]) -> Tome:
    return Tome(
        id=uuid.uuid4(),
        title=content[:40],
        content=content,
        summary=content[:80],
        category="perf",
        tags=["perf"],
        source_url=None,
        source_type=SourceType.AGENT_INPUT,
        confidence=0.8,
        embedding=embedding.astype(np.float32),
        created_at=datetime.now(UTC),
    )


def generate_tomes(
    *,
    total: int,
    duplicate_pairs: int,
    dimensions: int = 32,
) -> list[Tome]:
    rng = np.random.default_rng(7)
    tomes: list[Tome] = []

    for index in range(duplicate_pairs):
        base = rng.normal(size=dimensions).astype(np.float32)
        base /= np.linalg.norm(base)
        variant = base.copy()
        variant[0] += 0.01
        variant /= np.linalg.norm(variant)
        tomes.append(make_synthetic_tome(f"duplicate pair {index} source", base))
        tomes.append(make_synthetic_tome(f"duplicate pair {index} variant", variant))

    remaining = total - len(tomes)
    for index in range(remaining):
        vector = rng.normal(size=dimensions).astype(np.float32)
        vector /= np.linalg.norm(vector)
        tomes.append(make_synthetic_tome(f"unique tome {index}", vector))

    return tomes


@dataclass
class PerfTomeRepository(TomeRepository):
    tomes: list[Tome]
    tidy_settings: TidySettings

    async def insert(self, tome: Tome) -> UUID:
        self.tomes.append(tome)
        return tome.id

    async def delete(self, tome_id: UUID) -> bool:
        for index, tome in enumerate(self.tomes):
            if tome.id == tome_id:
                del self.tomes[index]
                return True
        return False

    async def update(
        self,
        tome_id: UUID,
        *,
        content: str | None = None,
        category: str | None = None,
        tags: list[str] | None = None,
        source_url: str | None = None,
        confidence: float | None = None,
    ) -> Tome | None:
        for index, tome in enumerate(self.tomes):
            if tome.id == tome_id:
                update_dict: dict[str, object] = {}
                if content is not None:
                    update_dict["content"] = content
                if category is not None:
                    update_dict["category"] = category
                if tags is not None:
                    update_dict["tags"] = tags
                if source_url is not None:
                    update_dict["source_url"] = source_url
                if confidence is not None:
                    update_dict["confidence"] = confidence
                if not update_dict:
                    return tome
                updated = tome.model_copy(update=update_dict)
                self.tomes[index] = updated
                return updated
        return None

    async def get_by_id(self, tome_id: UUID) -> Tome | None:
        for tome in self.tomes:
            if tome.id == tome_id:
                return tome
        return None

    async def search(
        self,
        query: str,
        top_k: int = 5,
        min_confidence: float = 0.5,
        category: str | None = None,
    ) -> list[tuple[Tome, float]]:
        del query
        return [
            (tome, 0.0)
            for tome in self.tomes
            if tome.confidence >= min_confidence and (category is None or tome.category == category)
        ][:top_k]

    async def find_near_duplicates(self, tome: Tome) -> list[Tome]:
        del tome
        return []

    async def find_all_near_duplicates(self, threshold: float = 0.95) -> DuplicateScanResult:
        settings = self.tidy_settings.model_copy(update={"threshold": threshold})
        return build_duplicate_groups(self.tomes, settings)

    async def list_all(
        self,
        limit: int = 100,
        offset: int = 0,
        *,
        category: str | None = None,
        min_confidence: float = 0.0,
        research_job_id: UUID | None = None,
    ) -> list[Tome]:
        filtered = [
            t
            for t in self.tomes
            if (category is None or t.category == category)
            and t.confidence >= min_confidence
            and (research_job_id is None or t.research_job_id == research_job_id)
        ]
        return filtered[offset : offset + limit]

    async def count(
        self,
        *,
        category: str | None = None,
        min_confidence: float = 0.0,
        research_job_id: UUID | None = None,
    ) -> int:
        return sum(
            1
            for t in self.tomes
            if (category is None or t.category == category)
            and t.confidence >= min_confidence
            and (research_job_id is None or t.research_job_id == research_job_id)
        )

    def close(self) -> None:
        self.tomes.clear()
