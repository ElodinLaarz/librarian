from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from uuid import UUID

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection
from pydantic import ValidationError
from pymongo.errors import (
    ConnectionFailure,
    DuplicateKeyError,
    PyMongoError,
    ServerSelectionTimeoutError,
)

from src.config import DatabaseSettings
from src.models.research_job import ResearchJob
from src.storage.errors import (
    BackendUnavailableError,
    DuplicateError,
    NotFoundError,
    StorageError,
)
from src.storage.mongo.client import build_motor_client
from src.storage.mongo.mongo_research_job import MongoResearchJob
from src.storage.research_job_repository import ResearchJobRepository


class MongoResearchJobRepository(ResearchJobRepository):
    def __init__(
        self,
        settings: DatabaseSettings,
        *,
        client: AsyncIOMotorClient[Mapping[str, Any]] | None = None,
        owns_client: bool | None = None,
    ) -> None:
        """Create a research-job repo against the given Mongo database.

        See :class:`MongoTomeRepository.__init__` for ``client`` / ``owns_client``
        semantics. The lifespan-owned shared client is passed in via ``client``
        so all Mongo repos share a single connection pool (issue #25).
        """
        self._client: AsyncIOMotorClient[Mapping[str, Any]] = (
            client if client is not None else build_motor_client(settings)
        )
        # See ``MongoTomeRepository.__init__`` for ownership semantics.
        self._owns_client = owns_client if owns_client is not None else (client is None)

        db = self._client.get_database(settings.database)
        self._collection: AsyncIOMotorCollection[Mapping[str, Any]] = db[settings.jobs_collection]

    async def insert(self, job: ResearchJob) -> UUID:
        doc = MongoResearchJob.from_domain(job).model_dump(by_alias=True)
        try:
            await self._collection.insert_one(doc)
        except DuplicateKeyError as exc:
            raise DuplicateError(f"Research job {job.id} already exists") from exc
        except (ServerSelectionTimeoutError, ConnectionFailure) as exc:
            raise BackendUnavailableError("Mongo backend unavailable") from exc
        except PyMongoError as exc:
            raise StorageError(f"Mongo insert failed for job {job.id}") from exc
        return job.id

    async def update(self, job: ResearchJob) -> None:
        doc = MongoResearchJob.from_domain(job).model_dump(by_alias=True, exclude={"id"})
        try:
            result = await self._collection.update_one({"_id": job.id}, {"$set": doc})
        except (ServerSelectionTimeoutError, ConnectionFailure) as exc:
            raise BackendUnavailableError("Mongo backend unavailable") from exc
        except PyMongoError as exc:
            raise StorageError(f"Mongo update failed for job {job.id}") from exc
        if result.matched_count == 0:
            raise NotFoundError(f"Research job {job.id} does not exist")

    async def get_by_id(self, job_id: UUID) -> ResearchJob | None:
        try:
            raw = await self._collection.find_one({"_id": job_id})
        except (ServerSelectionTimeoutError, ConnectionFailure) as exc:
            raise BackendUnavailableError("Mongo backend unavailable") from exc
        except PyMongoError as exc:
            raise StorageError(f"Mongo find_one failed for job {job_id}") from exc
        if not raw:
            return None
        try:
            return MongoResearchJob.model_validate(raw).to_domain()
        except ValidationError as exc:
            # Corrupt/legacy job documents are a storage problem; services
            # must see StorageError, never pydantic exception types.
            raise StorageError(f"Corrupt research-job document (_id={raw.get('_id')!r})") from exc

    def close(self) -> None:
        """Close the MongoDB client if this repo owns it (see MongoTomeRepository.close)."""
        if self._owns_client:
            self._client.close()
