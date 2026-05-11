from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any
from uuid import UUID

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection
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
from src.storage.mongo.mongo_research_job import MongoResearchJob
from src.storage.research_job_repository import ResearchJobRepository


class MongoResearchJobRepository(ResearchJobRepository):
    def __init__(self, settings: DatabaseSettings) -> None:
        kwargs: dict[str, Any] = {"uuidRepresentation": "standard"}
        if settings.tls:
            kwargs["tls"] = True
            kwargs["tlsCertificateKeyFile"] = os.path.expanduser(settings.tls_cert_path)
        else:
            kwargs["tls"] = False

        self._client: AsyncIOMotorClient[Mapping[str, Any]] = AsyncIOMotorClient(
            settings.uri, **kwargs
        )
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
        return MongoResearchJob.model_validate(raw).to_domain()

    def close(self) -> None:
        self._client.close()
