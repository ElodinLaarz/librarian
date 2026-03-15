from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any
from uuid import UUID

import numpy as np
from bson.binary import BinaryVector, BinaryVectorDtype
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection

from src.config import DatabaseSettings
from src.models.tome import Tome
from src.services.embedding import EmbeddingService
from src.storage.mongo.mongo_tome import MongoTome
from src.storage.tome_repository import TomeRepository


class MongoTomeRepository(TomeRepository):
    """MongoDB implementation of the TomeRepository using Atlas Search.

    This implementation expects a MongoDB Atlas cluster with a Search index
    configured to support both vector and lexical search.
    """

    def __init__(self, settings: DatabaseSettings, embedding_service: EmbeddingService) -> None:
        srv = f"mongodb+srv://{settings.uri}/?authSource=%24external&authMechanism=MONGODB-X509"
        cert = os.path.expanduser(settings.tls_cert_path)
        self._client: AsyncIOMotorClient[Mapping[str, Any]] = AsyncIOMotorClient(
            srv, tls=True, tlsCertificateKeyFile=cert
        )
        self._embedding_service = embedding_service
        self._db = self._client[settings.database]
        self._collection: AsyncIOMotorCollection[Mapping[str, Any]] = self._db[
            settings.tomes_collection
        ]

    async def insert(self, tome: Tome) -> UUID:
        """Insert a new Tome into MongoDB."""
        mongo_tome = MongoTome.from_tome(tome)
        await self._collection.insert_one(mongo_tome.model_dump(by_alias=True))
        return tome.id

    async def delete(self, tome_id: UUID) -> bool:
        """Permanently remove a Tome by ID."""
        result = await self._collection.delete_one({"_id": tome_id})
        return result.deleted_count > 0

    async def get_by_id(self, tome_id: UUID) -> Tome | None:
        """Retrieve a single Tome by its ID."""
        doc = await self._collection.find_one({"_id": tome_id})
        if not doc:
            return None
        return MongoTome.model_validate(doc).to_tome()

    async def search(
        self,
        query: str,
        top_k: int = 5,
        min_confidence: float = 0.5,
    ) -> list[tuple[Tome, float]]:
        """Perform hybrid search using Atlas Search (lexical) and Vector Search.

        Combines results from two separate search pipelines using average score ($scoreFusion).
        """
        query_embedding = await self._embedding_service.embed(query)
        query_vector = BinaryVector(
            np.array(query_embedding, dtype=np.float32).tolist(), BinaryVectorDtype.FLOAT32
        )

        pipeline: list[Mapping[str, Any]] = [
            {
                "$scoreFusion": {
                    "input": {
                        "pipelines": {
                            "lexical": [
                                {
                                    "$search": {
                                        "compound": {
                                            "filter": [
                                                {
                                                    "range": {
                                                        "path": "confidence",
                                                        "gte": min_confidence,
                                                    }
                                                }
                                            ],
                                            "should": [
                                                {
                                                    "text": {
                                                        "query": query,
                                                        "path": [
                                                            "title",
                                                            "content",
                                                            "summary",
                                                            "tags",
                                                        ],
                                                    }
                                                }
                                            ],
                                        }
                                    }
                                },
                                {"$limit": top_k * 10},
                            ],
                            "semantic": [
                                {
                                    "$vectorSearch": {
                                        "index": "vectors",
                                        "path": "embedding",
                                        "queryVector": query_vector,
                                        "filter": {"confidence": {"$gte": min_confidence}},
                                        "numCandidates": top_k * 10,
                                        "limit": top_k * 2,
                                    }
                                }
                            ],
                        },
                        "normalization": "sigmoid",
                    },
                    "combination": {"method": "avg"},
                }
            },
            {"$limit": top_k},
            {"$project": {"score": {"$meta": "searchScore"}, "document": "$$ROOT"}},
        ]

        results = []
        async for doc in self._collection.aggregate(pipeline):
            tome = MongoTome.model_validate(doc["document"]).to_tome()
            score = doc["score"]
            results.append((tome, score))

        return results

    async def find_near_duplicates(self, tome: Tome, threshold: float = 0.3) -> list[Tome]:
        """Find existing Tomes with cosine similarity above the threshold using $vectorSearch."""
        query_vector = BinaryVector(
            np.asarray(tome.embedding, dtype=np.float32).tolist(), BinaryVectorDtype.FLOAT32
        )

        pipeline: list[Mapping[str, Any]] = [
            {
                "$vectorSearch": {
                    "index": "vectors",
                    "path": "embedding",
                    "queryVector": query_vector,
                    "numCandidates": 100,
                    "limit": 10,
                }
            },
            {"$project": {"score": {"$meta": "vectorSearchScore"}, "document": "$$ROOT"}},
            {"$match": {"score": {"$gte": threshold}, "document._id": {"$ne": tome.id}}},
        ]

        duplicates = []
        async for doc in self._collection.aggregate(pipeline):
            duplicates.append(MongoTome.model_validate(doc["document"]).to_tome())

        return duplicates

    async def ensure_indexes(self) -> None:
        """Create standard MongoDB indexes."""
        ...
