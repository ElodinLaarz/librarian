from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping
from typing import Any
from uuid import UUID

import numpy as np
from bson.binary import Binary, BinaryVectorDtype
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
        cert = os.path.expanduser(settings.tls_cert_path)
        self._client: AsyncIOMotorClient[Mapping[str, Any]] = AsyncIOMotorClient(
            settings.uri, tls=True, tlsCertificateKeyFile=cert, uuidRepresentation="standard"
        )
        self._embedding_service = embedding_service
        db = self._client.get_database(settings.database)
        self._collection: AsyncIOMotorCollection[Mapping[str, Any]] = db[settings.tomes_collection]

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
        category: str | None = None,
    ) -> list[tuple[Tome, float]]:
        """Perform hybrid search using Atlas Search (lexical) and Vector Search.

        Runs both pipelines concurrently and combines results using Reciprocal Rank Fusion.
        """
        query_embedding = await self._embedding_service.embed(query)
        query_vector = Binary.from_vector(
            np.array(query_embedding, dtype=np.float32).tolist(), BinaryVectorDtype.FLOAT32
        )

        lexical_results, vector_results = await asyncio.gather(
            self._lexical_search(query, top_k, min_confidence, category),
            self._vector_search(query_vector, top_k, min_confidence, category),
        )

        return self._merge_results(lexical_results, vector_results, top_k)

    async def _lexical_search(
        self, query: str, top_k: int, min_confidence: float, category: str | None
    ) -> list[Tome]:
        filters: list[Mapping[str, Any]] = [
            {"range": {"path": "confidence", "gte": min_confidence}},
        ]
        if category is not None:
            filters.append({"equals": {"path": "category", "value": category}})

        pipeline: list[Mapping[str, Any]] = [
            {
                "$search": {
                    "compound": {
                        "filter": filters,
                        "should": [
                            {
                                "text": {
                                    "query": query,
                                    "path": ["title", "content", "summary", "tags"],
                                }
                            }
                        ],
                    }
                }
            },
            {"$project": {"score": {"$meta": "searchScore"}, "document": "$$ROOT"}},
            {"$sort": {"score": -1}},
            {"$limit": top_k * 10},
        ]

        results: list[Tome] = []
        async for doc in self._collection.aggregate(pipeline):
            results.append(MongoTome.model_validate(doc["document"]).to_tome())
        return results

    async def _vector_search(
        self, query_vector: Binary, top_k: int, min_confidence: float, category: str | None
    ) -> list[Tome]:
        vector_filter: dict[str, Any] = {"confidence": {"$gte": min_confidence}}
        if category is not None:
            vector_filter["category"] = category

        pipeline: list[Mapping[str, Any]] = [
            {
                "$vectorSearch": {
                    "index": "vectors",
                    "path": "embedding",
                    "queryVector": query_vector,
                    "filter": vector_filter,
                    "numCandidates": top_k * 10,
                    "limit": top_k * 2,
                }
            },
            {"$project": {"score": {"$meta": "vectorSearchScore"}, "document": "$$ROOT"}},
            {"$sort": {"score": -1}},
        ]

        results: list[Tome] = []
        async for doc in self._collection.aggregate(pipeline):
            results.append(MongoTome.model_validate(doc["document"]).to_tome())
        return results

    RRF_K = 60

    @staticmethod
    def _merge_results(
        lexical: list[Tome],
        vector: list[Tome],
        top_k: int,
    ) -> list[tuple[Tome, float]]:
        """Reciprocal Rank Fusion (RRF) over two ranked result lists.

        Each list is assumed to be pre-sorted by its native score descending.
        RRF score for a document is: sum(1 / (k + rank)) across the lists it
        appears in, where rank is 1-based.
        """
        k = MongoTomeRepository.RRF_K

        tome_by_id: dict[UUID, Tome] = {}
        rrf_scores: dict[UUID, float] = {}

        for rank, tome in enumerate(lexical, start=1):
            tome_by_id[tome.id] = tome
            rrf_scores[tome.id] = rrf_scores.get(tome.id, 0.0) + 1.0 / (k + rank)

        for rank, tome in enumerate(vector, start=1):
            tome_by_id[tome.id] = tome
            rrf_scores[tome.id] = rrf_scores.get(tome.id, 0.0) + 1.0 / (k + rank)

        combined = [(tome_by_id[tid], score) for tid, score in rrf_scores.items()]
        combined.sort(key=lambda x: x[1], reverse=True)
        return combined[:top_k]

    async def find_near_duplicates(self, tome: Tome, threshold: float = 0.3) -> list[Tome]:
        """Find existing Tomes with cosine similarity above the threshold using $vectorSearch."""
        query_vector = Binary.from_vector(
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
        # TODO: Create indexes for vectorSearch if they don't already exist
        ...

    def close(self) -> None:
        """Close the MongoDB client connection."""
        self._client.close()
