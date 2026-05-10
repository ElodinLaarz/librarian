from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Mapping
from typing import Any
from uuid import UUID

import numpy as np
from bson.binary import Binary, BinaryVectorDtype
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection
from pymongo.errors import CollectionInvalid, OperationFailure
from pymongo.operations import SearchIndexModel

from src.config import DatabaseSettings
from src.models.tome import Tome
from src.services.embedding import EmbeddingService
from src.storage.mongo.mongo_tome import MongoTome
from src.storage.tome_repository import TomeRepository

logger = logging.getLogger(__name__)

# Atlas Search pipeline stages ($listSearchIndexes, $search, $vectorSearch) are
# unsupported on stock community mongod. The server reports this as one of the
# codeNames below — treat them as "skip search-index setup" rather than as a
# fatal startup error so local CI fixtures keep working.
_ATLAS_SEARCH_UNSUPPORTED_CODE_NAMES = frozenset(
    {
        "CommandNotFound",
        "SearchNotEnabled",
        "InvalidPipelineOperator",
    }
)


def _is_atlas_search_unsupported(exc: OperationFailure) -> bool:
    """Return True if the failure means the backend lacks Atlas Search at all.

    Distinguishes a legitimate "this is plain mongod, skip search indexes"
    case from a transient Atlas outage that must surface loudly.
    """
    details = exc.details or {}
    code_name = details.get("codeName")
    if code_name in _ATLAS_SEARCH_UNSUPPORTED_CODE_NAMES:
        return True
    # Older server builds don't populate codeName; fall back to the message.
    message = str(exc)
    return "$listSearchIndexes" in message and (
        "Unrecognized pipeline stage" in message or "unknown" in message.lower()
    )


class MongoTomeRepository(TomeRepository):
    """MongoDB implementation of the TomeRepository using Atlas Search.

    This implementation expects a MongoDB Atlas cluster with a Search index
    configured to support both vector and lexical search.
    """

    def __init__(self, settings: DatabaseSettings, embedding_service: EmbeddingService) -> None:
        kwargs: dict[str, Any] = {"uuidRepresentation": "standard"}
        if settings.tls:
            kwargs["tls"] = True
            kwargs["tlsCertificateKeyFile"] = os.path.expanduser(settings.tls_cert_path)
        else:
            kwargs["tls"] = False

        self._client: AsyncIOMotorClient[Mapping[str, Any]] = AsyncIOMotorClient(
            settings.uri, **kwargs
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

    async def find_near_duplicates(self, tome: Tome, threshold: float = 0.95) -> list[Tome]:
        """Find existing Tomes with cosine similarity above the threshold using $vectorSearch."""
        if tome.embedding is None:
            return []

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
        """Create search and vector indexes programmatically.

        Failures during Atlas search-index setup are surfaced rather than
        silently swallowed (issue #27). The only legitimate skip is when the
        backend is plain mongod that lacks Atlas Search entirely; that case
        is detected via the server's codeName and logged as a warning.
        """
        # Ensure collection exists.
        try:
            await self._collection.database.create_collection(self._collection.name)
        except CollectionInvalid:
            # Collection already exists — the documented "already exists" path.
            pass
        except OperationFailure as exc:
            # Some server versions surface "already exists" as an OperationFailure
            # rather than CollectionInvalid; log at debug since it's expected on
            # subsequent startups.
            logger.debug("create_collection skipped: %s", exc)

        existing_search_indexes: list[str] = []

        try:
            async for index in self._collection.aggregate([{"$listSearchIndexes": {}}]):
                existing_search_indexes.append(index["name"])
        except OperationFailure as exc:
            if _is_atlas_search_unsupported(exc):
                logger.warning(
                    "Atlas search indexes unsupported on this backend; skipping search index setup",
                    exc_info=True,
                )
                return
            logger.error(
                "Failed to enumerate Atlas search indexes; aborting startup",
                exc_info=True,
            )
            raise

        # 2. Define Vector Search Index
        if "vectors" not in existing_search_indexes:
            vector_model = SearchIndexModel(
                definition={
                    "fields": [
                        {
                            "type": "vector",
                            "path": "embedding",
                            "numDimensions": self._embedding_service.dimensions,
                            "similarity": "cosine",
                        },
                        {"type": "filter", "path": "confidence"},
                        {"type": "filter", "path": "category"},
                    ]
                },
                name="vectors",
                type="vectorSearch",
            )
            try:
                await self._collection.create_search_index(model=vector_model)
            except OperationFailure:
                logger.error(
                    "Failed to create Atlas vector search index 'vectors'",
                    exc_info=True,
                )
                raise

        # 3. Define Lexical Search Index
        if "default" not in existing_search_indexes:
            lexical_model = SearchIndexModel(
                definition={"mappings": {"dynamic": True}}, name="default"
            )
            try:
                await self._collection.create_search_index(model=lexical_model)
            except OperationFailure:
                logger.error(
                    "Failed to create Atlas lexical search index 'default'",
                    exc_info=True,
                )
                raise

    def close(self) -> None:
        """Close the MongoDB client connection."""
        self._client.close()
