from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from typing import Any
from uuid import UUID

import numpy as np
from bson.binary import Binary, BinaryVectorDtype
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection
from pymongo.errors import (
    CollectionInvalid,
    ConnectionFailure,
    DuplicateKeyError,
    OperationFailure,
    PyMongoError,
    ServerSelectionTimeoutError,
)
from pymongo.operations import SearchIndexModel

from src.config import DatabaseSettings, TidySettings
from src.models.tome import Tome
from src.services.duplicate_detection import build_duplicate_groups
from src.services.embedding import EmbeddingService
from src.storage.errors import (
    BackendUnavailableError,
    DuplicateError,
    StorageError,
)
from src.storage.mongo.client import build_motor_client
from src.storage.mongo.mongo_tome import MongoTome
from src.storage.tome_repository import DuplicateScanResult, TomeRepository

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
    # Match case-insensitively so different server versions / mock implementations
    # that vary the casing (e.g. 'unrecognized' vs 'Unrecognized') still trip
    # the same skip path.
    message = str(exc).lower()
    return "$listsearchindexes" in message and (
        "unrecognized pipeline stage" in message or "unknown" in message
    )


def _wrap_mongo(exc: PyMongoError, context: str) -> StorageError:
    """Translate a generic PyMongoError into the right StorageError subclass.

    Used for read paths where DuplicateKeyError is not a concern. Connectivity
    failures map to ``BackendUnavailableError``; everything else falls through
    to a bare ``StorageError`` so the original cause is preserved via ``from``.
    """
    if isinstance(exc, ServerSelectionTimeoutError | ConnectionFailure):
        return BackendUnavailableError(f"Mongo backend unavailable during {context}")
    return StorageError(f"Mongo {context} failed: {exc.__class__.__name__}")


class MongoTomeRepository(TomeRepository):
    """MongoDB implementation of the TomeRepository using Atlas Search.

    This implementation expects a MongoDB Atlas cluster with a Search index
    configured to support both vector and lexical search.
    """

    def __init__(
        self,
        settings: DatabaseSettings,
        embedding_service: EmbeddingService,
        tidy_settings: TidySettings | None = None,
        *,
        client: AsyncIOMotorClient[Mapping[str, Any]] | None = None,
        owns_client: bool | None = None,
    ) -> None:
        """Create a tome repo against the given Mongo database.

        When ``client`` is provided the repository uses the shared client and
        does NOT close it on ``close()`` — ownership stays with the caller
        (typically :class:`LibrarianServer` lifespan). When ``client`` is
        omitted a private client is built from ``settings`` and closed on
        ``close()`` for backwards compatibility with the previous API and
        with tests that construct repos directly.
        """
        self._client: AsyncIOMotorClient[Mapping[str, Any]] = (
            client if client is not None else build_motor_client(settings)
        )
        # Default ownership tracks whether the caller supplied the client: a
        # repo that built its own client closes it; one handed a shared client
        # does not. ``owns_client`` lets callers override this for tests.
        self._owns_client = owns_client if owns_client is not None else (client is None)

        self._embedding_service = embedding_service
        self._tidy_settings = tidy_settings or TidySettings()
        db = self._client.get_database(settings.database)
        self._collection: AsyncIOMotorCollection[Mapping[str, Any]] = db[settings.tomes_collection]

    async def insert(self, tome: Tome) -> UUID:
        """Insert a new Tome into MongoDB."""
        mongo_tome = MongoTome.from_tome(tome)
        try:
            await self._collection.insert_one(mongo_tome.model_dump(by_alias=True))
        except DuplicateKeyError as exc:
            raise DuplicateError(f"Tome {tome.id} already exists") from exc
        except PyMongoError as exc:
            raise _wrap_mongo(exc, "insert") from exc
        return tome.id

    async def delete(self, tome_id: UUID) -> bool:
        """Permanently remove a Tome by ID."""
        try:
            result = await self._collection.delete_one({"_id": tome_id})
        except PyMongoError as exc:
            raise _wrap_mongo(exc, "delete") from exc
        return result.deleted_count > 0

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
        """Update mutable fields of a Tome. Returns updated Tome or None if not found."""
        update_dict: dict[str, object] = {}

        if content is not None:
            embedding = await self._embedding_service.embed(content)
            update_dict["content"] = content
            update_dict["embedding"] = Binary.from_uuid(embedding, subtype=BinaryVectorDtype.VECTOR)

        if category is not None:
            update_dict["category"] = category
        if tags is not None:
            update_dict["tags"] = tags
        if source_url is not None:
            update_dict["source_url"] = source_url
        if confidence is not None:
            update_dict["confidence"] = confidence

        if not update_dict:
            return await self.get_by_id(tome_id)

        try:
            result = await self._collection.update_one(
                {"_id": tome_id},
                {"$set": update_dict}
            )
        except PyMongoError as exc:
            raise _wrap_mongo(exc, "update") from exc

        if result.matched_count == 0:
            return None

        return await self.get_by_id(tome_id)

    async def get_by_id(self, tome_id: UUID) -> Tome | None:
        """Retrieve a single Tome by its ID."""
        try:
            doc = await self._collection.find_one({"_id": tome_id})
        except PyMongoError as exc:
            raise _wrap_mongo(exc, "get_by_id") from exc
        if not doc:
            return None
        return MongoTome.model_validate(doc).to_tome()

    @staticmethod
    def _build_list_filter(
        *,
        category: str | None,
        min_confidence: float,
        research_job_id: UUID | None,
        thread_id: UUID | None,
    ) -> dict[str, Any]:
        """Translate the public filter args into a Mongo ``find`` predicate.

        Only adds clauses for set filters so the default ``list_all()`` call
        is a full collection scan (matches the abstract contract).
        """
        query: dict[str, Any] = {}
        if category is not None:
            query["category"] = category
        if min_confidence > 0.0:
            query["confidence"] = {"$gte": min_confidence}
        if research_job_id is not None:
            query["research_job_id"] = research_job_id
        if thread_id is not None:
            query["thread_id"] = thread_id
        return query

    async def list_all(
        self,
        limit: int = 100,
        offset: int = 0,
        *,
        category: str | None = None,
        min_confidence: float = 0.0,
        research_job_id: UUID | None = None,
        thread_id: UUID | None = None,
    ) -> list[Tome]:
        """Return a filtered + paginated page of Tomes, newest first.

        When thread_id is set, return tomes in that thread sorted by thread_position
        ascending (conversational order). Otherwise, sort by created_at descending.
        """
        query = self._build_list_filter(
            category=category,
            min_confidence=min_confidence,
            research_job_id=research_job_id,
            thread_id=thread_id,
        )
        # If thread_id is set, sort by thread_position ascending; otherwise by created_at descending
        if thread_id is not None:
            sort_key = "thread_position"
            sort_order = 1
        else:
            sort_key = "created_at"
            sort_order = -1
        cursor = self._collection.find(query).sort(sort_key, sort_order).skip(offset).limit(limit)
        results = []
        try:
            async for doc in cursor:
                results.append(MongoTome.model_validate(doc).to_tome())
        except PyMongoError as exc:
            raise _wrap_mongo(exc, "list_all") from exc
        return results

    async def count(
        self,
        *,
        category: str | None = None,
        min_confidence: float = 0.0,
        research_job_id: UUID | None = None,
        thread_id: UUID | None = None,
    ) -> int:
        """Count Tomes matching the same filter predicates as :meth:`list_all`."""
        query = self._build_list_filter(
            category=category,
            min_confidence=min_confidence,
            research_job_id=research_job_id,
            thread_id=thread_id,
        )
        try:
            return int(await self._collection.count_documents(query))
        except PyMongoError as exc:
            raise _wrap_mongo(exc, "count") from exc

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

        try:
            lexical_results, vector_results = await asyncio.gather(
                self._lexical_search(query, top_k, min_confidence, category),
                self._vector_search(query_vector, top_k, min_confidence, category),
            )
        except PyMongoError as exc:
            raise _wrap_mongo(exc, "search") from exc

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

    async def find_near_duplicates(self, tome: Tome, threshold: float | None = None) -> list[Tome]:
        """Find existing Tomes with cosine similarity above the threshold using $vectorSearch."""
        if tome.embedding is None:
            return []

        effective_threshold = threshold if threshold is not None else self._tidy_settings.threshold
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
            {
                "$match": {
                    "score": {"$gte": effective_threshold},
                    "document._id": {"$ne": tome.id},
                }
            },
        ]

        duplicates = []
        try:
            async for doc in self._collection.aggregate(pipeline):
                duplicates.append(MongoTome.model_validate(doc["document"]).to_tome())
        except PyMongoError as exc:
            raise _wrap_mongo(exc, "find_near_duplicates") from exc

        return duplicates

    async def find_all_near_duplicates(self, threshold: float = 0.95) -> DuplicateScanResult:
        """Find duplicate groups for tidy-time consolidation."""
        projection = {
            "_id": 1,
            "title": 1,
            "content": 1,
            "summary": 1,
            "category": 1,
            "tags": 1,
            "source_url": 1,
            "source_type": 1,
            "confidence": 1,
            "research_job_id": 1,
            "embedding": 1,
            "created_at": 1,
        }
        cursor = self._collection.find({}, projection=projection).batch_size(
            self._tidy_settings.scan_batch_size
        )
        all_tomes: list[Tome] = []
        try:
            async for doc in cursor:
                all_tomes.append(MongoTome.model_validate(doc).to_tome())
        except PyMongoError as exc:
            raise _wrap_mongo(exc, "find_all_near_duplicates") from exc

        return build_duplicate_groups(
            all_tomes,
            self._tidy_settings.model_copy(update={"threshold": threshold}),
        )

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
            # Code 48 is NamespaceExists. Some server versions surface this as
            # an OperationFailure rather than CollectionInvalid; treat that as
            # the documented "already exists" path and re-raise everything
            # else (e.g. auth/permission failures) so startup fails loudly.
            if exc.code == 48:
                logger.debug("Collection already exists, skipping creation.")
            else:
                logger.error("Failed to create collection", exc_info=True)
                raise StorageError("Failed to create Mongo collection") from exc
        except PyMongoError as exc:
            raise _wrap_mongo(exc, "ensure_indexes") from exc

        # Standard secondary indexes for non-Atlas-Search queries — chiefly
        # ``list_all`` which sorts by ``created_at`` descending and may filter
        # by ``category`` or ``research_job_id``. Without these the server has
        # to do an in-memory sort, which Mongo caps at 32 MB and aborts on
        # large libraries. ``create_index`` is idempotent so we can call it
        # unconditionally on every startup.
        try:
            await self._collection.create_index([("created_at", -1)])
            await self._collection.create_index("category")
            await self._collection.create_index("research_job_id")
        except PyMongoError as exc:
            raise _wrap_mongo(exc, "ensure_indexes (standard indexes)") from exc

        existing_search_indexes: list[str] = []

        try:
            async for index in self._collection.aggregate([{"$listSearchIndexes": {}}]):
                existing_search_indexes.append(index["name"])
        except OperationFailure as exc:
            if _is_atlas_search_unsupported(exc):
                logger.warning(
                    "Atlas search indexes unsupported on this backend; skipping search index setup"
                )
                return
            logger.error(
                "Failed to enumerate Atlas search indexes; aborting startup",
                exc_info=True,
            )
            raise StorageError("Failed to enumerate Atlas search indexes") from exc
        except PyMongoError as exc:
            raise _wrap_mongo(exc, "ensure_indexes (list search indexes)") from exc

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
            except OperationFailure as exc:
                logger.error(
                    "Failed to create Atlas vector search index 'vectors'",
                    exc_info=True,
                )
                raise StorageError("Failed to create Atlas vector search index 'vectors'") from exc
            except PyMongoError as exc:
                raise _wrap_mongo(exc, "create_search_index (vectors)") from exc

        # 3. Define Lexical Search Index
        if "default" not in existing_search_indexes:
            lexical_model = SearchIndexModel(
                definition={"mappings": {"dynamic": True}}, name="default"
            )
            try:
                await self._collection.create_search_index(model=lexical_model)
            except OperationFailure as exc:
                logger.error(
                    "Failed to create Atlas lexical search index 'default'",
                    exc_info=True,
                )
                raise StorageError("Failed to create Atlas lexical search index 'default'") from exc
            except PyMongoError as exc:
                raise _wrap_mongo(exc, "create_search_index (default)") from exc

    def close(self) -> None:
        """Close the MongoDB client connection if this repo owns it.

        When the client was injected by the caller (e.g. the lifespan-owned
        shared client) ownership stays with the caller and ``close()`` is a
        no-op, so a sibling repo using the same client can keep working.
        """
        if self._owns_client:
            self._client.close()
