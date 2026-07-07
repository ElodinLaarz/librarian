from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import numpy as np
from bson.binary import Binary, BinaryVectorDtype
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection
from pydantic import ValidationError
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

# Upper bound on how long startup waits for freshly created Atlas search
# indexes to become queryable. atlas-local builds them in seconds; real Atlas
# on an empty collection takes well under a minute.
_SEARCH_INDEX_READY_TIMEOUT_S = 120.0


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


def _doc_to_tome(doc: Mapping[str, Any], context: str) -> Tome:
    """Validate a raw Mongo document into a domain ``Tome``.

    A document that no longer matches the schema (manual edits, a legacy
    writer, a partial migration) is a storage-layer corruption problem:
    surface it as ``StorageError`` instead of leaking pydantic's
    ``ValidationError`` to services, which must never see backend or
    serialization exception types.
    """
    try:
        return MongoTome.model_validate(doc).to_tome()
    except ValidationError as exc:
        raise StorageError(
            f"Corrupt tome document during {context} (_id={doc.get('_id')!r})"
        ) from exc


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

    async def get_by_id(self, tome_id: UUID) -> Tome | None:
        """Retrieve a single Tome by its ID."""
        try:
            doc = await self._collection.find_one({"_id": tome_id})
        except PyMongoError as exc:
            raise _wrap_mongo(exc, "get_by_id") from exc
        if not doc:
            return None
        return _doc_to_tome(doc, "get_by_id")

    async def mark_superseded(self, tome_id: UUID, by_tome_id: UUID) -> bool:
        """Mark tome_id as superseded by by_tome_id."""
        try:
            result = await self._collection.update_one(
                {"_id": tome_id},
                {"$set": {"superseded_by": by_tome_id}},
            )
        except PyMongoError as exc:
            raise _wrap_mongo(exc, "mark_superseded") from exc
        return result.modified_count > 0

    @staticmethod
    def _build_list_filter(
        *,
        category: str | None,
        min_confidence: float,
        research_job_id: UUID | None,
        thread_id: UUID | None = None,
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
        """Return a filtered + paginated page of Tomes, newest first."""
        query = self._build_list_filter(
            category=category,
            min_confidence=min_confidence,
            research_job_id=research_job_id,
            thread_id=thread_id,
        )
        sort_field = "thread_position" if thread_id is not None else "created_at"
        sort_dir = 1 if thread_id is not None else -1
        cursor = self._collection.find(query).sort(sort_field, sort_dir).skip(offset).limit(limit)
        results = []
        try:
            async for doc in cursor:
                results.append(_doc_to_tome(doc, "list_all"))
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
        """Update mutable fields on a Tome. Returns the updated Tome or None if not found."""
        patch: dict[str, Any] = {}
        if content is not None:
            patch["content"] = content
        if category is not None:
            patch["category"] = category
        if tags is not None:
            patch["tags"] = tags
        if source_url is not None:
            patch["source_url"] = source_url
        if confidence is not None:
            patch["confidence"] = confidence

        if not patch:
            return await self.get_by_id(tome_id)

        try:
            result = await self._collection.find_one_and_update(
                {"_id": tome_id},
                {"$set": patch},
                return_document=True,
            )
        except PyMongoError as exc:
            raise _wrap_mongo(exc, "update") from exc
        if result is None:
            return None
        return _doc_to_tome(result, "update")

    async def search(
        self,
        query: str,
        top_k: int = 5,
        min_confidence: float = 0.5,
        category: str | None = None,
        include_superseded: bool = False,
        recency_weight: float = 0.0,
        recency_half_life_days: float = 90.0,
    ) -> list[tuple[Tome, float]]:
        """Perform hybrid search using Atlas Search (lexical) and Vector Search.

        Runs both pipelines concurrently and combines results using Reciprocal Rank Fusion.
        When include_superseded is False (default), filters out tomes marked as superseded.
        When recency_weight > 0, blends RRF scores with exponential-decay recency scores.
        """
        recency_weight = max(0.0, min(1.0, recency_weight))
        recency_half_life_days = max(0.0, recency_half_life_days)
        try:
            query_embedding = await self._embedding_service.embed(query)
        except Exception as exc:
            # Repository callers only handle StorageError; an httpx / runtime
            # failure from the embedding provider must not leak through.
            raise StorageError(
                f"search: failed to embed query via {type(self._embedding_service).__name__}"
            ) from exc
        query_array = np.asarray(query_embedding, dtype=np.float32)
        # Atlas $vectorSearch raises OperationFailure for zero query vectors.
        # Fall back to lexical-only when the embedding is all zeros.
        has_valid_vector = query_array.size > 0 and bool(np.any(query_array))
        query_vector = (
            Binary.from_vector(query_array.tolist(), BinaryVectorDtype.FLOAT32)
            if has_valid_vector
            else None
        )

        try:
            if query_vector is not None:
                lexical_results, vector_results = await asyncio.gather(
                    self._lexical_search(
                        query, top_k, min_confidence, category, include_superseded
                    ),
                    self._vector_search(
                        query_vector, top_k, min_confidence, category, include_superseded
                    ),
                )
            else:
                lexical_results = await self._lexical_search(
                    query, top_k, min_confidence, category, include_superseded
                )
                vector_results = []
        except PyMongoError as exc:
            raise _wrap_mongo(exc, "search") from exc

        return self._merge_results(
            lexical_results,
            vector_results,
            top_k,
            recency_weight=recency_weight,
            recency_half_life_days=recency_half_life_days,
        )

    async def _lexical_search(
        self,
        query: str,
        top_k: int,
        min_confidence: float,
        category: str | None,
        include_superseded: bool = False,
    ) -> list[Tome]:
        filters: list[Mapping[str, Any]] = [
            {"range": {"path": "confidence", "gte": min_confidence}},
        ]
        if category is not None:
            filters.append({"equals": {"path": "category", "value": category}})

        compound: dict[str, Any] = {
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
        # Atlas Search $equals doesn't support null; use mustNot+exists to exclude
        # superseded docs before the $limit stage so the limit isn't wasted on them.
        if not include_superseded:
            compound["mustNot"] = [{"exists": {"path": "superseded_by"}}]

        pipeline: list[Mapping[str, Any]] = [
            {"$search": {"compound": compound}},
            {"$project": {"score": {"$meta": "searchScore"}, "document": "$$ROOT"}},
            {"$sort": {"score": -1}},
            {"$limit": top_k * 10},
        ]

        results: list[Tome] = []
        async for doc in self._collection.aggregate(pipeline):
            results.append(_doc_to_tome(doc["document"], "search (lexical)"))
        return results

    async def _vector_search(
        self,
        query_vector: Binary,
        top_k: int,
        min_confidence: float,
        category: str | None,
        include_superseded: bool = False,
    ) -> list[Tome]:
        vector_filter: dict[str, Any] = {"confidence": {"$gte": min_confidence}}
        if category is not None:
            vector_filter["category"] = category
        if not include_superseded:
            # superseded_by is declared as a filter field in the vectors index.
            vector_filter["superseded_by"] = None

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
            results.append(_doc_to_tome(doc["document"], "search (vector)"))
        return results

    RRF_K = 60

    @staticmethod
    def _merge_results(
        lexical: list[Tome],
        vector: list[Tome],
        top_k: int,
        recency_weight: float = 0.0,
        recency_half_life_days: float = 90.0,
    ) -> list[tuple[Tome, float]]:
        """Reciprocal Rank Fusion (RRF) over two ranked result lists.

        Each list is assumed to be pre-sorted by its native score descending.
        RRF score for a document is: sum(1 / (k + rank)) across the lists it
        appears in, where rank is 1-based.

        When recency_weight > 0, blends RRF with exponential-decay recency:
          recency_score = exp(-ln(2) * age_days / half_life_days)
          final = rrf * (1 - recency_weight) + recency_weight * recency_score
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

        if recency_weight <= 0.0:
            combined = [(tome_by_id[tid], score) for tid, score in rrf_scores.items()]
        else:
            # Normalize RRF into [0, 1] so recency_weight has consistent meaning.
            # Raw RRF sums top out around 1/(k+1) ≈ 0.016 with RRF_K=60, while
            # recency is already in [0, 1], causing recency to dominate at any weight.
            max_rrf = max(rrf_scores.values()) if rrf_scores else 1.0
            now = datetime.now(UTC)
            combined = []
            for tid, rrf in rrf_scores.items():
                tome = tome_by_id[tid]
                norm_rrf = rrf / max_rrf if max_rrf > 0 else 0.0
                created = tome.created_at
                if created is None:
                    recency = 0.0
                else:
                    if created.tzinfo is None:
                        created = created.replace(tzinfo=UTC)
                    age_days = max(0.0, (now - created).total_seconds() / 86400.0)
                    recency = (
                        2.0 ** (-age_days / recency_half_life_days)
                        if recency_half_life_days > 0
                        else 0.0
                    )
                score = norm_rrf * (1.0 - recency_weight) + recency_weight * recency
                combined.append((tome, score))

        combined.sort(key=lambda x: x[1], reverse=True)
        return combined[:top_k]

    async def find_near_duplicates(self, tome: Tome, threshold: float | None = None) -> list[Tome]:
        """Find existing Tomes with cosine similarity above the threshold using $vectorSearch."""
        if tome.embedding is None:
            return []

        effective_threshold = threshold if threshold is not None else self._tidy_settings.threshold
        # The configured threshold is a raw cosine similarity (the filesystem
        # backend compares numpy cosine against it directly). Atlas
        # $vectorSearch returns vectorSearchScore = (1 + cosine) / 2 for
        # cosine indexes, so convert into score space before matching —
        # otherwise this backend dedupes at a materially looser bar than
        # configured (cosine 0.90 when the setting says 0.95).
        score_threshold = (1.0 + effective_threshold) / 2.0
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
                    "score": {"$gte": score_threshold},
                    "document._id": {"$ne": tome.id},
                }
            },
        ]

        duplicates = []
        try:
            async for doc in self._collection.aggregate(pipeline):
                duplicates.append(_doc_to_tome(doc["document"], "find_near_duplicates"))
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
                all_tomes.append(_doc_to_tome(doc, "find_all_near_duplicates"))
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

        Blocks until the Atlas search indexes report ``queryable`` (bounded
        by ``_SEARCH_INDEX_READY_TIMEOUT_S``): Atlas builds search indexes
        asynchronously after creation, and running $search/$vectorSearch
        against a still-building index errors or silently returns nothing —
        so without the wait, a freshly booted server rejects its first
        ingests (dedup scan) and returns empty first searches.
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

        existing_search_indexes: dict[str, Mapping[str, Any]] = {}

        try:
            async for index in self._collection.aggregate([{"$listSearchIndexes": {}}]):
                existing_search_indexes[index["name"]] = index
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
        vectors_index = existing_search_indexes.get("vectors")
        if vectors_index is not None:
            self._check_vector_index_dimensions(vectors_index)
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
                        {"type": "filter", "path": "superseded_by"},
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

        await self._wait_for_search_indexes(("vectors", "default"))

    def _check_vector_index_dimensions(self, index: Mapping[str, Any]) -> None:
        """Fail fast when the existing vector index disagrees with the model.

        Atlas does not error on a dimension mismatch — it silently stops
        indexing documents whose embeddings do not fit the index, so after an
        embedding-model change vector search quietly degrades to partial or
        empty results. Startup is the one reliable place to catch that.
        """
        definition = index.get("latestDefinition") or index.get("definition") or {}
        for field in definition.get("fields", []):
            if field.get("type") == "vector" and field.get("path") == "embedding":
                indexed = field.get("numDimensions")
                configured = self._embedding_service.dimensions
                if isinstance(indexed, int) and indexed != configured:
                    raise StorageError(
                        f"Atlas vector index 'vectors' was built for numDimensions={indexed}, "
                        f"but the configured embedding model produces {configured}. Existing "
                        "embeddings are incompatible with the new model: either restore the "
                        "original embedding config, or drop the index and re-ingest."
                    )
                return
        logger.debug(
            "Existing 'vectors' index does not expose numDimensions; skipping dimension check"
        )

    async def _wait_for_search_indexes(
        self,
        names: tuple[str, ...],
        timeout_s: float = _SEARCH_INDEX_READY_TIMEOUT_S,
        poll_interval_s: float = 1.0,
    ) -> None:
        """Block until the named Atlas search indexes report ``queryable``.

        ``create_search_index`` returns as soon as the index is *registered*;
        the actual build is asynchronous. Querying a still-building index
        raises ``OperationFailure`` ($vectorSearch) or silently matches
        nothing ($search), so serving requests before readiness turns into
        rejected ingests and empty searches with no obvious cause. Backends
        that do not report a ``queryable`` field cannot be polled; they are
        skipped rather than hanging startup (mocked collections in unit
        tests, some server versions).
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        pending = set(names)
        while True:
            queryable: dict[str, Any] = {}
            try:
                async for index in self._collection.aggregate([{"$listSearchIndexes": {}}]):
                    name = index.get("name")
                    if name in pending:
                        queryable[name] = index.get("queryable")
            except PyMongoError as exc:
                raise _wrap_mongo(exc, "ensure_indexes (wait for search indexes)") from exc

            if queryable and all(status is None for status in queryable.values()):
                logger.debug(
                    "Search indexes do not report queryable status; skipping readiness wait"
                )
                return

            pending = {name for name in pending if not queryable.get(name)}
            if not pending:
                return
            if loop.time() >= deadline:
                raise StorageError(
                    f"Atlas search indexes {sorted(pending)} did not become queryable "
                    f"within {timeout_s:.0f}s of creation"
                )
            await asyncio.sleep(poll_interval_s)

    def close(self) -> None:
        """Close the MongoDB client connection if this repo owns it.

        When the client was injected by the caller (e.g. the lifespan-owned
        shared client) ownership stays with the caller and ``close()`` is a
        no-op, so a sibling repo using the same client can keep working.
        """
        if self._owns_client:
            self._client.close()
