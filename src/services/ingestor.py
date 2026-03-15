from __future__ import annotations

import asyncio
import logging
import uuid

from src.config import LibrarianConfig
from src.models.enums import IngestStatus, SourceType
from src.models.tome import Tome
from src.models.tool_schemas import IngestOutput
from src.services.verifier import Verifier
from src.storage.tome_repository import TomeRepository

SHARD_SIZE = 400
SHARD_OVERLAP = 100
SUMMARY_LENGTH = 200
TITLE_LENGTH = 120
UNVERIFIED_CONFIDENCE = 0.5


class ReshardError(Exception):
    """Raised when a reshard operation cannot be completed safely."""

    def __init__(self, message: str, tomes: list[Tome] | None = None) -> None:
        super().__init__(message)
        self.tomes = tomes or []


class Ingestor:
    """Receives raw knowledge, validates, reshards, embeds, and stores it as Tomes."""

    def __init__(
        self,
        config: LibrarianConfig,
        verifier: Verifier,
        tome_repo: TomeRepository,
    ) -> None:
        self._config = config
        self._verifier = verifier
        self._tome_repo = tome_repo

    async def ingest(self, blob: str) -> IngestOutput:
        """Convert unstructured text into one or more Tomes and save them."""
        if not blob.strip():
            return IngestOutput(
                tomes=[],
                status=IngestStatus.REJECTED,
                reject_reason="Content is empty",
            )

        shards = await self._reshard(blob)
        if not shards:
            return IngestOutput(
                tomes=[],
                status=IngestStatus.REJECTED,
                reject_reason=f"Unable to re-shard content {blob!r}",
            )

        tomes = await asyncio.gather(
            *[self._process_text(s) for s in shards],
            return_exceptions=True,
        )

        stored: list[Tome] = []
        any_rejected = False
        reject_reasons: list[str] = []

        for result in tomes:
            if isinstance(result, BaseException):
                any_rejected = True
                if isinstance(result, ReshardError):
                    stored.extend(result.tomes)
                    reject_reasons.append(str(result))
                else:
                    logging.error("Unhandled exception during ingest", exc_info=result)
                    reject_reasons.append(f"Unexpected error: {result}")
            elif result is None:
                any_rejected = True
            else:
                stored.extend(result)

        if not stored:
            reason = "All shards failed verification"
            if reject_reasons:
                reasons_str = "; ".join(reject_reasons)
                reason = f"All shards failed verification (or had errors): {reasons_str}"
            return IngestOutput(
                tomes=[],
                status=IngestStatus.REJECTED,
                reject_reason=reason,
            )

        status = IngestStatus.PARTIAL if any_rejected else IngestStatus.STORED
        final_reason = "; ".join(reject_reasons) if reject_reasons else None
        return IngestOutput(tomes=stored, status=status, reject_reason=final_reason)

    async def _process_text(self, text: str) -> list[Tome] | None:
        """Verify, classify, embed, and dedup/store a single text.

        Returns None if verification is enabled and confidence is below the reject
        threshold.  When verification is disabled the text is stored unconditionally
        with a confidence of UNVERIFIED_CONFIDENCE.
        """
        tome = await self._build_tome(text)
        if tome is None:
            return None
        return await self._dedup_and_store(tome)

    async def _build_tome(self, text: str) -> Tome | None:
        """Verify, classify, and embed a text, returning a Tome ready for storage.

        Returns None if verification rejects the text.  Does NOT persist anything.
        """
        if self._config.verification.enabled:
            (
                verification_result,
                (category, tags),
                (title, summary),
                embedding,
            ) = await asyncio.gather(
                self._verifier.verify(text),
                self._classify_and_tag(text),
                self._generate_title_and_summary(text),
                self._tome_repo.get_embedding(text),
            )
            if verification_result.confidence < self._config.verification.reject_threshold:
                return None
            confidence = verification_result.confidence
        else:
            (category, tags), (title, summary), embedding = await asyncio.gather(
                self._classify_and_tag(text),
                self._generate_title_and_summary(text),
                self._tome_repo.get_embedding(text),
            )
            confidence = UNVERIFIED_CONFIDENCE

        tome = Tome(
            id=uuid.uuid4(),
            content=text,
            summary=summary,
            title=title,
            category=category,
            tags=tags,
            embedding=embedding,
            source_url=None,
            source_type=SourceType.AGENT_INPUT,
            confidence=confidence,
        )
        try:
            self._validate(tome)
        except ValueError:
            logging.error("Tome validation failed", exc_info=True)
            return None
        return tome

    async def _dedup_and_store(self, tome: Tome) -> list[Tome]:
        """Insert tome, or reshard with any near-duplicates found in the repository.

        Reshard strategy (safe ordering):
        1. Build and verify all replacement tomes from combined content.
        2. Abort without touching existing tomes if no replacement survives verification.
        3. Delete old tomes only once at least one replacement is ready — checking
           each delete return value and raising ReshardError on failure.
        4. Insert all replacements.
        """
        duplicates = await self._tome_repo.find_near_duplicates(tome)
        if not duplicates:
            await self._tome_repo.insert(tome)
            return [tome]

        # Step 1 — build replacements from combined content (nothing persisted yet).
        combined = "\n\n".join([d.content for d in duplicates] + [tome.content])
        shards = await self._reshard(combined)
        replacement_results = await asyncio.gather(*[self._build_tome(c) for c in shards])
        replacements = [t for t in replacement_results if t is not None]

        # Step 2 — abort if verification left us with nothing to store.
        if not replacements:
            return []

        # Step 3 - Insert replacements.
        for replacement in replacements:
            await self._tome_repo.insert(replacement)

        # Step 4 - Delete old tomes.
        # Technically if we fail here, we may end up with duplicate data in the
        # library, but that seems like a better choice (IMO) than aborting.
        delete_errors = []
        for dup in duplicates:
            try:
                deleted = await self._tome_repo.delete(dup.id)
                if not deleted:
                    delete_errors.append(str(dup.id))
            except Exception as e:
                logging.warning("Exception deleting %s during reshard: %s", dup.id, e)
                delete_errors.append(str(dup.id))

        if delete_errors:
            failed_ids = ", ".join(delete_errors)
            msg = (
                "Failed to delete tomes during reshard "
                f"(duplicate data may exist for: {failed_ids})"
            )
            raise ReshardError(msg, tomes=replacements)

        return replacements

    async def _reshard(self, blob: str) -> list[str]:
        """Split blob into atomic, self-contained fact shards.

        For now, we just split into chunks of SHARD_SIZE characters with
        SHARD_OVERLAP overlap, but this should be replaced with LLM-driven
        decomposition that returns a list of atomic facts within the given
        character limit.
        """
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=SHARD_SIZE,
            chunk_overlap=SHARD_OVERLAP,
        )
        return splitter.split_text(blob)

    async def _classify_and_tag(
        self, text: str, category_hint: str | None = None
    ) -> tuple[str, list[str]]:
        """Auto-classify into a category and extract topic tags."""
        return category_hint or "Uncategorized", ["auto-tag"]

    async def _generate_title_and_summary(self, text: str) -> tuple[str, str]:
        """Generate a short title and one-to-two sentence summary for a text."""
        clean_text = text.strip().replace("\n", " ")
        title = clean_text[:TITLE_LENGTH] + "..." if len(clean_text) > TITLE_LENGTH else clean_text
        summary = clean_text[:SUMMARY_LENGTH]
        return title, summary

    def _validate(self, tome: Tome) -> None:
        """Post-construction checks; raises on failure."""
        if not tome.content.strip():
            raise ValueError("Tome content cannot be empty")
        if len(tome.title) > TITLE_LENGTH:
            raise ValueError(f"Tome title too long ({len(tome.title)} > {TITLE_LENGTH})")

        # In a real implementation we would enforce the embedding size.
        # Here we skip the dimension check if a dummy/string embedding is supplied.
        if (
            hasattr(tome.embedding, "shape")
            and tome.embedding.shape[0] != self._config.embedding.dimensions
        ):
            raise ValueError(
                f"Tome embedding has dimension {tome.embedding.shape[0]}, "
                f"expected {self._config.embedding.dimensions}"
            )
