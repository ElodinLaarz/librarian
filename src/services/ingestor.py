from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from collections.abc import Awaitable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Literal, TypeVar
from uuid import UUID

import httpx
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from src import constants
from src.config import LibrarianConfig
from src.models.enums import IngestStatus, SourceType
from src.models.tome import Tome, VerificationSummary
from src.models.tool_schemas import IngestOutput
from src.services.embedding import EmbeddingService
from src.services.verifier import Verifier
from src.storage.errors import StorageError
from src.storage.tome_repository import TomeRepository

# Format identifiers used by the syntax-aware chunker. "text" = generic prose
# (default LLM/recursive path); the rest skip the LLM "atomic facts" rewrite.
DetectedFormat = Literal["text", "code", "python", "yaml", "json", "markdown"]


def _detect_format(blob: str) -> DetectedFormat:
    """Heuristically classify a blob's format.

    Order matters — markdown fences are checked before Python keywords so that
    a markdown document containing fenced Python is not mis-routed.
    """
    if not blob.strip():
        return "text"

    # Triple-backtick fences strongly suggest Markdown.
    if "```" in blob:
        return "markdown"

    stripped = blob.lstrip()
    first_char = stripped[0]
    if first_char in "{[":
        return "json"

    # Top-level Python signals: def / class / import / from at column 0.
    if re.search(r"(?m)^(def |class |import |from \S+ import )", blob):
        return "python"

    # YAML: mapping keys at column 0 (`^[\w-]+:` followed by space/eol).
    yaml_key_lines = re.findall(r"(?m)^[A-Za-z_][\w-]*:(?:\s|$)", blob)
    if len(yaml_key_lines) >= 2:
        return "yaml"

    return "text"


def _balance_markdown_fences(chunk: str) -> str:
    """Ensure a markdown chunk has an even count of ``` triple-backticks.

    A language-aware split can leave a chunk that opens a fenced code block
    without closing it (or vice-versa). To keep each shard renderable on its
    own, append a closing fence when the count is odd. If the chunk already
    *ends* with a fence (i.e. the lone fence is a closing one and the body is
    mid-block), prepend an opening fence instead so we don't append a third
    backtick run that yields broken markdown.
    """
    if chunk.count("```") % 2 == 0:
        return chunk

    # Chunk ends with a fence -> the lone fence is a closing one; the body is
    # mid-block. Prepend an opening fence to make it self-contained.
    if chunk.rstrip().endswith("```"):
        prefix = "" if chunk.startswith("\n") else "\n"
        return f"```{prefix}{chunk}"

    suffix = "" if chunk.endswith("\n") else "\n"
    return f"{chunk}{suffix}```"


T = TypeVar("T")


@dataclass
class IngestCallOptions:
    skip_verify: bool = False
    source_type: SourceType = SourceType.AGENT_INPUT
    source_url: str | None = None
    research_job_id: UUID | None = None
    thread_id: UUID | None = None
    category_hint: str | None = None
    tags_hint: list[str] | None = None
    # ``extra_tags`` are appended to the tome's tag list *after* classification,
    # unlike ``tags_hint`` which (when supplied alongside ``category_hint``)
    # short-circuits the classifier. Use this for caller-supplied tags that
    # should accompany — not replace — content-derived tags (e.g. the research
    # job topic).
    extra_tags: list[str] | None = None
    force_format: DetectedFormat | None = None
    confidence: float | None = None
    """Explicit confidence for the stored tomes, in [0.0, 1.0].

    When set, it replaces both the verification-derived confidence and
    ``ingest.unverified_confidence``. The verification reject gate still
    applies — an override sets the stored value, it does not rescue content
    that verification rejected. Intended for first-hand knowledge ingested
    with ``skip_verify`` (agent session notes, memory mirrors), which would
    otherwise land at the unverified default and can fall below downstream
    ``min_confidence`` retrieval filters.
    """
    allow_short: bool = False
    """When True, _validate skips the minimum-shard-size floor.

    Set automatically by ingest() when the entire input blob is below the
    floor (e.g. tweets, short notes) so legitimate short documents are not
    rejected.
    """
    supersedes_tome_ids: list[str] | None = None


def _cap_tags(tags: list[str]) -> list[str]:
    """Dedupe, strip blanks, truncate each tag, and cap the list size.

    Order is preserved (first-seen wins). Length and count limits come from
    :mod:`src.constants` so all tag-producing paths stay aligned.
    """
    seen: dict[str, None] = {}
    for raw in tags:
        if not isinstance(raw, str):
            continue
        cleaned = raw.strip()
        if not cleaned:
            continue
        truncated = cleaned[: constants.MAX_TAG_LENGTH]
        seen.setdefault(truncated, None)
        if len(seen) >= constants.MAX_TAGS_PER_TOME:
            break
    return list(seen.keys())


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
        embedding_service: EmbeddingService,
        verifier: Verifier,
        tome_repo: TomeRepository,
    ) -> None:
        self._config = config
        self._embedding_service = embedding_service
        self._verifier = verifier
        self._tome_repo = tome_repo
        self._http_client: httpx.AsyncClient | None = None
        self._http_client_lock = asyncio.Lock()
        if config.ingest.use_llm_chunking:
            logging.warning(
                "ingest.use_llm_chunking=True: the ingestor will rewrite prose "
                "into LLM-generated atomic facts (NOT a deterministic split). "
                "Source text is truncated at ingest.llm_extraction_max_chars=%d "
                "and replaced with model output, so stored content may differ "
                "in wording and meaning from the original. See README/lld.md.",
                config.ingest.llm_extraction_max_chars,
            )

    async def _get_http_client(self) -> httpx.AsyncClient:
        """Lazily construct and reuse a single httpx client for LLM HTTP calls."""
        if self._http_client is None:
            async with self._http_client_lock:
                if self._http_client is None:
                    self._http_client = httpx.AsyncClient(timeout=120.0)
        return self._http_client

    async def aclose(self) -> None:
        """Release the persistent HTTP client, if one was created."""
        async with self._http_client_lock:
            client = self._http_client
            self._http_client = None
        if client is not None:
            await client.aclose()

    async def ingest(self, blob: str, options: IngestCallOptions | None = None) -> IngestOutput:
        """Convert unstructured text into one or more Tomes and save them."""
        opts = options or IngestCallOptions()
        if not blob.strip():
            return IngestOutput(
                tomes=[],
                status=IngestStatus.REJECTED,
                reject_reason="Content is empty",
            )

        shards = await self._reshard(blob, opts)
        if not shards:
            return IngestOutput(
                tomes=[],
                status=IngestStatus.REJECTED,
                reject_reason=(
                    f"Unable to re-shard content "
                    f"({len(blob)} chars, shard_size={self._config.ingest.shard_size}, "
                    f"shard_overlap={self._config.ingest.shard_overlap})"
                ),
            )

        # Allow-short rules:
        # - Whole-blob bypass: if the entire input is below the minimum-shard
        #   floor, treat it as a legitimately short document.
        # - Structured formats (code / json / yaml / markdown): individual
        #   structural shards may be below the floor, so skip the floor.
        # In both cases, _validate skips the minimum-shard-size enforcement.
        fmt = opts.force_format or _detect_format(blob)
        allow_short = (
            opts.allow_short
            or fmt != "text"
            or len(blob.strip()) < self._config.ingest.min_shard_chars
        )
        per_shard_opts = opts
        if allow_short and not opts.allow_short:
            per_shard_opts = replace(opts, allow_short=True)

        tomes = await asyncio.gather(
            *[self._process_text(s, per_shard_opts) for s in shards],
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
            elif not result:
                any_rejected = True
            else:
                stored.extend(result)

        if not stored:
            reason = "All shards failed verification"
            if reject_reasons:
                reasons_str = constants.JOIN_SEPARATOR.join(reject_reasons)
                reason = f"All shards failed verification (or had errors): {reasons_str}"
            return IngestOutput(
                tomes=[],
                status=IngestStatus.REJECTED,
                reject_reason=reason,
            )

        # Mark superseded tomes (soft-delete) if specified
        if opts.supersedes_tome_ids and stored:
            # When there are multiple stored tomes (from resharding), mark the
            # old tomes as superseded by the first stored tome. In practice,
            # when supersedes_tome_ids is set there should be one stored tome.
            new_tome_id = stored[0].id
            supersede_failures: list[str] = []
            for old_tome_id_str in opts.supersedes_tome_ids:
                try:
                    old_tome_id = UUID(old_tome_id_str)
                    marked = await self._tome_repo.mark_superseded(old_tome_id, new_tome_id)
                    if not marked:
                        supersede_failures.append(f"{old_tome_id_str} (not found)")
                except (ValueError, StorageError) as exc:
                    logging.warning(
                        "Failed to mark %s as superseded by %s: %s",
                        old_tome_id_str,
                        new_tome_id,
                        exc,
                    )
                    supersede_failures.append(f"{old_tome_id_str} ({exc})")
            if supersede_failures:
                # The caller asked for supersession that did not happen — the
                # old tomes are still live in search. Reporting STORED here
                # would hide that, so degrade to PARTIAL with the reason.
                any_rejected = True
                reject_reasons.append(
                    "Failed to mark tomes as superseded (they remain live in search): "
                    + ", ".join(supersede_failures)
                )

        status = IngestStatus.PARTIAL if any_rejected else IngestStatus.STORED
        final_reason = constants.JOIN_SEPARATOR.join(reject_reasons) if reject_reasons else None
        return IngestOutput(tomes=stored, status=status, reject_reason=final_reason)

    async def _process_text(self, text: str, opts: IngestCallOptions) -> list[Tome] | None:
        """Verify, classify, embed, and dedup/store a single text.

        Returns None if verification is enabled and confidence is below the reject
        threshold.  When verification is disabled the text is stored unconditionally
        with a confidence of ingest.unverified_confidence.
        """
        tome = await self._build_tome(text, opts)
        if tome is None:
            return None
        return await self._dedup_and_store(tome, opts)

    async def _build_tome(self, text: str, opts: IngestCallOptions) -> Tome | None:
        """Verify, classify, and embed a text, returning a Tome ready for storage.

        Returns None if verification rejects the text.  Does NOT persist anything.
        """
        should_verify = self._config.verification.enabled and not opts.skip_verify
        verification_summary: VerificationSummary | None
        if should_verify:
            verification_result = await self._verifier.verify(text)
            if verification_result.confidence < self._config.verification.reject_threshold:
                return None
            confidence = verification_result.confidence
            verification_summary = verification_result.to_summary()
        else:
            confidence = self._config.ingest.unverified_confidence
            # Surface "skipped because the caller asked us to / verification is
            # disabled" so search consumers can distinguish it from a real run
            # with zero claims (e.g. offline verifier).
            verification_summary = VerificationSummary(skipped=True)

        if opts.confidence is not None:
            confidence = min(1.0, max(0.0, opts.confidence))

        (category, tags), (title, summary), embedding = await asyncio.gather(
            self._classify_and_tag(text, opts.category_hint, opts.tags_hint),
            self._generate_title_and_summary(text),
            self._embedding_service.embed(text),
        )

        # Append caller-supplied extra tags (e.g. researcher job topic) AFTER
        # classification so they accompany — rather than replace — the
        # content-derived tag set. Cap each tag length and the list size to
        # keep tag UIs and indexes bounded.
        tags = _cap_tags([*tags, *(opts.extra_tags or [])])

        # Compute thread_position: if thread_id is set, query repo for max position
        thread_position: int | None = None
        if opts.thread_id is not None:
            existing = await self._tome_repo.list_all(thread_id=opts.thread_id)
            positions = [t.thread_position for t in existing if t.thread_position is not None]
            max_position = max(positions, default=-1)
            thread_position = max_position + 1

        tome = Tome(
            id=uuid.uuid4(),
            content=text,
            summary=summary,
            title=title,
            category=category,
            tags=tags,
            embedding=embedding,
            source_url=opts.source_url,
            source_type=opts.source_type,
            confidence=confidence,
            research_job_id=opts.research_job_id,
            thread_id=opts.thread_id,
            thread_position=thread_position,
            verification=verification_summary,
            created_at=datetime.now(UTC),
        )
        self._validate(tome, allow_short=opts.allow_short)
        return tome

    async def consolidate(self, tomes: list[Tome], skip_verify: bool = False) -> list[Tome]:
        """Merge a list of existing tomes into a new set of resharded tomes.

        Useful for background 'garbage collection' or manual library tidying.
        """
        if not tomes:
            return []
        if len(tomes) == 1:
            return tomes

        # Build replacements from combined content.
        combined = constants.CONTENT_SEPARATOR.join([t.content for t in tomes])
        shards = await self._reshard(combined)

        # Deduplicate shards to avoid redundant tomes.
        unique_shards = list(dict.fromkeys([s.strip() for s in shards if s.strip()]))

        # Use the first tome's metadata as a hint for replacements.
        first = tomes[0]
        opts = IngestCallOptions(
            skip_verify=skip_verify,
            category_hint=first.category,
            tags_hint=first.tags,
            source_url=first.source_url,
            source_type=first.source_type,
            research_job_id=first.research_job_id,
        )

        replacements = await self._build_replacements(unique_shards, opts)

        if not replacements:
            return tomes  # Fallback to original if resharding failed/empty

        # Insert replacements. If any insert fails, compensating-delete
        # the ones that succeeded so we leave neither orphans (no replacement
        # without delete) nor partial duplication. Originals stay untouched.
        insert_results = await self._run_in_batches(
            [self._tome_repo.insert(r) for r in replacements],
            self._config.ingest.write_batch_size,
            return_exceptions=True,
        )
        inserted_ok: list[Tome] = []
        insert_errors: list[str] = []
        first_error: BaseException | None = None
        for replacement, insert_result in zip(replacements, insert_results, strict=True):
            if isinstance(insert_result, BaseException):
                logging.warning(
                    "Exception inserting replacement %s during consolidate: %s",
                    replacement.id,
                    insert_result,
                )
                insert_errors.append(str(replacement.id))
                if not first_error:
                    first_error = insert_result
            else:
                inserted_ok.append(replacement)

        if insert_errors:
            # Best-effort rollback of the partial inserts.
            rollback_results = await asyncio.gather(
                *[self._tome_repo.delete(r.id) for r in inserted_ok],
                return_exceptions=True,
            )
            residual_ids = [
                str(r.id)
                for r, res in zip(inserted_ok, rollback_results, strict=True)
                if isinstance(res, BaseException) or not res
            ]
            if residual_ids:
                logging.error(
                    "Consolidate rollback left residual replacements in store: %s",
                    constants.ID_SEPARATOR.join(residual_ids),
                )
            failed_ids = constants.ID_SEPARATOR.join(insert_errors)
            msg = f"Consolidate aborted: insert failure for replacement(s) {failed_ids}"
            if first_error:
                msg += f" (first error: {first_error})"
            raise ReshardError(msg, tomes=[])

        # Delete old tomes.
        delete_results = await self._run_in_batches(
            [self._tome_repo.delete(tome.id) for tome in tomes],
            self._config.ingest.write_batch_size,
            return_exceptions=True,
        )
        delete_errors = []
        for tome, delete_result in zip(tomes, delete_results, strict=True):
            if isinstance(delete_result, Exception):
                logging.warning(
                    "Exception deleting %s during consolidate: %s", tome.id, delete_result
                )
                delete_errors.append(str(tome.id))
            elif not delete_result:
                delete_errors.append(str(tome.id))

        if delete_errors:
            failed_ids = constants.ID_SEPARATOR.join(delete_errors)
            msg = (
                "Failed to delete tomes during consolidate "
                f"(duplicate data may exist for: {failed_ids})"
            )
            raise ReshardError(msg, tomes=replacements)

        return replacements

    async def _dedup_and_store(self, tome: Tome, opts: IngestCallOptions) -> list[Tome]:
        """Insert tome, or reshard with any near-duplicates found in the repository.

        Reshard strategy (safe ordering):
        1. Build and verify all replacement tomes from combined content.
        2. Abort without touching existing tomes if no replacement survives verification.
        3. Insert all replacements first to ensure no data loss if the operation is interrupted.
        4. Delete old tomes only once all replacements are successfully persisted.
        """
        try:
            duplicates = await self._tome_repo.find_near_duplicates(tome)
        except StorageError as exc:
            logging.error("Storage error during find_near_duplicates for %s: %s", tome.id, exc)
            raise ReshardError(f"Storage error during dedup scan: {exc}", tomes=[]) from exc
        if not duplicates:
            try:
                await self._tome_repo.insert(tome)
            except StorageError as exc:
                logging.error("Storage error during insert for %s: %s", tome.id, exc)
                raise ReshardError(f"Storage error during insert: {exc}", tomes=[]) from exc
            return [tome]

        # Step 1 — build replacements from combined content (nothing persisted yet).
        combined = constants.CONTENT_SEPARATOR.join(
            [d.content for d in duplicates] + [tome.content]
        )
        shards = await self._reshard(combined, opts)

        # Deduplicate shards to avoid redundant tomes.
        unique_shards = list(dict.fromkeys([s.strip() for s in shards if s.strip()]))
        replacements = await self._build_replacements(unique_shards, opts)

        # Step 2 — abort if verification left us with nothing to store.
        if not replacements:
            return []

        # Step 3 - Insert replacements. If any insert fails, compensating-delete
        # the ones that succeeded so we leave neither orphans (no replacement
        # without delete) nor partial duplication. Originals stay untouched.
        insert_results = await self._run_in_batches(
            [self._tome_repo.insert(r) for r in replacements],
            self._config.ingest.write_batch_size,
            return_exceptions=True,
        )
        inserted_ok: list[Tome] = []
        insert_errors: list[str] = []
        first_error: BaseException | None = None
        for replacement, insert_result in zip(replacements, insert_results, strict=True):
            if isinstance(insert_result, BaseException):
                logging.warning(
                    "Exception inserting replacement %s during reshard: %s",
                    replacement.id,
                    insert_result,
                )
                insert_errors.append(str(replacement.id))
                if not first_error:
                    first_error = insert_result
            else:
                inserted_ok.append(replacement)

        if insert_errors:
            # Best-effort rollback of the partial inserts.
            rollback_results = await asyncio.gather(
                *[self._tome_repo.delete(r.id) for r in inserted_ok],
                return_exceptions=True,
            )
            residual_ids = [
                str(r.id)
                for r, res in zip(inserted_ok, rollback_results, strict=True)
                if isinstance(res, BaseException) or not res
            ]
            if residual_ids:
                logging.error(
                    "Reshard rollback left residual replacements in store: %s",
                    constants.ID_SEPARATOR.join(residual_ids),
                )
            failed_ids = constants.ID_SEPARATOR.join(insert_errors)
            msg = f"Reshard aborted: insert failure for replacement(s) {failed_ids}"
            if first_error:
                msg += f" (first error: {first_error})"
            raise ReshardError(msg, tomes=[])

        # Step 4 - Delete old tomes.
        # Technically if we fail here, we may end up with duplicate data in the
        # library, but that seems like a better choice (IMO) than aborting.
        delete_results = await self._run_in_batches(
            [self._tome_repo.delete(dup.id) for dup in duplicates],
            self._config.ingest.write_batch_size,
            return_exceptions=True,
        )
        delete_errors = []
        for dup, delete_result in zip(duplicates, delete_results, strict=True):
            if isinstance(delete_result, BaseException):
                logging.warning("Exception deleting %s during reshard: %s", dup.id, delete_result)
                delete_errors.append(str(dup.id))
            elif not delete_result:
                delete_errors.append(str(dup.id))

        if delete_errors:
            failed_ids = constants.ID_SEPARATOR.join(delete_errors)
            msg = (
                "Failed to delete tomes during reshard "
                f"(duplicate data may exist for: {failed_ids})"
            )
            raise ReshardError(msg, tomes=replacements)

        return replacements

    async def _reshard(self, blob: str, opts: IngestCallOptions | None = None) -> list[str]:
        """Split blob into syntax-aware shards.

        Detects the input format (or honours ``opts.force_format``) and routes
        to a language-aware splitter. Code, config, and markdown skip the LLM
        "atomic facts" rewrite, which would destroy their syntax.

        For prose, enforces a minimum-size floor (chars and words) by bundling
        adjacent sub-floor shards together. Structured formats (code / json /
        yaml / markdown) intentionally skip floor bundling because that would
        break structural integrity (e.g. concatenating two JSON arrays). The
        caller must therefore mark structured ingests with ``allow_short=True``
        to bypass the per-tome floor in ``_validate``.
        """
        opts = opts or IngestCallOptions()

        fmt: DetectedFormat = opts.force_format or _detect_format(blob)

        # Whole-blob bypass for prose: a legitimately short document.
        if fmt == "text" and len(blob.strip()) < self._config.ingest.min_shard_chars:
            stripped = blob.strip()
            return [stripped] if stripped else []

        if fmt == "text":
            if self._config.ingest.use_llm_chunking:
                llm_shards = await self._extract_facts_llm(blob)
                if llm_shards:
                    bundled = self._apply_min_floor(llm_shards)
                    if bundled:
                        return bundled
                    # LLM returned only sub-floor fragments and bundling
                    # collapsed them away — fall back to the recursive
                    # splitter.
            return self._apply_min_floor(await self._split_text_recursive(blob))

        # Structured formats: preserve structural integrity, no floor bundling.
        return await self._split_structured(blob, fmt)

    async def _split_text_recursive(self, blob: str) -> list[str]:
        """Generic recursive character split — used as fallback for prose."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._config.ingest.shard_size,
            chunk_overlap=self._config.ingest.shard_overlap,
        )
        return await asyncio.to_thread(splitter.split_text, blob)

    def _apply_min_floor(self, shards: list[str]) -> list[str]:
        """Bundle adjacent shards in order until each meets the size floor.

        - Walks shards left-to-right, accumulating into a buffer until the
          buffer has both >= min_shard_chars and >= min_shard_words.
        - Any final undersized tail is merged into the previous shard.
        - Empty / whitespace-only shards are dropped.
        - Returns [] when the floor cannot be reached even after bundling
          everything (caller decides whether to fall back).
        """
        min_chars = self._config.ingest.min_shard_chars
        min_words = self._config.ingest.min_shard_words

        cleaned = [s.strip() for s in shards if s and s.strip()]
        if not cleaned:
            return []

        out: list[str] = []
        buf: str = ""
        for shard in cleaned:
            buf = shard if not buf else f"{buf}{constants.CONTENT_SEPARATOR}{shard}"
            if len(buf) >= min_chars and len(buf.split()) >= min_words:
                out.append(buf)
                buf = ""

        if buf:
            # Undersized tail: merge with the previous shard if any, else
            # accept-as-is so we don't lose data when the entire combined
            # input cannot reach the floor.
            if out:
                out[-1] = f"{out[-1]}{constants.CONTENT_SEPARATOR}{buf}"
            else:
                # Nothing met the floor — signal failure so the caller can
                # fall back to a different strategy.
                return []

        return out

    async def _build_replacements(self, shards: list[str], opts: IngestCallOptions) -> list[Tome]:
        replacement_results = await self._gather_limited(
            [self._build_tome(chunk, opts) for chunk in shards],
            self._config.ingest.build_concurrency,
        )
        return [tome for tome in replacement_results if tome is not None]

    async def _gather_limited(
        self,
        operations: list[Awaitable[T | None]],
        concurrency: int,
    ) -> list[T | None]:
        semaphore = asyncio.Semaphore(max(1, concurrency))

        async def _run(operation: Awaitable[T | None]) -> T | None:
            async with semaphore:
                return await operation

        return await asyncio.gather(*[_run(operation) for operation in operations])

    async def _run_in_batches(
        self,
        operations: list[Awaitable[T]],
        batch_size: int,
        *,
        return_exceptions: bool = False,
    ) -> list[T | BaseException]:
        results: list[T | BaseException] = []
        for start in range(0, len(operations), max(1, batch_size)):
            batch = operations[start : start + max(1, batch_size)]
            batch_results = await asyncio.gather(*batch, return_exceptions=return_exceptions)
            results.extend(batch_results)
        return results

    @staticmethod
    def _strip_code_fences(message: str) -> str:
        """Strip surrounding ```json ... ``` markdown fences from an LLM response."""
        message = message.strip()
        if message.startswith("```"):
            message = re.sub(r"^```(?:json)?\s*", "", message)
            message = re.sub(r"\s*```$", "", message)
        return message

    async def _split_structured(self, blob: str, fmt: DetectedFormat) -> list[str]:
        """Format-aware splitting for code / markdown / yaml / json.

        ``RecursiveCharacterTextSplitter.split_text`` is CPU-bound (synchronous
        Python with potentially heavy regex work on large blobs). Every call to
        it is offloaded via ``asyncio.to_thread`` so the event loop stays
        responsive — see issue #24.
        """
        chunk_size = self._config.ingest.shard_size
        chunk_overlap = self._config.ingest.shard_overlap

        if fmt in ("python", "code"):
            splitter = RecursiveCharacterTextSplitter.from_language(
                Language.PYTHON,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            return await asyncio.to_thread(splitter.split_text, blob)

        if fmt == "markdown":
            splitter = RecursiveCharacterTextSplitter.from_language(
                Language.MARKDOWN,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            shards = await asyncio.to_thread(splitter.split_text, blob)
            return [_balance_markdown_fences(s) for s in shards]

        if fmt == "yaml":
            return await self._split_yaml(blob)

        if fmt == "json":
            return await self._split_json(blob)

        # Unreachable given DetectedFormat literal — defensive fallback.
        return await self._split_text_recursive(blob)

    async def _split_yaml(self, blob: str) -> list[str]:
        """Split YAML at top-level keys; coalesce small keys to fill ``shard_size``.

        We collect each top-level key (and its indented body) as a "section",
        then *pack* sections greedily into chunks while staying under
        ``shard_size``. Oversized single sections are handed to the recursive
        splitter. This avoids excessive fragmentation on configs with many
        small keys (per Gemini code-review feedback on PR #62).

        Oversized sections route through ``RecursiveCharacterTextSplitter``,
        which is offloaded via ``asyncio.to_thread`` to keep the event loop
        responsive (issue #24).
        """
        # Group top-level keys (lines starting at column 0 with `key:`).
        lines = blob.splitlines(keepends=True)
        sections: list[str] = []
        current: list[str] = []
        top_key = re.compile(r"^[A-Za-z_][\w-]*:(?:\s|$)")
        for line in lines:
            if top_key.match(line) and current:
                section = "".join(current).strip("\n")
                if section.strip():
                    sections.append(section)
                current = [line]
            else:
                current.append(line)
        if current:
            section = "".join(current).strip("\n")
            if section.strip():
                sections.append(section)

        if not sections:
            return []

        shard_size = self._config.ingest.shard_size
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=shard_size,
            chunk_overlap=self._config.ingest.shard_overlap,
            separators=["\n\n", "\n- ", "\n", " "],
        )

        # Greedily pack sections into chunks under shard_size; split oversized
        # sections recursively.
        out: list[str] = []
        buf: list[str] = []
        buf_len = 0
        for section in sections:
            if len(section) > shard_size:
                # Flush current buffer first to preserve order.
                if buf:
                    out.append("\n".join(buf))
                    buf, buf_len = [], 0
                out.extend(await asyncio.to_thread(splitter.split_text, section))
                continue
            # +1 accounts for the joining newline once a section is appended.
            projected = buf_len + len(section) + (1 if buf else 0)
            if buf and projected > shard_size:
                out.append("\n".join(buf))
                buf, buf_len = [section], len(section)
            else:
                buf.append(section)
                buf_len = projected
        if buf:
            out.append("\n".join(buf))
        return out

    async def _split_json(self, blob: str) -> list[str]:
        """Split a JSON document at its top-level structural elements.

        For arrays we *greedily pack* multiple elements into a single JSON-array
        chunk, keeping each chunk under ``shard_size``. This avoids the extreme
        fragmentation that one-element-per-chunk would cause for arrays of many
        small objects (per Gemini code-review feedback on PR #62, comment 1).
        For objects we keep the dictionary intact (so each shard retains its
        surrounding-key context) and only fall through to the recursive
        splitter if the serialized form exceeds ``shard_size``.

        Oversized chunks route through ``RecursiveCharacterTextSplitter``,
        which is offloaded via ``asyncio.to_thread`` to keep the event loop
        responsive (issue #24). ``json.loads`` is also offloaded because
        parsing very large blobs is CPU-bound (Gemini review on this PR).
        """
        try:
            parsed = await asyncio.to_thread(json.loads, blob)
        except (json.JSONDecodeError, ValueError):
            parsed = None

        shard_size = self._config.ingest.shard_size

        chunks: list[str]
        if isinstance(parsed, list):
            chunks = self._pack_json_array(parsed, shard_size)
        elif isinstance(parsed, dict):
            # Preserve the whole object as one chunk; the size guard below
            # will only fragment it when it actually exceeds shard_size.
            chunks = [json.dumps(parsed, indent=2)]
        else:
            chunks = [blob]

        # Enforce shard_size: oversized chunks fall back to recursive splitting.
        # Separators target property boundaries inside large indent=2 objects
        # (per Gemini code-review feedback on PR #62, comment 3).
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=shard_size,
            chunk_overlap=self._config.ingest.shard_overlap,
            separators=[',\n  "', ',\n    "', ',\n      "', "\n", " ", ""],
        )
        out: list[str] = []
        for chunk in chunks:
            if len(chunk) <= shard_size:
                out.append(chunk)
            else:
                out.extend(await asyncio.to_thread(splitter.split_text, chunk))
        return [c for c in out if c.strip()]

    @staticmethod
    def _pack_json_array(elements: list[object], shard_size: int) -> list[str]:
        """Greedily pack array elements into JSON-array chunks under ``shard_size``.

        Each chunk is a valid JSON array (``[ ... ]``). Single elements that
        already exceed ``shard_size`` are emitted as their own chunk and let
        the size-guard / recursive splitter handle them downstream.
        """
        if not elements:
            return []

        out: list[str] = []
        buf: list[object] = []

        def flush() -> None:
            if buf:
                out.append(json.dumps(buf, indent=2))
                buf.clear()

        for elem in elements:
            serialized = json.dumps(elem, indent=2)
            # Oversized single element — flush current batch, emit on its own.
            if len(serialized) > shard_size:
                flush()
                out.append(serialized)
                continue
            # Project the size of buf + this element wrapped as a JSON array.
            trial = [*buf, elem]
            if len(json.dumps(trial, indent=2)) > shard_size and buf:
                flush()
                buf.append(elem)
            else:
                buf.append(elem)
        flush()
        return out

    async def _extract_facts_llm(self, blob: str) -> list[str] | None:
        """Use an LLM to **rewrite** text into atomic factual statements.

        WARNING — this is *not* chunking. Unlike the deterministic
        ``RecursiveCharacterTextSplitter`` fallback, the returned items are
        generated text that may differ in wording (and meaning) from the
        source.  The source is also silently truncated at
        ``ingest.llm_extraction_max_chars`` before the prompt is built; long
        inputs lose their tail without warning.  Enable
        ``ingest.use_llm_chunking`` only when these semantics are acceptable.
        """
        base = self._config.ingest.ollama_base_url.rstrip("/")
        model = self._config.ingest.extraction_model
        max_chars = self._config.ingest.llm_extraction_max_chars
        prompt = (
            "Decompose the following text into a list of atomic, self-contained factual "
            "statements or concepts. Each statement or concept must contain enough context "
            "to be fully understood on its own. "
            f"Do not exceed {self._config.ingest.shard_size} characters per fact. "
            f"Each fact must be at least {self._config.ingest.min_shard_chars} characters "
            f"and {self._config.ingest.min_shard_words} words; combine related sub-facts to "
            "meet this minimum. "
            'Output JSON only with the shape `{"facts": ["...", "..."]}`.\n\nTEXT:\n'
            f"{blob[:max_chars]}"
        )
        url = f"{base}/v1/chat/completions"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
        }
        try:
            client = await self._get_http_client()
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            logging.debug("_extract_facts_llm HTTP/parse error: %s", exc)
            return None

        try:
            message = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            return None

        message = self._strip_code_fences(message)

        try:
            parsed = json.loads(message)
            facts = parsed.get("facts") if isinstance(parsed, dict) else None
            if not isinstance(facts, list):
                return None
            out = [str(f).strip() for f in facts if str(f).strip()]
        except (json.JSONDecodeError, TypeError, ValueError):
            return None

        if not out:
            return None
        return out

    async def _classify_and_tag(
        self,
        text: str,
        category_hint: str | None = None,
        tags_hint: list[str] | None = None,
    ) -> tuple[str, list[str]]:
        """Auto-classify into a category and extract topic tags.

        Resolution order:
        1. If both ``category_hint`` and ``tags_hint`` are supplied, short-circuit
           and skip the LLM entirely.
        2. Otherwise, when ``ingest.use_llm_classification`` is on, call the
           extraction model for a `(category, tags)` pair.
        3. Hints always win over LLM output for category; tag hints are merged
           uniformly with the chosen base tags (LLM tags when available, else
           ``ingest.default_tags``) — hints first, deduped, order-preserving.
        4. On any HTTP/parse error or out-of-taxonomy category, fall back to
           ``ingest.default_category`` / ``ingest.default_tags``.
        """
        default_category = self._config.ingest.default_category
        default_tags = list(self._config.ingest.default_tags)

        if category_hint and tags_hint:
            merged = list(dict.fromkeys([*tags_hint, *default_tags]))
            return category_hint, merged

        llm_category: str | None = None
        llm_tags: list[str] | None = None
        if self._config.ingest.use_llm_classification:
            try:
                llm_category, llm_tags = await self._classify_and_tag_llm(text)
            except (httpx.HTTPError, ValueError) as exc:
                logging.debug("_classify_and_tag LLM error: %s", exc)
                llm_category, llm_tags = None, None

        category = category_hint or llm_category or default_category

        # Uniform merge: prefer LLM tags when present, otherwise fall back to
        # configured defaults; user-supplied hints always take precedence and
        # are deduplicated while preserving order.
        base_tags = llm_tags if llm_tags else default_tags
        tags = list(dict.fromkeys([*(tags_hint or []), *base_tags]))

        return category, tags

    async def _classify_and_tag_llm(self, text: str) -> tuple[str | None, list[str] | None]:
        """Use the extraction model to pick a category from the configured taxonomy.

        Returns (category_or_None, tags_or_None).  ``category`` is ``None`` if the
        LLM picked something outside ``ingest.taxonomy``.  ``tags`` is ``None`` if
        the model produced no usable list.  Errors are *not* swallowed here — the
        caller catches them.
        """
        base = self._config.ingest.ollama_base_url.rstrip("/")
        model = self._config.ingest.extraction_model
        taxonomy = list(self._config.ingest.taxonomy)
        taxonomy_str = ", ".join(taxonomy)
        prompt = (
            "Classify the following text into exactly one category from this list: "
            f"[{taxonomy_str}]. Then list up to 5 short, lowercase topic tags. "
            'Output JSON only with the shape `{"category": "...", "tags": ["..."]}`.'
            "\n\nTEXT:\n"
            f"{text[:8000]}"
        )
        url = f"{base}/v1/chat/completions"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
        }

        client = await self._get_http_client()
        response = await client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()

        try:
            message = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            return None, None

        message = message.strip()
        # Extract JSON object even when the LLM wraps it in markdown fences or
        # prefixes conversational filler (e.g. "Sure, here is the JSON: ...").
        match = re.search(r"\{.*\}", message, re.DOTALL)
        if match:
            message = match.group(0)

        try:
            parsed = json.loads(message)
        except (json.JSONDecodeError, TypeError, ValueError):
            return None, None
        if not isinstance(parsed, dict):
            return None, None

        raw_category = parsed.get("category")
        category: str | None = None
        if isinstance(raw_category, str):
            # Case-insensitive match against taxonomy: LLMs often vary casing
            # (e.g. "science" vs "Science"). Preserve the canonical taxonomy
            # form so downstream filters stay consistent.
            normalised_request = raw_category.strip().lower()
            for item in taxonomy:
                if item.lower() == normalised_request:
                    category = item
                    break

        raw_tags = parsed.get("tags")
        tags: list[str] | None = None
        if isinstance(raw_tags, list):
            normalised: list[str] = []
            for t in raw_tags:
                if not isinstance(t, str):
                    continue
                cleaned = t.strip().lower()
                if cleaned:
                    normalised.append(cleaned)
            if normalised:
                tags = list(dict.fromkeys(normalised))[:5]

        return category, tags

    async def _generate_title_and_summary(self, text: str) -> tuple[str, str]:
        """Generate a short title and one-to-two sentence summary for a text.

        When ``ingest.use_llm_summary`` is enabled, ask an Ollama-compatible
        chat model to produce both fields as JSON.  Falls back to the
        deterministic truncation path on any error (network, timeout, malformed
        JSON, missing fields, etc.).
        """
        if self._config.ingest.use_llm_summary:
            llm_result = await self._summarize_llm(text)
            if llm_result is not None:
                return llm_result
        return self._truncation_title_and_summary(text)

    def _truncation_title_and_summary(self, text: str) -> tuple[str, str]:
        """Deterministic fallback: head-of-text truncation."""
        clean_text = text.strip().replace("\n", " ")
        return self._apply_title_summary_limits(clean_text, clean_text)

    async def _summarize_llm(self, text: str) -> tuple[str, str] | None:
        """Ask an LLM for a short title and 1–2 sentence summary as JSON.

        Returns ``None`` on any failure so the caller can fall back to the
        deterministic truncation path.  Mirrors :meth:`_extract_facts_llm`.
        """
        title_length = self._config.ingest.title_length
        summary_length = self._config.ingest.summary_length
        base = self._config.ingest.ollama_base_url.rstrip("/")
        model = self._config.ingest.extraction_model
        prompt = (
            "Read the text and return JSON only with shape "
            f'{{"title":"<={title_length} chars>","summary":"<1-2 sentences, '
            f'<={summary_length} chars>"}}.\n\nTEXT:\n'
            f"{text[: constants.MAX_LLM_INPUT_CHARS]}"
        )
        url = f"{base}/v1/chat/completions"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
        }
        try:
            client = await self._get_http_client()
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            logging.debug("_summarize_llm HTTP/parse error: %s", exc)
            return None

        try:
            message = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            return None

        message = self._strip_code_fences(message)

        try:
            parsed = json.loads(message)
        except (json.JSONDecodeError, TypeError, ValueError):
            return None
        if not isinstance(parsed, dict):
            return None

        raw_title = parsed.get("title")
        raw_summary = parsed.get("summary")
        if not isinstance(raw_title, str) or not isinstance(raw_summary, str):
            return None

        title = raw_title.strip().replace("\n", " ")
        summary = raw_summary.strip().replace("\n", " ")
        if not title or not summary:
            return None

        return self._apply_title_summary_limits(title, summary)

    def _apply_title_summary_limits(self, title: str, summary: str) -> tuple[str, str]:
        """Enforce the configured title/summary length caps with the truncation suffix."""
        title_length = self._config.ingest.title_length
        summary_length = self._config.ingest.summary_length
        if len(title) > title_length:
            suffix = constants.TRUNCATION_SUFFIX
            title = title[: title_length - len(suffix)] + suffix
        if len(summary) > summary_length:
            summary = summary[:summary_length]
        return title, summary

    def _validate(self, tome: Tome, *, allow_short: bool = False) -> None:
        """Post-construction checks; raises on failure.

        ``allow_short`` skips the minimum-shard-size floor — used when the
        whole ingest blob was legitimately smaller than the floor.
        """
        if not tome.content.strip():
            raise ValueError("Tome content cannot be empty")
        if len(tome.title) > self._config.ingest.title_length:
            raise ValueError(
                f"Tome title too long ({len(tome.title)} > {self._config.ingest.title_length})"
            )

        if not allow_short:
            min_chars = self._config.ingest.min_shard_chars
            min_words = self._config.ingest.min_shard_words
            content_len = len(tome.content)
            content_words = len(tome.content.split())
            if content_len < min_chars or content_words < min_words:
                raise ValueError(
                    f"Tome content below minimum shard size "
                    f"(chars={content_len}<{min_chars}, words={content_words}<{min_words})"
                )

        # Enforce embedding size against what the model actually produces, not the
        # configured value (which may be the default and disagree with the model).
        expected_dim = self._embedding_service.dimensions
        if tome.embedding is not None and tome.embedding.shape[0] != expected_dim:
            raise ValueError(
                f"Tome embedding has dimension {tome.embedding.shape[0]}, expected {expected_dim}"
            )
