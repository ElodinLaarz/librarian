from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from uuid import UUID

import httpx

from src import constants
from src.config import LibrarianConfig
from src.models.enums import IngestStatus, SourceType
from src.models.tome import Tome
from src.models.tool_schemas import IngestOutput
from src.services.embedding import EmbeddingService
from src.services.verifier import Verifier
from src.storage.tome_repository import TomeRepository


@dataclass
class IngestCallOptions:
    skip_verify: bool = False
    source_type: SourceType = SourceType.AGENT_INPUT
    source_url: str | None = None
    research_job_id: UUID | None = None
    category_hint: str | None = None
    tags_hint: list[str] | None = None
    allow_short: bool = False
    """When True, _validate skips the minimum-shard-size floor.

    Set automatically by ingest() when the entire input blob is below the
    floor (e.g. tweets, short notes) so legitimate short documents are not
    rejected.
    """


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

    async def ingest(self, blob: str, options: IngestCallOptions | None = None) -> IngestOutput:
        """Convert unstructured text into one or more Tomes and save them."""
        opts = options or IngestCallOptions()
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
                reject_reason=(
                    f"Unable to re-shard content "
                    f"({len(blob)} chars, shard_size={self._config.ingest.shard_size}, "
                    f"shard_overlap={self._config.ingest.shard_overlap})"
                ),
            )

        # Whole-blob bypass: if the entire input is below the minimum-shard
        # floor, treat it as a legitimately short document and skip the floor
        # enforcement during _validate.
        allow_short = len(blob.strip()) < self._config.ingest.min_shard_chars
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
        if should_verify:
            verification_result = await self._verifier.verify(text)
            if verification_result.confidence < self._config.verification.reject_threshold:
                return None
            confidence = verification_result.confidence
        else:
            confidence = self._config.ingest.unverified_confidence

        (category, tags), (title, summary), embedding = await asyncio.gather(
            self._classify_and_tag(text, opts.category_hint, opts.tags_hint),
            self._generate_title_and_summary(text),
            self._embedding_service.embed(text),
        )

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
            created_at=datetime.now(UTC),
        )
        try:
            self._validate(tome, allow_short=opts.allow_short)
        except ValueError:
            logging.error("Tome validation failed", exc_info=True)
            return None
        return tome

    async def _dedup_and_store(self, tome: Tome, opts: IngestCallOptions) -> list[Tome]:
        """Insert tome, or reshard with any near-duplicates found in the repository.

        Reshard strategy (safe ordering):
        1. Build and verify all replacement tomes from combined content.
        2. Abort without touching existing tomes if no replacement survives verification.
        3. Insert all replacements first to ensure no data loss if the operation is interrupted.
        4. Delete old tomes only once all replacements are successfully persisted.
        """
        duplicates = await self._tome_repo.find_near_duplicates(tome)
        if not duplicates:
            await self._tome_repo.insert(tome)
            return [tome]

        # Step 1 — build replacements from combined content (nothing persisted yet).
        combined = constants.CONTENT_SEPARATOR.join(
            [d.content for d in duplicates] + [tome.content]
        )
        shards = await self._reshard(combined)
        replacement_results = await asyncio.gather(*[self._build_tome(c, opts) for c in shards])
        replacements = [t for t in replacement_results if t is not None]

        # Step 2 — abort if verification left us with nothing to store.
        if not replacements:
            return []

        # Step 3 - Insert replacements. If any insert fails, compensating-delete
        # the ones that succeeded so we leave neither orphans (no replacement
        # without delete) nor partial duplication. Originals stay untouched.
        insert_results = await asyncio.gather(
            *[self._tome_repo.insert(r) for r in replacements],
            return_exceptions=True,
        )
        inserted_ok: list[Tome] = []
        insert_errors: list[str] = []
        for replacement, result in zip(replacements, insert_results, strict=True):
            if isinstance(result, BaseException):
                logging.warning(
                    "Exception inserting replacement %s during reshard: %s",
                    replacement.id,
                    result,
                )
                insert_errors.append(str(replacement.id))
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
            raise ReshardError(
                f"Reshard aborted: insert failure for replacement(s) {failed_ids}",
                tomes=[],
            )

        # Step 4 - Delete old tomes.
        # Technically if we fail here, we may end up with duplicate data in the
        # library, but that seems like a better choice (IMO) than aborting.
        delete_results = await asyncio.gather(
            *[self._tome_repo.delete(dup.id) for dup in duplicates],
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

    async def _reshard(self, blob: str) -> list[str]:
        """Split blob into atomic, self-contained fact shards.

        Enforces a minimum-size floor (chars and words) by bundling adjacent
        sub-floor shards together. Legitimate short inputs (whole blob below
        the floor) bypass bundling and pass through unchanged.
        """
        # Whole-blob bypass: a legitimately short document.
        if len(blob.strip()) < self._config.ingest.min_shard_chars:
            stripped = blob.strip()
            return [stripped] if stripped else []

        # Lazy import to keep startup light and to allow tests to monkeypatch.
        import langchain_text_splitters as _lcts

        def _heuristic() -> list[str]:
            splitter = _lcts.RecursiveCharacterTextSplitter(
                chunk_size=self._config.ingest.shard_size,
                chunk_overlap=self._config.ingest.shard_overlap,
            )
            return splitter.split_text(blob)

        if self._config.ingest.use_llm_chunking:
            llm_shards = await self._reshard_llm(blob)
            if llm_shards:
                bundled = self._apply_min_floor(llm_shards)
                if bundled:
                    return bundled
                # LLM returned only sub-floor fragments and bundling collapsed
                # them away — fall back to the heuristic splitter.

        return self._apply_min_floor(_heuristic())

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

    async def _reshard_llm(self, blob: str) -> list[str] | None:
        """Use an LLM agent to decompose text into atomic facts."""
        base = self._config.ingest.ollama_base_url.rstrip("/")
        model = self._config.ingest.extraction_model
        prompt = (
            "Decompose the following text into a list of atomic, self-contained factual "
            "statements or concepts. Each statement or concept must contain enough context "
            "to be fully understood on its own. "
            f"Do not exceed {self._config.ingest.shard_size} characters per fact. "
            f"Each fact must be at least {self._config.ingest.min_shard_chars} characters "
            f"and {self._config.ingest.min_shard_words} words; combine related sub-facts to "
            "meet this minimum. "
            'Output JSON only with the shape `{"facts": ["...", "..."]}`.\n\nTEXT:\n'
            f"{blob[:8000]}"
        )
        url = f"{base}/v1/chat/completions"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
        }
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
        except (httpx.HTTPError, OSError, ValueError, KeyError) as exc:
            logging.debug("_reshard_llm HTTP/parse error: %s", exc)
            return None

        try:
            message = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            return None

        message = message.strip()
        if message.startswith("```"):
            message = re.sub(r"(?i)^```\s*(?:json)?\s*", "", message)
            message = re.sub(r"\s*```$", "", message)

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
            except (httpx.HTTPError, OSError, ValueError) as exc:
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

        async with httpx.AsyncClient(timeout=120.0) as client:
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
        """Generate a short title and one-to-two sentence summary for a text."""
        clean_text = text.strip().replace("\n", " ")
        if len(clean_text) > self._config.ingest.title_length:
            suffix = constants.TRUNCATION_SUFFIX
            title = clean_text[: self._config.ingest.title_length - len(suffix)] + suffix
        else:
            title = clean_text
        summary = clean_text[: self._config.ingest.summary_length]
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
