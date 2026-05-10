from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal
from uuid import UUID

import httpx
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from src import constants
from src.config import LibrarianConfig
from src.models.enums import IngestStatus, SourceType
from src.models.tome import Tome
from src.models.tool_schemas import IngestOutput
from src.services.embedding import EmbeddingService
from src.services.verifier import Verifier
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


@dataclass
class IngestCallOptions:
    skip_verify: bool = False
    source_type: SourceType = SourceType.AGENT_INPUT
    source_url: str | None = None
    research_job_id: UUID | None = None
    category_hint: str | None = None
    tags_hint: list[str] | None = None
    force_format: DetectedFormat | None = None


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

        tomes = await asyncio.gather(
            *[self._process_text(s, opts) for s in shards],
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
            self._validate(tome)
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
        shards = await self._reshard(combined, opts)
        replacement_results = await asyncio.gather(*[self._build_tome(c, opts) for c in shards])
        replacements = [t for t in replacement_results if t is not None]

        # Step 2 — abort if verification left us with nothing to store.
        if not replacements:
            return []

        # Step 3 - Insert replacements.
        await asyncio.gather(*[self._tome_repo.insert(r) for r in replacements])

        # Step 4 - Delete old tomes.
        # Technically if we fail here, we may end up with duplicate data in the
        # library, but that seems like a better choice (IMO) than aborting.
        delete_results = await asyncio.gather(
            *[self._tome_repo.delete(dup.id) for dup in duplicates],
            return_exceptions=True,
        )
        delete_errors = []
        for dup, result in zip(duplicates, delete_results, strict=True):
            if isinstance(result, Exception):
                logging.warning("Exception deleting %s during reshard: %s", dup.id, result)
                delete_errors.append(str(dup.id))
            elif not result:
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
        """
        opts = opts or IngestCallOptions()
        fmt: DetectedFormat = opts.force_format or _detect_format(blob)

        if fmt == "text":
            if self._config.ingest.use_llm_chunking:
                llm_shards = await self._reshard_llm(blob)
                if llm_shards:
                    return llm_shards
            return self._split_text_recursive(blob)

        return self._split_structured(blob, fmt)

    def _split_text_recursive(self, blob: str) -> list[str]:
        """Generic recursive character split — used as fallback for prose."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._config.ingest.shard_size,
            chunk_overlap=self._config.ingest.shard_overlap,
        )
        return splitter.split_text(blob)

    def _split_structured(self, blob: str, fmt: DetectedFormat) -> list[str]:
        """Format-aware splitting for code / markdown / yaml / json."""
        chunk_size = self._config.ingest.shard_size
        chunk_overlap = self._config.ingest.shard_overlap

        if fmt in ("python", "code"):
            splitter = RecursiveCharacterTextSplitter.from_language(
                Language.PYTHON,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            return splitter.split_text(blob)

        if fmt == "markdown":
            splitter = RecursiveCharacterTextSplitter.from_language(
                Language.MARKDOWN,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            shards = splitter.split_text(blob)
            return [_balance_markdown_fences(s) for s in shards]

        if fmt == "yaml":
            return self._split_yaml(blob)

        if fmt == "json":
            return self._split_json(blob)

        # Unreachable given DetectedFormat literal — defensive fallback.
        return self._split_text_recursive(blob)

    def _split_yaml(self, blob: str) -> list[str]:
        """Split YAML at top-level keys; coalesce small keys to fill ``shard_size``.

        We collect each top-level key (and its indented body) as a "section",
        then *pack* sections greedily into chunks while staying under
        ``shard_size``. Oversized single sections are handed to the recursive
        splitter. This avoids excessive fragmentation on configs with many
        small keys (per Gemini code-review feedback on PR #62).
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
                out.extend(splitter.split_text(section))
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

    def _split_json(self, blob: str) -> list[str]:
        """Split a JSON document at its top-level structural elements.

        For arrays we *greedily pack* multiple elements into a single JSON-array
        chunk, keeping each chunk under ``shard_size``. This avoids the extreme
        fragmentation that one-element-per-chunk would cause for arrays of many
        small objects (per Gemini code-review feedback on PR #62, comment 1).
        For objects we keep the dictionary intact (so each shard retains its
        surrounding-key context) and only fall through to the recursive
        splitter if the serialized form exceeds ``shard_size``.
        """
        try:
            parsed = json.loads(blob)
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
                out.extend(splitter.split_text(chunk))
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

    async def _reshard_llm(self, blob: str) -> list[str] | None:
        """Use an LLM agent to decompose text into atomic facts."""
        base = self._config.ingest.ollama_base_url.rstrip("/")
        model = self._config.ingest.extraction_model
        prompt = (
            "Decompose the following text into a list of atomic, self-contained factual "
            "statements or concepts. Each statement or concept must contain enough context "
            "to be fully understood on its own. "
            f"Do not exceed {self._config.ingest.shard_size} characters per fact. "
            'Output JSON only with the shape `{"facts": ["...", "..."]}`.\\n\\nTEXT:\\n'
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
            message = re.sub(r"^```(?:json)?\\s*", "", message)
            message = re.sub(r"\\s*```$", "", message)

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
        """Auto-classify into a category and extract topic tags."""
        category = category_hint or self._config.ingest.default_category
        if tags_hint:
            merged = list(dict.fromkeys([*tags_hint, *self._config.ingest.default_tags]))
            return category, merged
        return category, list(self._config.ingest.default_tags)

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

    def _validate(self, tome: Tome) -> None:
        """Post-construction checks; raises on failure."""
        if not tome.content.strip():
            raise ValueError("Tome content cannot be empty")
        if len(tome.title) > self._config.ingest.title_length:
            raise ValueError(
                f"Tome title too long ({len(tome.title)} > {self._config.ingest.title_length})"
            )

        # In a real implementation we would enforce the embedding size.
        # Here we skip the dimension check if a dummy/string embedding is supplied.
        if (
            tome.embedding is not None
            and tome.embedding.shape[0] != self._config.embedding.dimensions
        ):
            raise ValueError(
                f"Tome embedding has dimension {tome.embedding.shape[0]}, "
                f"expected {self._config.embedding.dimensions}"
            )
