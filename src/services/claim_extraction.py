"""Extract factual claims from text for verification (LLM or heuristic fallback)."""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

import httpx
from pydantic import BaseModel, Field

from src import constants
from src.config import LibrarianConfig

if TYPE_CHECKING:
    pass


class ClaimList(BaseModel):
    """Structured response for instructor / JSON parsing."""

    claims: list[str] = Field(
        ...,
        description="Standalone factual claims, each one short sentence.",
        min_length=1,
    )


def _heuristic_claims(content: str) -> list[str]:
    """Split into sentence-like units and take a bounded slice."""
    text = content.strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    claims = [p.strip() for p in parts if len(p.strip()) > 20]
    if not claims:
        claims = [text[:500]]
    return claims[: constants.MAX_CLAIMS]


async def extract_claims_llm(content: str, config: LibrarianConfig) -> list[str] | None:
    """Ask Ollama (OpenAI-compatible API) for JSON claims. Returns None on failure."""
    base = config.verification.ollama_base_url.rstrip("/")
    model = config.verification.claim_model
    prompt = (
        "Extract between 3 and 7 short, self-contained factual claims from the text below. "
        "Each claim must be a single sentence. Output JSON only with shape "
        '{"claims": ["...", "..."]}.\n\nTEXT:\n'
        f"{content[:6000]}"
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
        logging.debug("extract_claims_llm HTTP/parse error: %s", exc)
        return None

    try:
        message = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None

    message = message.strip()
    if message.startswith("```"):
        message = re.sub(r"^```(?:json)?\s*", "", message)
        message = re.sub(r"\s*```$", "", message)

    try:
        parsed = json.loads(message)
        claims = parsed.get("claims") if isinstance(parsed, dict) else None
        if not isinstance(claims, list):
            return None
        out = [str(c).strip() for c in claims if str(c).strip()]
    except (json.JSONDecodeError, TypeError, ValueError):
        return None

    if len(out) < constants.MIN_CLAIMS:
        return None
    return out[: constants.MAX_CLAIMS]


async def extract_claims(content: str, config: LibrarianConfig) -> list[str]:
    """Prefer LLM extraction when enabled; otherwise heuristic sentences."""
    if config.verification.use_llm_claims:
        llm_claims = await extract_claims_llm(content, config)
        if llm_claims:
            return llm_claims
    return _heuristic_claims(content)
