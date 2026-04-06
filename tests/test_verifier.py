"""Tests for the web-backed Verifier."""

from __future__ import annotations

import pytest

from src.config import VerificationSettings
from src.services.verifier import Verifier
from src.services.web_search import (
    UnavailableWebSearchClient,
    WebSearchClient,
    WebSearchResult,
)
from tests.conftest import make_test_config


class _FakeWeb(WebSearchClient):
    def __init__(self, snippet: str) -> None:
        self._snippet = snippet

    async def search(self, query: str, max_results: int = 3) -> list[WebSearchResult]:
        return [
            WebSearchResult(
                title="ref",
                url="https://example.com/a",
                snippet=self._snippet,
            )
        ]

    def is_available(self) -> bool:
        return True


@pytest.mark.asyncio
async def test_verifier_skips_when_search_unavailable() -> None:
    config = make_test_config(verification=VerificationSettings(enabled=True))
    v = Verifier(config, UnavailableWebSearchClient())
    r = await v.verify("Anything.")
    assert r.skipped is True
    assert r.confidence == pytest.approx(config.verification.mock_confidence)


@pytest.mark.asyncio
async def test_verifier_scores_supported_claims() -> None:
    config = make_test_config(
        verification=VerificationSettings(enabled=True, use_llm_claims=False),
    )
    web = _FakeWeb("Photosynthesis converts light energy into chemical energy in plants.")
    v = Verifier(config, web)
    r = await v.verify(
        "Photosynthesis converts light energy into chemical energy in plants and algae."
    )
    assert r.skipped is False
    assert r.confidence >= 0.5
