"""Tests for web_search.py — factory, backends, and fetch helpers."""

from __future__ import annotations

import pytest

from src.config import WebSearchSettings
from src.services.web_search import (
    BraveWebSearchClient,
    SerperWebSearchClient,
    TavilyWebSearchClient,
    UnavailableWebSearchClient,
    build_web_search_client,
    fetch_url_main_text,
)
from tests.conftest import make_test_config

# ── UnavailableWebSearchClient ───────────────────────────────────────────────


def test_unavailable_client_is_not_available() -> None:
    assert UnavailableWebSearchClient().is_available() is False


async def test_unavailable_client_search_returns_empty() -> None:
    results = await UnavailableWebSearchClient().search("anything")
    assert results == []


# ── build_web_search_client factory ─────────────────────────────────────────


def test_build_returns_unavailable_when_no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    monkeypatch.delenv("SERPER_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("SEARCH_API_KEY", raising=False)
    config = make_test_config()
    client = build_web_search_client(config)
    assert isinstance(client, UnavailableWebSearchClient)


def test_build_returns_brave_when_key_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BRAVE_API_KEY", "test-key")
    from src.config import WebSearchSettings

    config = make_test_config(web_search=WebSearchSettings(provider="brave"))
    client = build_web_search_client(config)
    assert isinstance(client, BraveWebSearchClient)
    assert client.is_available()


def test_build_returns_serper_when_key_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SERPER_API_KEY", "test-key")
    from src.config import WebSearchSettings

    config = make_test_config(web_search=WebSearchSettings(provider="serper"))
    client = build_web_search_client(config)
    assert isinstance(client, SerperWebSearchClient)
    assert client.is_available()


def test_build_returns_tavily_when_key_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    from src.config import WebSearchSettings

    config = make_test_config(web_search=WebSearchSettings(provider="tavily"))
    client = build_web_search_client(config)
    assert isinstance(client, TavilyWebSearchClient)
    assert client.is_available()


def test_build_returns_unavailable_for_unknown_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.config import WebSearchSettings

    monkeypatch.setenv("SEARCH_API_KEY", "test-key")
    config = make_test_config(web_search=WebSearchSettings(provider="unknown_engine"))
    client = build_web_search_client(config)
    assert isinstance(client, UnavailableWebSearchClient)


# ── BraveWebSearchClient.search ──────────────────────────────────────────────


async def test_brave_search_parses_response(monkeypatch: pytest.MonkeyPatch) -> None:
    from unittest.mock import AsyncMock, MagicMock

    api_response = {
        "web": {
            "results": [
                {"title": "Result 1", "url": "https://example.com/1", "description": "Snippet 1"},
                {"title": "Result 2", "url": "https://example.com/2", "description": "Snippet 2"},
            ]
        }
    }
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json = MagicMock(return_value=api_response)

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=mock_response)

    monkeypatch.setattr("httpx.AsyncClient", lambda **kwargs: mock_client)

    client = BraveWebSearchClient(WebSearchSettings(), "fake-key")
    results = await client.search("test query", max_results=2)

    assert len(results) == 2
    assert results[0].title == "Result 1"
    assert results[0].url == "https://example.com/1"
    assert results[0].snippet == "Snippet 1"


async def test_brave_search_returns_empty_on_no_web_block(monkeypatch: pytest.MonkeyPatch) -> None:
    from unittest.mock import AsyncMock, MagicMock

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json = MagicMock(return_value={})

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=mock_response)

    monkeypatch.setattr("httpx.AsyncClient", lambda **kwargs: mock_client)

    client = BraveWebSearchClient(WebSearchSettings(), "fake-key")
    results = await client.search("test")
    assert results == []


# ── SerperWebSearchClient.search ─────────────────────────────────────────────


async def test_serper_search_parses_response(monkeypatch: pytest.MonkeyPatch) -> None:
    from unittest.mock import AsyncMock, MagicMock

    api_response = {
        "organic": [
            {"title": "Serper 1", "link": "https://example.com/s1", "snippet": "Serper snip 1"},
        ]
    }
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json = MagicMock(return_value=api_response)

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_response)

    monkeypatch.setattr("httpx.AsyncClient", lambda **kwargs: mock_client)

    client = SerperWebSearchClient(WebSearchSettings(), "fake-key")
    results = await client.search("test query")

    assert len(results) == 1
    assert results[0].url == "https://example.com/s1"


# ── TavilyWebSearchClient.search ─────────────────────────────────────────────


async def test_tavily_search_parses_response(monkeypatch: pytest.MonkeyPatch) -> None:
    from unittest.mock import AsyncMock, MagicMock

    api_response = {
        "results": [
            {"title": "Tavily 1", "url": "https://example.com/t1", "content": "Tavily content 1"},
        ]
    }
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json = MagicMock(return_value=api_response)

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_response)

    monkeypatch.setattr("httpx.AsyncClient", lambda **kwargs: mock_client)

    client = TavilyWebSearchClient(WebSearchSettings(), "fake-key")
    results = await client.search("test query")

    assert len(results) == 1
    assert results[0].snippet == "Tavily content 1"


# ── fetch_url_main_text ───────────────────────────────────────────────────────


async def test_fetch_url_returns_empty_on_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from unittest.mock import AsyncMock

    import httpx

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))

    monkeypatch.setattr("httpx.AsyncClient", lambda **kwargs: mock_client)

    result = await fetch_url_main_text("https://example.com")
    assert result == ""
