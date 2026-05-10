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
    monkeypatch.delenv("LIBRARIAN_WEB_SEARCH_API_KEY", raising=False)
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


# ── SSRF guard ───────────────────────────────────────────────────────────────


def _stub_resolver(monkeypatch: pytest.MonkeyPatch, mapping: dict[str, list[str]]) -> None:
    """Replace `_resolve_host_ips` so we can control DNS in tests."""

    async def fake_resolve(host: str) -> list[str]:
        return list(mapping.get(host, []))

    monkeypatch.setattr("src.services.web_search._resolve_host_ips", fake_resolve)


def _stub_httpx_get(
    monkeypatch: pytest.MonkeyPatch,
    *,
    status_code: int = 200,
    text: str = "<html><body><p>hello world this is some content</p></body></html>",
    headers: dict[str, str] | None = None,
) -> dict[str, int]:
    """Replace httpx.AsyncClient with a stub that records the number of GETs."""
    from unittest.mock import AsyncMock, MagicMock

    counter = {"calls": 0}

    async def fake_get(url: str, **_: object) -> MagicMock:
        counter["calls"] += 1
        resp = MagicMock()
        resp.status_code = status_code
        resp.text = text
        resp.headers = headers or {}
        resp.raise_for_status = MagicMock()
        if status_code >= 400:
            import httpx

            req = MagicMock()
            resp.raise_for_status.side_effect = httpx.HTTPStatusError(
                "boom", request=req, response=resp
            )
        return resp

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(side_effect=fake_get)

    monkeypatch.setattr("httpx.AsyncClient", lambda **kwargs: mock_client)
    return counter


async def test_fetch_url_rejects_file_scheme(monkeypatch: pytest.MonkeyPatch) -> None:
    counter = _stub_httpx_get(monkeypatch)
    assert await fetch_url_main_text("file:///etc/passwd") == ""
    assert counter["calls"] == 0


@pytest.mark.parametrize(
    "host,ips",
    [
        ("127.0.0.1", ["127.0.0.1"]),
        ("[::1]", ["::1"]),
        ("0.0.0.0", ["0.0.0.0"]),
        ("localhost", ["127.0.0.1"]),
    ],
)
async def test_fetch_url_rejects_localhost(
    monkeypatch: pytest.MonkeyPatch, host: str, ips: list[str]
) -> None:
    # localhost lookups must fail closed; bare IPs are validated directly.
    resolve_key = "localhost" if host == "localhost" else host.strip("[]")
    _stub_resolver(monkeypatch, {resolve_key: ips})
    counter = _stub_httpx_get(monkeypatch)
    assert await fetch_url_main_text(f"http://{host}/") == ""
    assert counter["calls"] == 0


async def test_fetch_url_rejects_aws_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_resolver(monkeypatch, {"169.254.169.254": ["169.254.169.254"]})
    counter = _stub_httpx_get(monkeypatch)
    assert await fetch_url_main_text("http://169.254.169.254/latest/meta-data/") == ""
    assert counter["calls"] == 0


@pytest.mark.parametrize(
    "host",
    [
        "10.0.0.1",
        "172.16.0.1",
        "192.168.1.1",
        "[fc00::1]",
        "[fe80::1]",
    ],
)
async def test_fetch_url_rejects_rfc1918(monkeypatch: pytest.MonkeyPatch, host: str) -> None:
    bare = host.strip("[]")
    _stub_resolver(monkeypatch, {bare: [bare]})
    counter = _stub_httpx_get(monkeypatch)
    assert await fetch_url_main_text(f"http://{host}/") == ""
    assert counter["calls"] == 0


async def test_fetch_url_allows_public_host(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_resolver(monkeypatch, {"example.com": ["93.184.216.34"]})
    counter = _stub_httpx_get(
        monkeypatch,
        text=(
            "<html><body><article><p>"
            "This is a public page with enough text for trafilatura to extract."
            " More content goes here so the extractor returns a non-empty result."
            "</p></article></body></html>"
        ),
    )
    result = await fetch_url_main_text("https://example.com/")
    assert counter["calls"] == 1
    # Don't assert exact content (trafilatura may or may not yield text); only
    # require that we did issue exactly one GET to the public host.
    assert isinstance(result, str)


async def test_fetch_url_rejects_redirect_to_private_ip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import AsyncMock, MagicMock

    _stub_resolver(
        monkeypatch,
        {
            "example.com": ["93.184.216.34"],
            "169.254.169.254": ["169.254.169.254"],
        },
    )

    counter = {"calls": 0}

    async def fake_get(url: str, **_: object) -> MagicMock:
        counter["calls"] += 1
        resp = MagicMock()
        # First (and only legal) hop returns a redirect to AWS metadata.
        resp.status_code = 302
        resp.headers = {"location": "http://169.254.169.254/latest/meta-data/"}
        resp.text = ""
        resp.raise_for_status = MagicMock()
        return resp

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(side_effect=fake_get)

    monkeypatch.setattr("httpx.AsyncClient", lambda **kwargs: mock_client)

    result = await fetch_url_main_text("https://example.com/")
    assert result == ""
    # Only the first request should have been issued; redirect target rejected.
    assert counter["calls"] == 1


async def test_fetch_url_rejects_dns_rebind(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_resolver(monkeypatch, {"rebind.example": ["1.1.1.1", "127.0.0.1"]})
    counter = _stub_httpx_get(monkeypatch)
    assert await fetch_url_main_text("https://rebind.example/") == ""
    assert counter["calls"] == 0
