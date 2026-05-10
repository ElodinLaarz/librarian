from __future__ import annotations

import asyncio
import ipaddress
import logging
import os
import urllib.parse
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import httpx
import trafilatura
from pydantic import BaseModel

from src import constants
from src.config import LibrarianConfig, WebSearchSettings


class WebSearchResult(BaseModel):
    title: str
    url: str
    snippet: str


class WebSearchClient(ABC):
    """Abstract interface for web search.

    Concrete implementations (Brave, Serper, Tavily, etc.) must
    provide the methods below.
    """

    @abstractmethod
    async def search(
        self, query: str, max_results: int = constants.DEFAULT_MAX_RESULTS
    ) -> list[WebSearchResult]:
        """Issue a search query and return ranked results."""
        ...

    async def fetch_page_content(self, url: str) -> str:
        """Fetch a URL and extract the main body text, stripping boilerplate."""
        return await fetch_url_main_text(url)

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if the search backend is configured and reachable."""
        ...


_ALLOWED_SCHEMES = frozenset({"http", "https"})
_BLOCKED_HOSTNAMES = frozenset({"localhost", "metadata.google.internal"})
_MAX_REDIRECTS = 5


def _is_safe_ip(ip: str) -> bool:
    """Return True if `ip` is a routable, public IP (v4 or v6).

    Rejects loopback, link-local, multicast, private, reserved, unspecified,
    and IPv6 site-local addresses. This is the core SSRF guard predicate.
    """
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    if (
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_multicast
        or addr.is_reserved
        or addr.is_unspecified
    ):
        return False
    return not (isinstance(addr, ipaddress.IPv6Address) and addr.is_site_local)


async def _resolve_host_ips(host: str) -> list[str]:
    """Resolve `host` to all known IPs. Returns [] on failure.

    Defined at module scope so tests can monkeypatch DNS without touching
    the actual network.
    """
    try:
        loop = asyncio.get_running_loop()
        infos = await loop.getaddrinfo(host, None)
    except (OSError, asyncio.CancelledError):
        return []
    return [str(info[4][0]) for info in infos]


async def _validate_url(url: str) -> bool:
    """Validate `url` for SSRF safety.

    Checks: scheme allowlist, hostname blocklist, and that *every* DNS
    resolution result is a public, routable IP. Fails closed on any error.
    """
    try:
        parsed = urllib.parse.urlsplit(url)
    except ValueError:
        return False
    if parsed.scheme.lower() not in _ALLOWED_SCHEMES:
        return False
    host = parsed.hostname
    if not host:
        return False
    host_lower = host.lower()
    if host_lower in _BLOCKED_HOSTNAMES:
        return False
    # If host is itself a literal IP, validate it directly (no DNS needed).
    try:
        ipaddress.ip_address(host)
    except ValueError:
        ips = await _resolve_host_ips(host)
    else:
        ips = [host]
    if not ips:
        return False
    return all(_is_safe_ip(ip) for ip in ips)


async def fetch_url_main_text(url: str, *, timeout: float = 25.0) -> str:
    """Download HTML and return extracted main text (best-effort).

    Includes an SSRF guard: validates the scheme + DNS-resolved IPs of the
    initial URL and every redirect hop, refusing to follow redirects to
    private, loopback, link-local, or otherwise non-public addresses.
    """
    headers = {"User-Agent": "LibrarianBot/1.0 (+https://github.com/librarian)"}
    current_url = url
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
            html: str | None = None
            for _ in range(_MAX_REDIRECTS + 1):
                if not await _validate_url(current_url):
                    logging.debug(
                        "fetch_url_main_text refused unsafe URL: %s", current_url
                    )
                    return ""
                response = await client.get(current_url, headers=headers)
                if 300 <= response.status_code < 400:
                    location = response.headers.get("location")
                    if not location:
                        return ""
                    current_url = urllib.parse.urljoin(current_url, location)
                    continue
                response.raise_for_status()
                html = response.text
                break
            else:
                # Exceeded redirect budget.
                logging.debug(
                    "fetch_url_main_text exceeded redirect limit for %s", url
                )
                return ""
    except (httpx.HTTPError, OSError) as exc:
        logging.debug("fetch_url_main_text failed for %s: %s", url, exc)
        return ""
    if html is None:
        return ""
    extracted = trafilatura.extract(html)
    return extracted or ""


class UnavailableWebSearchClient(WebSearchClient):
    """Placeholder when no search API key is configured."""

    async def search(
        self, query: str, max_results: int = constants.DEFAULT_MAX_RESULTS
    ) -> list[WebSearchResult]:
        return []

    def is_available(self) -> bool:
        return False


class BraveWebSearchClient(WebSearchClient):
    def __init__(self, settings: WebSearchSettings, api_key: str) -> None:
        self._settings = settings
        self._api_key = api_key

    async def search(
        self, query: str, max_results: int = constants.DEFAULT_MAX_RESULTS
    ) -> list[WebSearchResult]:
        params: Mapping[str, str | int] = {"q": query, "count": max_results}
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self._api_key,
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                self._settings.brave_api_base, params=params, headers=headers
            )
            response.raise_for_status()
            data: dict[str, Any] = response.json()

        web_block = data.get("web") or {}
        raw_results = web_block.get("results") or []
        out: list[WebSearchResult] = []
        for item in raw_results[:max_results]:
            out.append(
                WebSearchResult(
                    title=str(item.get("title") or ""),
                    url=str(item.get("url") or ""),
                    snippet=str(item.get("description") or item.get("snippet") or ""),
                )
            )
        return out

    def is_available(self) -> bool:
        return bool(self._api_key.strip())


class SerperWebSearchClient(WebSearchClient):
    def __init__(self, settings: WebSearchSettings, api_key: str) -> None:
        self._settings = settings
        self._api_key = api_key

    async def search(
        self, query: str, max_results: int = constants.DEFAULT_MAX_RESULTS
    ) -> list[WebSearchResult]:
        url = f"{self._settings.serper_api_base.rstrip('/')}/search"
        payload = {"q": query, "num": max_results}
        headers = {"X-API-KEY": self._api_key, "Content-Type": "application/json"}
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data: dict[str, Any] = response.json()

        organic = data.get("organic") or []
        out: list[WebSearchResult] = []
        for item in organic[:max_results]:
            out.append(
                WebSearchResult(
                    title=str(item.get("title") or ""),
                    url=str(item.get("link") or ""),
                    snippet=str(item.get("snippet") or ""),
                )
            )
        return out

    def is_available(self) -> bool:
        return bool(self._api_key.strip())


class TavilyWebSearchClient(WebSearchClient):
    def __init__(self, settings: WebSearchSettings, api_key: str) -> None:
        self._settings = settings
        self._api_key = api_key

    async def search(
        self, query: str, max_results: int = constants.DEFAULT_MAX_RESULTS
    ) -> list[WebSearchResult]:
        url = f"{self._settings.tavily_api_base.rstrip('/')}/search"
        payload = {
            "api_key": self._api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data: dict[str, Any] = response.json()

        raw_results = data.get("results") or []
        out: list[WebSearchResult] = []
        for item in raw_results[:max_results]:
            out.append(
                WebSearchResult(
                    title=str(item.get("title") or ""),
                    url=str(item.get("url") or ""),
                    snippet=str(item.get("content") or ""),
                )
            )
        return out

    def is_available(self) -> bool:
        return bool(self._api_key.strip())


def _resolve_api_key(settings: WebSearchSettings) -> tuple[str, str | None]:
    """Return (provider, api_key_or_none)."""
    provider = (settings.provider or "brave").strip().lower()
    if provider in {"", "none", "off", "disabled"}:
        return provider, None

    explicit = (settings.api_key or "").strip()
    env_candidates: dict[str, tuple[str, ...]] = {
        "brave": ("BRAVE_API_KEY", "SEARCH_API_KEY"),
        "serper": ("SERPER_API_KEY", "SEARCH_API_KEY"),
        "tavily": ("TAVILY_API_KEY", "SEARCH_API_KEY"),
    }
    keys_to_try = env_candidates.get(provider, ("SEARCH_API_KEY",))
    for var in keys_to_try:
        env_val = os.environ.get(var, "").strip()
        if env_val:
            return provider, explicit or env_val
    return provider, explicit or None


def build_web_search_client(config: LibrarianConfig) -> WebSearchClient:
    """Instantiate the configured web search client (or unavailable stub)."""
    settings = config.web_search
    provider, key = _resolve_api_key(settings)
    if not key or provider in {"", "none", "off", "disabled"}:
        return UnavailableWebSearchClient()

    if provider == "brave":
        return BraveWebSearchClient(settings, key)
    if provider == "serper":
        return SerperWebSearchClient(settings, key)
    if provider == "tavily":
        return TavilyWebSearchClient(settings, key)

    logging.warning("Unknown web search provider %r; search disabled.", provider)
    return UnavailableWebSearchClient()
