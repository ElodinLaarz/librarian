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
    the actual network. ``getaddrinfo`` may return the same address under
    multiple socket types, so we de-duplicate before returning.

    ``asyncio.CancelledError`` is intentionally **not** caught — cancellation
    must propagate so the surrounding task can shut down cleanly.
    """
    try:
        loop = asyncio.get_running_loop()
        infos = await loop.getaddrinfo(host, None)
    except OSError:
        return []
    return list({str(info[4][0]) for info in infos})


async def _validate_url(url: str) -> tuple[bool, str | None]:
    """Validate `url` for SSRF safety, returning (is_safe, pinned_ip).

    Checks: scheme allowlist, hostname blocklist, and that *every* DNS
    resolution result is a public, routable IP. Fails closed on any error.

    Returns (False, None) if validation fails.
    Returns (True, pinned_ip) if validation succeeds; pinned_ip is the first
    safe IP from DNS resolution (or the literal IP if host is already an IP).
    """
    try:
        parsed = urllib.parse.urlsplit(url)
    except ValueError:
        return False, None
    if parsed.scheme.lower() not in _ALLOWED_SCHEMES:
        return False, None
    host = parsed.hostname
    if not host:
        return False, None
    host_lower = host.lower()
    if host_lower in _BLOCKED_HOSTNAMES:
        return False, None
    # If host is itself a literal IP, validate it directly (no DNS needed).
    try:
        ipaddress.ip_address(host)
    except ValueError:
        ips = await _resolve_host_ips(host)
    else:
        ips = [host]
    if not ips:
        return False, None
    if not all(_is_safe_ip(ip) for ip in ips):
        return False, None
    # Return the first safe IP as the pinned IP.
    return True, ips[0]


class _PinnedDNSTransport(httpx.AsyncHTTPTransport):
    """Custom AsyncHTTPTransport that pins a validated IP.

    Prevents DNS rebinding TOCTOU: we resolve + validate DNS once during the
    validation phase, then use a custom transport that always connects to that
    pinned IP, even if DNS re-resolves to a different (malicious) address.
    """

    def __init__(self, pinned_ip: str, hostname: str) -> None:
        """
        Args:
            pinned_ip: The validated IP address to pin for all connections.
            hostname: The original hostname (preserved for SNI + Host header).
        """
        super().__init__()
        self._pinned_ip = pinned_ip
        self._hostname = hostname

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        """Override to replace hostname with pinned IP in the request URL.

        Sets the Host header and SNI hostname extension to preserve HTTPS
        certificate validation and virtual hosting support.
        """
        # Preserve the original hostname for HTTPS/SNI validation.
        headers = request.headers.copy()
        headers["host"] = self._hostname

        # Set SNI hostname in extensions for TLS handshake.
        extensions = dict(request.extensions)
        extensions["sni_hostname"] = self._hostname

        # Create a new URL with the pinned IP as the host.
        pinned_url = request.url.copy_with(host=self._pinned_ip)

        # httpx.Request is immutable, so create a new one with the modified URL.
        request = httpx.Request(
            method=request.method,
            url=pinned_url,
            content=request.content,
            headers=headers,
            extensions=extensions,
        )

        return await super().handle_async_request(request)


async def _fetch_url_main_text_inner(url: str, *, timeout: float) -> str:
    """Inner fetch loop. ``timeout`` is the per-HTTP-request budget;
    the caller wraps the whole call in ``asyncio.wait_for`` so the
    *overall* deadline (including DNS resolution) is bounded too.

    Uses a custom httpx transport that pins the validated IP to prevent
    DNS rebinding TOCTOU attacks.
    """
    headers = {"User-Agent": "LibrarianBot/1.0 (+https://github.com/librarian)"}
    current_url = url
    client = None
    try:
        html: str | None = None
        for _ in range(_MAX_REDIRECTS + 1):
            # Validate the URL and get the pinned IP.
            is_safe, pinned_ip = await _validate_url(current_url)
            if not is_safe or pinned_ip is None:
                logging.debug("fetch_url_main_text refused unsafe URL: %s", current_url)
                return ""

            # Extract hostname for transport creation.
            try:
                parsed = urllib.parse.urlsplit(current_url)
                hostname = parsed.hostname or ""
            except ValueError:
                return ""

            # Create a new client with pinned DNS transport for this URL.
            if client is not None:
                await client.aclose()
            transport = _PinnedDNSTransport(
                pinned_ip=pinned_ip,
                hostname=hostname,
            )
            client = httpx.AsyncClient(
                transport=transport,
                timeout=timeout,
                follow_redirects=False,
            )

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
            logging.debug("fetch_url_main_text exceeded redirect limit for %s", url)
            return ""
    except (httpx.HTTPError, OSError) as exc:
        logging.debug("fetch_url_main_text failed for %s: %s", url, exc)
        return ""
    finally:
        if client is not None:
            await client.aclose()

    if html is None:
        return ""
    # ``trafilatura.extract`` is CPU-bound and can be slow on large HTML;
    # offload it to a worker thread so we don't block the event loop.
    extracted = await asyncio.to_thread(trafilatura.extract, html)
    return extracted or ""


async def fetch_url_main_text(url: str, *, timeout: float = 25.0) -> str:
    """Download HTML and return extracted main text (best-effort).

    Includes an SSRF guard: validates the scheme + DNS-resolved IPs of the
    initial URL and every redirect hop, refusing to follow redirects to
    private, loopback, link-local, or otherwise non-public addresses.

    The ``timeout`` is a hard wall-clock cap on the entire operation —
    DNS resolution + every HTTP hop — enforced via ``asyncio.wait_for``.
    A slow/unresponsive resolver therefore cannot block longer than
    ``timeout``.
    """
    try:
        return await asyncio.wait_for(
            _fetch_url_main_text_inner(url, timeout=timeout),
            timeout=timeout,
        )
    except TimeoutError:
        # ``asyncio.TimeoutError`` is an alias of the builtin ``TimeoutError``
        # on Python 3.11+, which is the project's minimum (see pyproject.toml
        # ``requires-python = ">=3.11"``). Ruff's UP041 enforces the builtin.
        logging.debug("fetch_url_main_text timed out for %s", url)
        return ""


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
