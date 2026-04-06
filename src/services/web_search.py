from __future__ import annotations

from abc import ABC, abstractmethod

import httpx
import trafilatura
from pydantic import BaseModel

from src import constants
from src.config import LibrarianConfig


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

    @abstractmethod
    async def fetch_page_content(self, url: str) -> str:
        """Fetch a URL and extract the main body text, stripping boilerplate."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if the search backend is configured and reachable."""
        ...


class BraveWebSearchClient(WebSearchClient):
    """Concrete implementation of WebSearchClient using Brave Search API."""

    def __init__(self, config: LibrarianConfig) -> None:
        self._config = config
        self._api_key = config.web_search.api_key

    async def search(
        self, query: str, max_results: int = constants.DEFAULT_MAX_RESULTS
    ) -> list[WebSearchResult]:
        if not self._api_key:
            print("Brave Search API key not set. Skipping search.")
            return []

        url = "https://api.search.brave.com/res/v1/web/search"
        params: dict[str, str | int] = {"q": query, "count": max_results}

        try:
            headers = {"X-Subscription-Token": self._api_key} if self._api_key else {}
            async with httpx.AsyncClient(headers=headers) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                data = response.json()

            results = []
            web_results = data.get("web", {}).get("results", [])
            for r in web_results:
                results.append(
                    WebSearchResult(
                        title=r.get("title", ""),
                        url=r.get("url", ""),
                        snippet=r.get("description", ""),
                    )
                )
            return results
        except httpx.HTTPStatusError as e:
            print(f"Brave Search API error: {e}")
            return []
        except Exception as e:
            print(f"Error during search: {e}")
            return []

    async def fetch_page_content(self, url: str) -> str:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                response.raise_for_status()
                html = response.text

            # Use trafilatura to extract text
            text = trafilatura.extract(html)
            return text or ""
        except Exception as e:
            print(f"Error fetching page content for {url}: {e}")
            return ""

    def is_available(self) -> bool:
        return bool(self._api_key)
