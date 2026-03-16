from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel

from src import constants


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
