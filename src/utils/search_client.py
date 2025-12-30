"""
Search Engine Client - Integration with search APIs.
Supports Google Custom Search, Bing, and fallback scraping.
"""

import asyncio
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import aiohttp

from src.utils.structured_logging import get_logger
from config.settings import settings

logger = get_logger("search_client")


class SearchEngine(str, Enum):
    """Supported search engines."""
    GOOGLE = "google"
    BING = "bing"
    DUCKDUCKGO = "duckduckgo"


@dataclass
class SearchResult:
    """A single search result."""
    title: str
    url: str
    snippet: str
    position: int
    engine: SearchEngine
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "position": self.position,
            "engine": self.engine.value
        }


@dataclass
class SearchResponse:
    """Complete search response."""
    query: str
    results: List[SearchResult]
    total_results: Optional[int] = None
    search_time_ms: float = 0
    engine: SearchEngine = SearchEngine.GOOGLE
    
    @property
    def result_count(self) -> int:
        return len(self.results)


class GoogleSearchClient:
    """Google Custom Search API client."""
    
    BASE_URL = "https://www.googleapis.com/customsearch/v1"
    
    def __init__(self, api_key: str, cx: str):
        """
        Initialize Google search client.
        
        Args:
            api_key: Google API key
            cx: Custom Search Engine ID
        """
        self.api_key = api_key
        self.cx = cx
    
    async def search(
        self,
        query: str,
        num_results: int = 10,
        start: int = 1
    ) -> SearchResponse:
        """
        Perform Google search.
        
        Args:
            query: Search query
            num_results: Number of results (max 10 per request)
            start: Starting position
            
        Returns:
            Search response
        """
        params = {
            "key": self.api_key,
            "cx": self.cx,
            "q": query,
            "num": min(num_results, 10),
            "start": start
        }
        
        start_time = datetime.now()
        
        async with aiohttp.ClientSession() as session:
            async with session.get(self.BASE_URL, params=params) as response:
                data = await response.json()
        
        search_time = (datetime.now() - start_time).total_seconds() * 1000
        
        results = []
        for i, item in enumerate(data.get("items", [])):
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("link", ""),
                snippet=item.get("snippet", ""),
                position=start + i,
                engine=SearchEngine.GOOGLE
            ))
        
        return SearchResponse(
            query=query,
            results=results,
            total_results=int(data.get("searchInformation", {}).get("totalResults", 0)),
            search_time_ms=search_time,
            engine=SearchEngine.GOOGLE
        )


class BingSearchClient:
    """Bing Search API client."""
    
    BASE_URL = "https://api.bing.microsoft.com/v7.0/search"
    
    def __init__(self, api_key: str):
        """
        Initialize Bing search client.
        
        Args:
            api_key: Bing Search API key
        """
        self.api_key = api_key
    
    async def search(
        self,
        query: str,
        num_results: int = 10,
        offset: int = 0
    ) -> SearchResponse:
        """
        Perform Bing search.
        
        Args:
            query: Search query
            num_results: Number of results
            offset: Result offset
            
        Returns:
            Search response
        """
        headers = {"Ocp-Apim-Subscription-Key": self.api_key}
        params = {
            "q": query,
            "count": num_results,
            "offset": offset
        }
        
        start_time = datetime.now()
        
        async with aiohttp.ClientSession() as session:
            async with session.get(
                self.BASE_URL,
                headers=headers,
                params=params
            ) as response:
                data = await response.json()
        
        search_time = (datetime.now() - start_time).total_seconds() * 1000
        
        results = []
        for i, item in enumerate(data.get("webPages", {}).get("value", [])):
            results.append(SearchResult(
                title=item.get("name", ""),
                url=item.get("url", ""),
                snippet=item.get("snippet", ""),
                position=offset + i + 1,
                engine=SearchEngine.BING
            ))
        
        return SearchResponse(
            query=query,
            results=results,
            total_results=data.get("webPages", {}).get("totalEstimatedMatches"),
            search_time_ms=search_time,
            engine=SearchEngine.BING
        )


class UnifiedSearchClient:
    """
    Unified search client with fallback support.
    Uses available search engines with automatic fallback.
    """
    
    def __init__(self):
        """Initialize unified search client."""
        self._google: Optional[GoogleSearchClient] = None
        self._bing: Optional[BingSearchClient] = None
        
        # Initialize available engines
        if hasattr(settings, 'GOOGLE_API_KEY') and settings.GOOGLE_API_KEY and \
           hasattr(settings, 'GOOGLE_CX') and settings.GOOGLE_CX:
            self._google = GoogleSearchClient(
                settings.GOOGLE_API_KEY,
                settings.GOOGLE_CX
            )
        
        if hasattr(settings, 'BING_API_KEY') and settings.BING_API_KEY:
            self._bing = BingSearchClient(settings.BING_API_KEY)
    
    async def search(
        self,
        query: str,
        num_results: int = 10,
        engine: SearchEngine = None
    ) -> SearchResponse:
        """
        Perform search with automatic engine selection.
        
        Args:
            query: Search query
            num_results: Number of results
            engine: Preferred engine (optional)
            
        Returns:
            Search response
        """
        # Select engine
        if engine == SearchEngine.GOOGLE and self._google:
            return await self._google.search(query, num_results)
        elif engine == SearchEngine.BING and self._bing:
            return await self._bing.search(query, num_results)
        
        # Auto-select with fallback
        if self._google:
            try:
                return await self._google.search(query, num_results)
            except Exception as e:
                logger.warning(f"Google search failed: {e}")
        
        if self._bing:
            try:
                return await self._bing.search(query, num_results)
            except Exception as e:
                logger.warning(f"Bing search failed: {e}")
        
        logger.error("No search engines available")
        return SearchResponse(query=query, results=[])
    
    async def multi_search(
        self,
        queries: List[str],
        num_results: int = 10
    ) -> Dict[str, SearchResponse]:
        """
        Search multiple queries.
        
        Returns:
            Dict mapping query to results
        """
        tasks = [self.search(q, num_results) for q in queries]
        results = await asyncio.gather(*tasks)
        return dict(zip(queries, results))
