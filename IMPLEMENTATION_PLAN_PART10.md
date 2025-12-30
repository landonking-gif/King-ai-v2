# King AI v2 - Implementation Plan Part 10
## Sub-Agent: Research (Web/API)

**Target Timeline:** Week 7-8
**Objective:** Implement a comprehensive research agent capable of web scraping, API integrations, and market research.

---

## Overview of All Parts

| Part | Title | Status |
|------|-------|--------|
| 1 | Infrastructure Layer & Core System Hardening | ✅ Complete |
| 2 | Master AI Brain - Core Enhancements | ✅ Complete |
| 3 | Master AI Brain - Context & Memory System | ✅ Complete |
| 4 | Master AI Brain - Planning & ReAct Implementation | ✅ Complete |
| 5 | Evolution Engine - Core Models & Proposal System | ✅ Complete |
| 6 | Evolution Engine - Code Analysis & AST Tools | ✅ Complete |
| 7 | Evolution Engine - Code Patching & Generation | ✅ Complete |
| 8 | Evolution Engine - Git Integration & Rollback | ✅ Complete |
| 9 | Evolution Engine - Sandbox Testing | ✅ Complete |
| **10** | **Sub-Agent: Research (Web/API)** | 🔄 Current |
| 11 | Sub-Agent: Code Generator | ⏳ Pending |
| 12 | Sub-Agent: Content (Blog/SEO) | ⏳ Pending |
| 13 | Sub-Agent: Commerce - Shopify | ⏳ Pending |
| 14 | Sub-Agent: Commerce - Suppliers | ⏳ Pending |
| 15 | Sub-Agent: Finance - Stripe | ⏳ Pending |
| 16 | Sub-Agent: Finance - Plaid/Banking | ⏳ Pending |
| 17 | Sub-Agent: Analytics | ⏳ Pending |
| 18 | Sub-Agent: Legal | ⏳ Pending |
| 19 | Business: Lifecycle Engine | ⏳ Pending |
| 20 | Business: Playbook System | ⏳ Pending |
| 21 | Business: Portfolio Management | ⏳ Pending |
| 22 | Dashboard: React Components | ⏳ Pending |
| 23 | Dashboard: Approval Workflows | ⏳ Pending |
| 24 | Dashboard: WebSocket & Monitoring | ⏳ Pending |

---

## Part 10 Scope

This part focuses on:
1. Web scraping utilities with rate limiting
2. Search engine integration (Google, Bing)
3. Market research data collection
4. Competitor analysis tools
5. API data fetching with caching
6. Research result aggregation

---

## Task 10.1: Create Web Scraper Utility

**File:** `src/utils/web_scraper.py` (CREATE NEW FILE)

```python
"""
Web Scraper Utility - Safe, rate-limited web scraping.
Handles common scraping patterns with retry logic and caching.
"""

import asyncio
import hashlib
import re
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from urllib.parse import urljoin, urlparse
import json

import aiohttp
from bs4 import BeautifulSoup

from src.utils.structured_logging import get_logger
from config.settings import settings

logger = get_logger("web_scraper")


@dataclass
class ScrapedPage:
    """Result of scraping a web page."""
    url: str
    status_code: int
    title: str
    text_content: str
    html_content: str
    links: List[str]
    metadata: Dict[str, str]
    scraped_at: datetime = field(default_factory=datetime.now)
    
    @property
    def word_count(self) -> int:
        """Get word count of text content."""
        return len(self.text_content.split())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "url": self.url,
            "status_code": self.status_code,
            "title": self.title,
            "text_content": self.text_content[:5000],
            "word_count": self.word_count,
            "links_count": len(self.links),
            "metadata": self.metadata,
            "scraped_at": self.scraped_at.isoformat()
        }


class RateLimiter:
    """Token bucket rate limiter."""
    
    def __init__(self, rate: float, burst: int = 1):
        """
        Initialize rate limiter.
        
        Args:
            rate: Requests per second
            burst: Maximum burst size
        """
        self.rate = rate
        self.burst = burst
        self._tokens = burst
        self._last_update = datetime.now()
        self._lock = asyncio.Lock()
    
    async def acquire(self):
        """Acquire a token, waiting if necessary."""
        async with self._lock:
            now = datetime.now()
            elapsed = (now - self._last_update).total_seconds()
            self._tokens = min(self.burst, self._tokens + elapsed * self.rate)
            self._last_update = now
            
            if self._tokens < 1:
                wait_time = (1 - self._tokens) / self.rate
                await asyncio.sleep(wait_time)
                self._tokens = 0
            else:
                self._tokens -= 1


class WebScraper:
    """
    Async web scraper with rate limiting and caching.
    """
    
    DEFAULT_HEADERS = {
        "User-Agent": "Mozilla/5.0 (compatible; KingAI Research Bot/1.0)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }
    
    def __init__(
        self,
        rate_limit: float = 1.0,
        timeout: int = 30,
        max_retries: int = 3
    ):
        """
        Initialize web scraper.
        
        Args:
            rate_limit: Requests per second per domain
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
        """
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_retries = max_retries
        
        # Per-domain rate limiters
        self._rate_limiters: Dict[str, RateLimiter] = {}
        self._default_rate = rate_limit
        
        # Cache
        self._cache: Dict[str, ScrapedPage] = {}
        self._cache_ttl = timedelta(hours=1)
        
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=self.timeout,
                headers=self.DEFAULT_HEADERS
            )
        return self._session
    
    def _get_rate_limiter(self, url: str) -> RateLimiter:
        """Get rate limiter for a domain."""
        domain = urlparse(url).netloc
        if domain not in self._rate_limiters:
            self._rate_limiters[domain] = RateLimiter(self._default_rate, burst=3)
        return self._rate_limiters[domain]
    
    def _get_cache_key(self, url: str) -> str:
        """Generate cache key for URL."""
        return hashlib.md5(url.encode()).hexdigest()
    
    def _check_cache(self, url: str) -> Optional[ScrapedPage]:
        """Check if URL is in cache and not expired."""
        key = self._get_cache_key(url)
        if key in self._cache:
            page = self._cache[key]
            if datetime.now() - page.scraped_at < self._cache_ttl:
                return page
            else:
                del self._cache[key]
        return None
    
    async def scrape(
        self,
        url: str,
        use_cache: bool = True,
        extract_links: bool = True
    ) -> Optional[ScrapedPage]:
        """
        Scrape a web page.
        
        Args:
            url: URL to scrape
            use_cache: Whether to use cached results
            extract_links: Whether to extract links from page
            
        Returns:
            Scraped page data or None on failure
        """
        # Check cache
        if use_cache:
            cached = self._check_cache(url)
            if cached:
                logger.debug(f"Cache hit: {url}")
                return cached
        
        # Rate limiting
        limiter = self._get_rate_limiter(url)
        await limiter.acquire()
        
        # Fetch with retries
        for attempt in range(self.max_retries):
            try:
                session = await self._get_session()
                
                async with session.get(url) as response:
                    html = await response.text()
                    
                    # Parse HTML
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Extract title
                    title_tag = soup.find('title')
                    title = title_tag.get_text().strip() if title_tag else ""
                    
                    # Extract text content
                    for script in soup(['script', 'style', 'nav', 'footer']):
                        script.decompose()
                    text_content = soup.get_text(separator=' ', strip=True)
                    
                    # Extract links
                    links = []
                    if extract_links:
                        for a in soup.find_all('a', href=True):
                            href = a['href']
                            full_url = urljoin(url, href)
                            if full_url.startswith('http'):
                                links.append(full_url)
                    
                    # Extract metadata
                    metadata = self._extract_metadata(soup)
                    
                    page = ScrapedPage(
                        url=url,
                        status_code=response.status,
                        title=title,
                        text_content=text_content,
                        html_content=html,
                        links=list(set(links)),
                        metadata=metadata
                    )
                    
                    # Cache result
                    if use_cache:
                        self._cache[self._get_cache_key(url)] = page
                    
                    logger.info(f"Scraped: {url}", words=page.word_count)
                    return page
                    
            except asyncio.TimeoutError:
                logger.warning(f"Timeout scraping {url}, attempt {attempt + 1}")
            except Exception as e:
                logger.warning(f"Error scraping {url}: {e}, attempt {attempt + 1}")
            
            if attempt < self.max_retries - 1:
                await asyncio.sleep(2 ** attempt)
        
        logger.error(f"Failed to scrape {url} after {self.max_retries} attempts")
        return None
    
    def _extract_metadata(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extract metadata from HTML."""
        metadata = {}
        
        # Meta tags
        for meta in soup.find_all('meta'):
            name = meta.get('name', meta.get('property', ''))
            content = meta.get('content', '')
            if name and content:
                metadata[name] = content[:500]
        
        # Open Graph
        for og in ['og:title', 'og:description', 'og:image']:
            tag = soup.find('meta', property=og)
            if tag and tag.get('content'):
                metadata[og] = tag['content']
        
        return metadata
    
    async def scrape_multiple(
        self,
        urls: List[str],
        max_concurrent: int = 5
    ) -> List[ScrapedPage]:
        """
        Scrape multiple URLs concurrently.
        
        Args:
            urls: List of URLs to scrape
            max_concurrent: Maximum concurrent requests
            
        Returns:
            List of scraped pages (excludes failures)
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def scrape_with_semaphore(url: str) -> Optional[ScrapedPage]:
            async with semaphore:
                return await self.scrape(url)
        
        tasks = [scrape_with_semaphore(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return [r for r in results if isinstance(r, ScrapedPage)]
    
    async def close(self):
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()


class ContentExtractor:
    """Extract structured content from scraped pages."""
    
    @staticmethod
    def extract_article(page: ScrapedPage) -> Dict[str, Any]:
        """Extract article content from a page."""
        soup = BeautifulSoup(page.html_content, 'html.parser')
        
        # Try to find article content
        article = soup.find('article') or soup.find('main') or soup.find('body')
        
        # Extract paragraphs
        paragraphs = []
        for p in article.find_all('p'):
            text = p.get_text().strip()
            if len(text) > 50:
                paragraphs.append(text)
        
        # Extract headings
        headings = []
        for h in article.find_all(['h1', 'h2', 'h3']):
            headings.append({
                "level": int(h.name[1]),
                "text": h.get_text().strip()
            })
        
        return {
            "title": page.title,
            "paragraphs": paragraphs,
            "headings": headings,
            "word_count": page.word_count,
            "url": page.url
        }
    
    @staticmethod
    def extract_product_info(page: ScrapedPage) -> Dict[str, Any]:
        """Extract product information from an e-commerce page."""
        soup = BeautifulSoup(page.html_content, 'html.parser')
        
        product = {
            "name": page.title,
            "url": page.url,
            "price": None,
            "description": None,
            "images": []
        }
        
        # Try to find price patterns
        price_patterns = [
            r'\$[\d,]+\.?\d*',
            r'[\d,]+\.?\d*\s*(?:USD|EUR|GBP)'
        ]
        
        for pattern in price_patterns:
            match = re.search(pattern, page.text_content)
            if match:
                product["price"] = match.group()
                break
        
        # Find product images
        for img in soup.find_all('img'):
            src = img.get('src', '')
            if 'product' in src.lower() or 'item' in src.lower():
                product["images"].append(src)
        
        # Description from meta
        product["description"] = page.metadata.get('description', '')[:500]
        
        return product
```

---

## Task 10.2: Create Search Engine Integration

**File:** `src/utils/search_client.py` (CREATE NEW FILE)

```python
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
        if settings.GOOGLE_API_KEY and settings.GOOGLE_CX:
            self._google = GoogleSearchClient(
                settings.GOOGLE_API_KEY,
                settings.GOOGLE_CX
            )
        
        if settings.BING_API_KEY:
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
```

---

## Task 10.3: Create Research Agent

**File:** `src/agents/research.py` (REPLACE EXISTING FILE)

```python
"""
Research Agent - Comprehensive research capabilities.
Handles web research, market analysis, and competitor tracking.
"""

import asyncio
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from src.agents.base import BaseAgent, AgentCapability, AgentResult
from src.utils.web_scraper import WebScraper, ScrapedPage, ContentExtractor
from src.utils.search_client import UnifiedSearchClient, SearchResponse, SearchResult
from src.utils.ollama_client import OllamaClient
from src.database.vector_store import VectorStore
from src.utils.structured_logging import get_logger

logger = get_logger("research_agent")


class ResearchType(str, Enum):
    """Types of research tasks."""
    MARKET_RESEARCH = "market_research"
    COMPETITOR_ANALYSIS = "competitor_analysis"
    TREND_ANALYSIS = "trend_analysis"
    PRODUCT_RESEARCH = "product_research"
    GENERAL_SEARCH = "general_search"


@dataclass
class ResearchQuery:
    """A research query."""
    query: str
    research_type: ResearchType
    depth: int = 1  # 1=shallow, 2=medium, 3=deep
    max_sources: int = 10
    follow_links: bool = False


@dataclass
class ResearchSource:
    """A source used in research."""
    url: str
    title: str
    relevance_score: float
    content_summary: str
    scraped_at: datetime


@dataclass
class ResearchReport:
    """Complete research report."""
    query: str
    research_type: ResearchType
    summary: str
    key_findings: List[str]
    sources: List[ResearchSource]
    raw_data: Dict[str, Any]
    generated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "research_type": self.research_type.value,
            "summary": self.summary,
            "key_findings": self.key_findings,
            "sources": [
                {"url": s.url, "title": s.title, "relevance": s.relevance_score}
                for s in self.sources
            ],
            "generated_at": self.generated_at.isoformat()
        }


class ResearchAgent(BaseAgent):
    """
    Research agent for comprehensive web and market research.
    """
    
    CAPABILITIES = [
        AgentCapability.RESEARCH,
        AgentCapability.WEB_SCRAPING,
        AgentCapability.DATA_ANALYSIS
    ]
    
    def __init__(
        self,
        llm_client: OllamaClient,
        vector_store: VectorStore
    ):
        """
        Initialize research agent.
        
        Args:
            llm_client: LLM client for analysis
            vector_store: Vector store for caching
        """
        super().__init__("research", llm_client)
        self.vector_store = vector_store
        self.scraper = WebScraper(rate_limit=0.5)
        self.search_client = UnifiedSearchClient()
        self.extractor = ContentExtractor()
    
    async def execute(self, task: Dict[str, Any]) -> AgentResult:
        """Execute a research task."""
        query = ResearchQuery(
            query=task.get("query", ""),
            research_type=ResearchType(task.get("type", "general_search")),
            depth=task.get("depth", 1),
            max_sources=task.get("max_sources", 10)
        )
        
        try:
            report = await self.research(query)
            
            return AgentResult(
                success=True,
                data=report.to_dict(),
                message=f"Research complete: {len(report.sources)} sources analyzed"
            )
        except Exception as e:
            logger.error(f"Research failed: {e}")
            return AgentResult(
                success=False,
                error=str(e),
                message="Research failed"
            )
    
    async def research(self, query: ResearchQuery) -> ResearchReport:
        """
        Perform comprehensive research.
        
        Args:
            query: Research query configuration
            
        Returns:
            Complete research report
        """
        logger.info(
            f"Starting research",
            query=query.query,
            type=query.research_type.value
        )
        
        # Step 1: Search for sources
        search_results = await self._search_sources(query)
        
        # Step 2: Scrape top sources
        scraped_pages = await self._scrape_sources(
            search_results,
            max_pages=query.max_sources
        )
        
        # Step 3: Extract and analyze content
        sources = await self._analyze_sources(scraped_pages, query)
        
        # Step 4: Generate report
        report = await self._generate_report(query, sources)
        
        # Step 5: Store in vector store for future reference
        await self._store_research(report)
        
        logger.info(
            f"Research complete",
            query=query.query,
            sources=len(sources),
            findings=len(report.key_findings)
        )
        
        return report
    
    async def _search_sources(
        self,
        query: ResearchQuery
    ) -> List[SearchResult]:
        """Search for relevant sources."""
        # Construct search queries based on research type
        search_queries = self._build_search_queries(query)
        
        all_results = []
        for sq in search_queries:
            response = await self.search_client.search(
                sq,
                num_results=query.max_sources
            )
            all_results.extend(response.results)
        
        # Deduplicate by URL
        seen_urls = set()
        unique_results = []
        for result in all_results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                unique_results.append(result)
        
        return unique_results[:query.max_sources * 2]
    
    def _build_search_queries(self, query: ResearchQuery) -> List[str]:
        """Build search queries based on research type."""
        base_query = query.query
        
        if query.research_type == ResearchType.MARKET_RESEARCH:
            return [
                f"{base_query} market size",
                f"{base_query} market trends 2024",
                f"{base_query} industry analysis"
            ]
        elif query.research_type == ResearchType.COMPETITOR_ANALYSIS:
            return [
                f"{base_query} competitors",
                f"{base_query} alternatives comparison",
                f"{base_query} vs"
            ]
        elif query.research_type == ResearchType.TREND_ANALYSIS:
            return [
                f"{base_query} trends",
                f"{base_query} future predictions",
                f"{base_query} growth statistics"
            ]
        elif query.research_type == ResearchType.PRODUCT_RESEARCH:
            return [
                f"{base_query} review",
                f"{base_query} features",
                f"{base_query} pricing"
            ]
        else:
            return [base_query]
    
    async def _scrape_sources(
        self,
        results: List[SearchResult],
        max_pages: int
    ) -> List[ScrapedPage]:
        """Scrape top search results."""
        urls = [r.url for r in results[:max_pages]]
        pages = await self.scraper.scrape_multiple(urls, max_concurrent=3)
        return pages
    
    async def _analyze_sources(
        self,
        pages: List[ScrapedPage],
        query: ResearchQuery
    ) -> List[ResearchSource]:
        """Analyze scraped pages for relevance."""
        sources = []
        
        for page in pages:
            # Calculate relevance score
            relevance = self._calculate_relevance(page, query.query)
            
            # Generate summary
            summary = await self._summarize_content(page, query.query)
            
            sources.append(ResearchSource(
                url=page.url,
                title=page.title,
                relevance_score=relevance,
                content_summary=summary,
                scraped_at=page.scraped_at
            ))
        
        # Sort by relevance
        sources.sort(key=lambda s: s.relevance_score, reverse=True)
        
        return sources
    
    def _calculate_relevance(self, page: ScrapedPage, query: str) -> float:
        """Calculate relevance score for a page."""
        query_terms = query.lower().split()
        content_lower = page.text_content.lower()
        title_lower = page.title.lower()
        
        # Term frequency in content
        term_freq = sum(
            content_lower.count(term) for term in query_terms
        ) / max(page.word_count, 1)
        
        # Title match bonus
        title_match = sum(
            1 for term in query_terms if term in title_lower
        ) / len(query_terms)
        
        # Combine scores
        score = (term_freq * 0.6 + title_match * 0.4) * 100
        return min(score, 100)
    
    async def _summarize_content(
        self,
        page: ScrapedPage,
        query: str
    ) -> str:
        """Summarize page content using LLM."""
        prompt = f"""Summarize the following content in relation to: "{query}"

Content from {page.title}:
{page.text_content[:3000]}

Provide a 2-3 sentence summary focusing on the most relevant information."""

        response = await self.llm.generate(prompt)
        return response[:500]
    
    async def _generate_report(
        self,
        query: ResearchQuery,
        sources: List[ResearchSource]
    ) -> ResearchReport:
        """Generate final research report."""
        # Prepare source summaries
        source_text = "\n".join([
            f"Source: {s.title}\nSummary: {s.content_summary}\n"
            for s in sources[:10]
        ])
        
        prompt = f"""Based on the following research sources, generate a comprehensive report.

Research Query: {query.query}
Research Type: {query.research_type.value}

Sources:
{source_text}

Generate:
1. A comprehensive summary (3-5 paragraphs)
2. 5-10 key findings as bullet points

Format as:
SUMMARY:
[Your summary here]

KEY FINDINGS:
- [Finding 1]
- [Finding 2]
..."""

        response = await self.llm.generate(prompt)
        
        # Parse response
        summary, findings = self._parse_report_response(response)
        
        return ResearchReport(
            query=query.query,
            research_type=query.research_type,
            summary=summary,
            key_findings=findings,
            sources=sources,
            raw_data={
                "total_sources_searched": len(sources),
                "depth": query.depth
            }
        )
    
    def _parse_report_response(self, response: str) -> tuple[str, List[str]]:
        """Parse LLM report response."""
        summary = ""
        findings = []
        
        # Split by sections
        if "SUMMARY:" in response and "KEY FINDINGS:" in response:
            parts = response.split("KEY FINDINGS:")
            summary = parts[0].replace("SUMMARY:", "").strip()
            
            findings_text = parts[1] if len(parts) > 1 else ""
            findings = [
                line.strip().lstrip("- •*")
                for line in findings_text.split("\n")
                if line.strip() and line.strip()[0] in "-•*"
            ]
        else:
            summary = response[:1000]
        
        return summary, findings
    
    async def _store_research(self, report: ResearchReport):
        """Store research in vector store for future reference."""
        try:
            await self.vector_store.upsert(
                id=f"research_{hash(report.query)}_{int(datetime.now().timestamp())}",
                text=f"{report.summary}\n\nKey Findings:\n" + "\n".join(report.key_findings),
                metadata={
                    "type": "research_report",
                    "query": report.query,
                    "research_type": report.research_type.value,
                    "source_count": len(report.sources)
                }
            )
        except Exception as e:
            logger.warning(f"Failed to store research: {e}")
    
    async def competitor_analysis(
        self,
        company: str,
        industry: str
    ) -> ResearchReport:
        """
        Perform competitor analysis.
        
        Args:
            company: Company to analyze
            industry: Industry context
            
        Returns:
            Competitor analysis report
        """
        query = ResearchQuery(
            query=f"{company} {industry}",
            research_type=ResearchType.COMPETITOR_ANALYSIS,
            depth=2,
            max_sources=15
        )
        return await self.research(query)
    
    async def market_research(
        self,
        market: str,
        region: str = "global"
    ) -> ResearchReport:
        """
        Perform market research.
        
        Args:
            market: Market to research
            region: Geographic region
            
        Returns:
            Market research report
        """
        query = ResearchQuery(
            query=f"{market} market {region}",
            research_type=ResearchType.MARKET_RESEARCH,
            depth=2,
            max_sources=20
        )
        return await self.research(query)
    
    async def close(self):
        """Clean up resources."""
        await self.scraper.close()
```

---

## Task 10.4: Add Settings for Search APIs

**File:** `config/settings.py` (MODIFY - add these settings)

Add the following to the Settings class:

```python
    # Search API Configuration
    GOOGLE_API_KEY: Optional[str] = None
    GOOGLE_CX: Optional[str] = None  # Custom Search Engine ID
    BING_API_KEY: Optional[str] = None
    
    # Web Scraping Configuration
    SCRAPER_RATE_LIMIT: float = 1.0  # Requests per second
    SCRAPER_TIMEOUT: int = 30
    SCRAPER_MAX_RETRIES: int = 3
    SCRAPER_CACHE_TTL_HOURS: int = 1
```

---

## Task 10.5: Create Research API Routes

**File:** `src/api/routes/research.py` (CREATE NEW FILE)

```python
"""
Research API Routes - REST endpoints for research operations.
"""

from typing import Optional, List
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

from src.agents.research import (
    ResearchAgent,
    ResearchQuery,
    ResearchType,
    ResearchReport
)
from src.utils.ollama_client import OllamaClient
from src.database.vector_store import VectorStore
from src.utils.structured_logging import get_logger

logger = get_logger("research_api")
router = APIRouter(prefix="/research", tags=["research"])


class ResearchRequest(BaseModel):
    """Request for research task."""
    query: str
    research_type: str = "general_search"
    depth: int = 1
    max_sources: int = 10


class CompetitorAnalysisRequest(BaseModel):
    """Request for competitor analysis."""
    company: str
    industry: str


class MarketResearchRequest(BaseModel):
    """Request for market research."""
    market: str
    region: str = "global"


class ResearchResponse(BaseModel):
    """Research response."""
    query: str
    research_type: str
    summary: str
    key_findings: List[str]
    source_count: int


# Global agent instance
_research_agent: Optional[ResearchAgent] = None


async def get_research_agent() -> ResearchAgent:
    """Get or create research agent."""
    global _research_agent
    if _research_agent is None:
        llm = OllamaClient()
        vector_store = VectorStore()
        _research_agent = ResearchAgent(llm, vector_store)
    return _research_agent


@router.post("/search", response_model=ResearchResponse)
async def perform_research(request: ResearchRequest):
    """Perform research on a topic."""
    try:
        agent = await get_research_agent()
        
        query = ResearchQuery(
            query=request.query,
            research_type=ResearchType(request.research_type),
            depth=request.depth,
            max_sources=request.max_sources
        )
        
        report = await agent.research(query)
        
        return ResearchResponse(
            query=report.query,
            research_type=report.research_type.value,
            summary=report.summary,
            key_findings=report.key_findings,
            source_count=len(report.sources)
        )
        
    except Exception as e:
        logger.error(f"Research failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/competitors", response_model=ResearchResponse)
async def analyze_competitors(request: CompetitorAnalysisRequest):
    """Analyze competitors for a company."""
    try:
        agent = await get_research_agent()
        report = await agent.competitor_analysis(
            request.company,
            request.industry
        )
        
        return ResearchResponse(
            query=report.query,
            research_type=report.research_type.value,
            summary=report.summary,
            key_findings=report.key_findings,
            source_count=len(report.sources)
        )
        
    except Exception as e:
        logger.error(f"Competitor analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/market", response_model=ResearchResponse)
async def market_research(request: MarketResearchRequest):
    """Perform market research."""
    try:
        agent = await get_research_agent()
        report = await agent.market_research(
            request.market,
            request.region
        )
        
        return ResearchResponse(
            query=report.query,
            research_type=report.research_type.value,
            summary=report.summary,
            key_findings=report.key_findings,
            source_count=len(report.sources)
        )
        
    except Exception as e:
        logger.error(f"Market research failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

---

## Testing Requirements

**File:** `tests/test_research.py` (CREATE NEW FILE)

```python
"""Tests for research agent."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.research import (
    ResearchAgent,
    ResearchQuery,
    ResearchType,
    ResearchReport
)
from src.utils.web_scraper import WebScraper, ScrapedPage
from src.utils.search_client import UnifiedSearchClient, SearchResult, SearchResponse


class TestWebScraper:
    """Tests for WebScraper."""
    
    @pytest.fixture
    def scraper(self):
        return WebScraper(rate_limit=10)  # Fast for tests
    
    @pytest.mark.asyncio
    async def test_rate_limiter(self, scraper):
        """Test rate limiting works."""
        limiter = scraper._get_rate_limiter("https://example.com")
        
        # Should not block for first request
        await limiter.acquire()
        assert True
    
    def test_cache_key_generation(self, scraper):
        """Test cache key is consistent."""
        url = "https://example.com/page"
        key1 = scraper._get_cache_key(url)
        key2 = scraper._get_cache_key(url)
        
        assert key1 == key2
        assert len(key1) == 32  # MD5 hex length


class TestSearchClient:
    """Tests for search client."""
    
    def test_unified_client_creation(self):
        """Test unified client can be created."""
        client = UnifiedSearchClient()
        assert client is not None


class TestResearchAgent:
    """Tests for ResearchAgent."""
    
    @pytest.fixture
    def mock_llm(self):
        llm = AsyncMock()
        llm.generate = AsyncMock(return_value="""
SUMMARY:
This is a test summary.

KEY FINDINGS:
- Finding 1
- Finding 2
""")
        return llm
    
    @pytest.fixture
    def mock_vector_store(self):
        store = AsyncMock()
        store.upsert = AsyncMock()
        return store
    
    @pytest.fixture
    def agent(self, mock_llm, mock_vector_store):
        return ResearchAgent(mock_llm, mock_vector_store)
    
    def test_build_search_queries_market(self, agent):
        """Test market research query building."""
        query = ResearchQuery(
            query="AI software",
            research_type=ResearchType.MARKET_RESEARCH
        )
        
        queries = agent._build_search_queries(query)
        
        assert len(queries) == 3
        assert any("market size" in q for q in queries)
    
    def test_build_search_queries_competitor(self, agent):
        """Test competitor analysis query building."""
        query = ResearchQuery(
            query="OpenAI",
            research_type=ResearchType.COMPETITOR_ANALYSIS
        )
        
        queries = agent._build_search_queries(query)
        
        assert len(queries) == 3
        assert any("competitors" in q for q in queries)
    
    def test_calculate_relevance(self, agent):
        """Test relevance calculation."""
        page = ScrapedPage(
            url="https://example.com",
            status_code=200,
            title="AI Market Analysis",
            text_content="The AI market is growing rapidly with new developments.",
            html_content="<html></html>",
            links=[],
            metadata={}
        )
        
        relevance = agent._calculate_relevance(page, "AI market")
        
        assert relevance > 0
        assert relevance <= 100
    
    def test_parse_report_response(self, agent):
        """Test report parsing."""
        response = """
SUMMARY:
This is the summary.

KEY FINDINGS:
- First finding
- Second finding
- Third finding
"""
        
        summary, findings = agent._parse_report_response(response)
        
        assert "summary" in summary.lower()
        assert len(findings) == 3
```

---

## Acceptance Criteria

- [ ] `src/utils/web_scraper.py` - Rate-limited web scraping
- [ ] `src/utils/search_client.py` - Google/Bing search integration
- [ ] `src/agents/research.py` - Complete research agent
- [ ] `src/api/routes/research.py` - REST API endpoints
- [ ] `tests/test_research.py` - All tests passing
- [ ] Rate limiting working correctly
- [ ] Search fallback working
- [ ] Research reports generated properly

---

## File Summary

| Action | File Path |
|--------|-----------|
| CREATE | `src/utils/web_scraper.py` |
| CREATE | `src/utils/search_client.py` |
| REPLACE | `src/agents/research.py` |
| MODIFY | `config/settings.py` |
| CREATE | `src/api/routes/research.py` |
| CREATE | `tests/test_research.py` |

---

*End of Part 10*
