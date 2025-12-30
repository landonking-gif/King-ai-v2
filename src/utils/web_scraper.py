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
