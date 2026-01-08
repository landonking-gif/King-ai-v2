"""
Web Tools - Real-time data access for King AI.
Provides web search, stock prices, news, and date/time utilities.

No API keys required - uses free APIs and DuckDuckGo.
"""

import asyncio
import aiohttp
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json
import re

from src.utils.structured_logging import get_logger

logger = get_logger("web_tools")


@dataclass
class SearchResult:
    """A single search result."""
    title: str
    url: str
    snippet: str
    
    def to_dict(self) -> Dict:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet
        }


@dataclass 
class StockQuote:
    """Stock price information."""
    symbol: str
    price: float
    change: float
    change_percent: float
    volume: int
    timestamp: datetime
    
    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "price": self.price,
            "change": self.change,
            "change_percent": self.change_percent,
            "volume": self.volume,
            "timestamp": self.timestamp.isoformat()
        }
    
    def __str__(self) -> str:
        direction = "▲" if self.change >= 0 else "▼"
        return f"{self.symbol}: ${self.price:.2f} {direction} {abs(self.change_percent):.2f}%"


class WebTools:
    """
    Unified web tools for real-time data access.
    Uses free APIs - no API keys required.
    """
    
    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create an aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                headers={"User-Agent": "King-AI/2.0"}
            )
        return self._session
    
    async def close(self):
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    # ===== DATE/TIME =====
    
    def get_current_datetime(self) -> Dict[str, str]:
        """
        Get current date and time in multiple formats.
        
        Returns:
            Dict with date, time, datetime, timezone, and day_of_week
        """
        now = datetime.now()
        utc_now = datetime.now(timezone.utc)
        
        return {
            "date": now.strftime("%B %d, %Y"),  # January 6, 2026
            "time": now.strftime("%I:%M %p"),  # 9:28 PM
            "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
            "day_of_week": now.strftime("%A"),  # Monday
            "timezone": "Local",
            "utc_datetime": utc_now.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "iso": now.isoformat(),
            "timestamp": int(now.timestamp())
        }
    
    def get_datetime_context(self) -> str:
        """Get formatted datetime context for LLM prompts."""
        dt = self.get_current_datetime()
        return f"Current Date: {dt['day_of_week']}, {dt['date']}\nCurrent Time: {dt['time']} ({dt['timezone']})"
    
    # ===== WEB SEARCH (DuckDuckGo - Free) =====
    
    async def web_search(
        self,
        query: str,
        max_results: int = 5
    ) -> List[SearchResult]:
        """
        Search the web using DuckDuckGo (API + HTML Fallback).
        Free, no API key required.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            List of search results
        """
        results = []
        try:
            session = await self._get_session()
            
            # 1. Try Instant Answers API (Fast, Clean)
            try:
                url = "https://api.duckduckgo.com/"
                params = {
                    "q": query,
                    "format": "json",
                    "no_html": 1,
                    "skip_disambig": 1
                }
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        content_type = response.headers.get("Content-Type", "")
                        if "json" in content_type or "javascript" in content_type:
                            data = await response.json(content_type=None) # Handle x-javascript
                            
                            # Abstract (main answer)
                            if data.get("Abstract"):
                                results.append(SearchResult(
                                    title=data.get("Heading", "Summary"),
                                    url=data.get("AbstractURL", ""),
                                    snippet=data.get("Abstract", "")
                                ))
                            
                            # Related topics
                            for topic in data.get("RelatedTopics", [])[:max_results]:
                                if isinstance(topic, dict) and "Text" in topic:
                                    results.append(SearchResult(
                                        title=topic.get("Text", "")[:50],
                                        url=topic.get("FirstURL", ""),
                                        snippet=topic.get("Text", "")
                                    ))
            except Exception as e:
                logger.warning(f"DDG API failed: {e}")

            # 2. If no results, try HTML Search (Scraping - Slower but comprehensive)
            if not results:
                try:
                    url = "https://html.duckduckgo.com/html/"
                    data = {"q": query}
                    headers = {
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                    }
                    
                    async with session.post(url, data=data, headers=headers) as response:
                        if response.status == 200:
                            html = await response.text()
                            
                            # Regex to extract results (Simple parser)
                            # Looking for <a class="result__a" href="...">Title</a> and <a class="result__snippet" ...>Snippet</a>
                            import re
                            
                            # Find result blocks
                            link_pattern = r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>'
                            snippet_pattern = r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>'
                            
                            links = re.findall(link_pattern, html)
                            snippets = re.findall(snippet_pattern, html)
                            
                            for i in range(min(len(links), len(snippets), max_results)):
                                link_url, link_title = links[i]
                                snippet_text = snippets[i]
                                
                                # Clean HTML tags from title/snippet
                                link_title = re.sub(r'<[^>]+>', '', link_title).strip()
                                snippet_text = re.sub(r'<[^>]+>', '', snippet_text).strip()
                                
                                results.append(SearchResult(
                                    title=link_title,
                                    url=link_url,
                                    snippet=snippet_text
                                ))
                except Exception as e:
                    logger.warning(f"DDG HTML search failed: {e}")

            logger.info(f"Web search for '{query}' returned {len(results)} results")
            return results[:max_results]
            
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return []
    
    async def search_formatted(self, query: str, max_results: int = 5) -> str:
        """
        Search and return formatted results for LLM context.
        
        Returns:
            Formatted string of search results
        """
        results = await self.web_search(query, max_results)
        
        if not results:
            return f"No web search results found for: {query}"
        
        formatted = f"Web Search Results for '{query}':\n"
        for i, r in enumerate(results, 1):
            formatted += f"\n{i}. {r.title}\n"
            formatted += f"   {r.snippet}\n"
            if r.url:
                formatted += f"   Source: {r.url}\n"
        
        return formatted
    
    # ===== STOCK PRICES (Yahoo Finance - Free) =====
    
    async def get_stock_price(self, symbol: str) -> Optional[StockQuote]:
        """
        Get current stock price using Yahoo Finance.
        Free, no API key required.
        
        Args:
            symbol: Stock ticker symbol (e.g., "AAPL", "GOOGL")
            
        Returns:
            StockQuote or None if failed
        """
        try:
            session = await self._get_session()
            
            # Yahoo Finance API endpoint
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            params = {
                "interval": "1d",
                "range": "1d"
            }
            
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    logger.warning(f"Stock price fetch failed for {symbol}: {response.status}")
                    return None
                
                data = await response.json()
            
            result = data.get("chart", {}).get("result", [])
            if not result:
                return None
            
            meta = result[0].get("meta", {})
            
            current_price = meta.get("regularMarketPrice", 0)
            previous_close = meta.get("previousClose", current_price)
            change = current_price - previous_close
            change_percent = (change / previous_close * 100) if previous_close else 0
            
            return StockQuote(
                symbol=symbol.upper(),
                price=current_price,
                change=change,
                change_percent=change_percent,
                volume=meta.get("regularMarketVolume", 0),
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Stock price error for {symbol}: {e}")
            return None
    
    async def get_multiple_stocks(self, symbols: List[str]) -> Dict[str, Optional[StockQuote]]:
        """Get multiple stock prices concurrently."""
        tasks = [self.get_stock_price(s) for s in symbols]
        results = await asyncio.gather(*tasks)
        return dict(zip(symbols, results))
    
    async def get_market_summary(self) -> str:
        """
        Get a summary of major market indices.
        
        Returns:
            Formatted market summary string
        """
        indices = ["^DJI", "^GSPC", "^IXIC"]  # Dow, S&P 500, NASDAQ
        names = {"^DJI": "Dow Jones", "^GSPC": "S&P 500", "^IXIC": "NASDAQ"}
        
        quotes = await self.get_multiple_stocks(indices)
        
        summary = "Market Summary:\n"
        for symbol, quote in quotes.items():
            name = names.get(symbol, symbol)
            if quote:
                direction = "▲" if quote.change >= 0 else "▼"
                summary += f"  {name}: {quote.price:,.2f} {direction} {abs(quote.change_percent):.2f}%\n"
            else:
                summary += f"  {name}: Data unavailable\n"
        
        return summary
    
    # ===== NEWS HEADLINES =====
    
    async def get_news_headlines(self, topic: str = "business", max_headlines: int = 5) -> List[Dict]:
        """
        Get news headlines using web search.
        
        Args:
            topic: News topic/category
            max_headlines: Maximum number of headlines
            
        Returns:
            List of headline dictionaries
        """
        query = f"{topic} news today"
        results = await self.web_search(query, max_headlines)
        
        return [
            {
                "title": r.title,
                "summary": r.snippet,
                "url": r.url
            }
            for r in results
        ]
    
    async def get_news_formatted(self, topic: str = "business") -> str:
        """Get formatted news headlines for LLM context."""
        headlines = await self.get_news_headlines(topic)
        
        if not headlines:
            return f"No recent news found for: {topic}"
        
        result = f"Recent {topic.title()} News:\n"
        for i, h in enumerate(headlines, 1):
            result += f"\n{i}. {h['title']}\n"
            result += f"   {h['summary']}\n"
        
        return result
    
    # ===== UTILITY METHODS =====
    
    def detect_realtime_query(self, query: str) -> Dict[str, bool]:
        """
        Detect if a query requires real-time data.
        
        Args:
            query: User query
            
        Returns:
            Dict indicating which types of real-time data are needed
        """
        query_lower = query.lower()
        
        return {
            "needs_datetime": any(word in query_lower for word in [
                "time", "date", "today", "now", "current day", "what day"
            ]),
            "needs_web_search": any(word in query_lower for word in [
                "news", "latest", "recent", "search", "find out", "look up",
                "what is happening", "trending"
            ]),
            "needs_stock_data": any(word in query_lower for word in [
                "stock", "price", "market", "nasdaq", "dow", "s&p",
                "trading", "shares", "ticker"
            ]),
            "needs_market_trends": any(word in query_lower for word in [
                "market trend", "market analysis", "financial news",
                "economy", "recession", "inflation"
            ])
        }
    
    async def get_realtime_context(self, query: str) -> str:
        """
        Build real-time context based on query needs.
        
        Args:
            query: User query
            
        Returns:
            Formatted context string with relevant real-time data
        """
        needs = self.detect_realtime_query(query)
        context_parts = []
        
        # Always include datetime
        context_parts.append(self.get_datetime_context())
        
        # Add stock/market data if needed
        if needs["needs_stock_data"] or needs["needs_market_trends"]:
            try:
                market = await self.get_market_summary()
                context_parts.append(market)
            except Exception as e:
                logger.warning(f"Failed to get market data: {e}")
        
        # Add web search if needed
        if needs["needs_web_search"] or needs["needs_market_trends"]:
            try:
                search_results = await self.search_formatted(query)
                context_parts.append(search_results)
            except Exception as e:
                logger.warning(f"Failed to get web search results: {e}")
        
        return "\n\n".join(context_parts)


# Singleton instance
_web_tools: Optional[WebTools] = None

def get_web_tools() -> WebTools:
    """Get the singleton WebTools instance."""
    global _web_tools
    if _web_tools is None:
        _web_tools = WebTools()
    return _web_tools


# Simple math evaluation for basic queries
def evaluate_simple_math(query: str) -> Optional[str]:
    """
    Evaluate simple math expressions in a query.
    
    Args:
        query: User query that might contain math
        
    Returns:
        Result string or None if not a math query
    """
    # Match patterns like "what is 5 times 3" or "5 * 3"
    patterns = [
        (r"what is (\d+(?:\.\d+)?)\s*(?:times|x|\*)\s*(\d+(?:\.\d+)?)", lambda a, b: a * b),
        (r"what is (\d+(?:\.\d+)?)\s*(?:plus|\+)\s*(\d+(?:\.\d+)?)", lambda a, b: a + b),
        (r"what is (\d+(?:\.\d+)?)\s*(?:minus|-)\s*(\d+(?:\.\d+)?)", lambda a, b: a - b),
        (r"what is (\d+(?:\.\d+)?)\s*(?:divided by|/)\s*(\d+(?:\.\d+)?)", lambda a, b: a / b if b != 0 else None),
        (r"(\d+(?:\.\d+)?)\s*(?:\*|x)\s*(\d+(?:\.\d+)?)", lambda a, b: a * b),
        (r"(\d+(?:\.\d+)?)\s*\+\s*(\d+(?:\.\d+)?)", lambda a, b: a + b),
        (r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)", lambda a, b: a - b),
        (r"(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)", lambda a, b: a / b if b != 0 else None),
    ]
    
    query_lower = query.lower()
    
    for pattern, operation in patterns:
        match = re.search(pattern, query_lower)
        if match:
            try:
                a = float(match.group(1))
                b = float(match.group(2))
                result = operation(a, b)
                if result is not None:
                    # Format nicely
                    if result == int(result):
                        return str(int(result))
                    return f"{result:.2f}"
            except (ValueError, ZeroDivisionError):
                pass
    
    return None


# ===== EXTENDED CAPABILITIES =====

async def get_crypto_price(symbol: str = "bitcoin") -> Optional[Dict]:
    """
    Get cryptocurrency price from CoinGecko (free, no API key).
    
    Args:
        symbol: Crypto symbol (bitcoin, ethereum, etc.)
        
    Returns:
        Dict with price info or None
    """
    web_tools = get_web_tools()
    try:
        session = await web_tools._get_session()
        url = f"https://api.coingecko.com/api/v3/simple/price"
        params = {
            "ids": symbol.lower(),
            "vs_currencies": "usd",
            "include_24hr_change": "true"
        }
        
        async with session.get(url, params=params) as response:
            if response.status != 200:
                return None
            data = await response.json()
            
        if symbol.lower() in data:
            coin_data = data[symbol.lower()]
            return {
                "symbol": symbol.upper(),
                "price_usd": coin_data.get("usd", 0),
                "change_24h": coin_data.get("usd_24h_change", 0)
            }
        return None
    except Exception as e:
        logger.error(f"Crypto price error: {e}")
        return None


async def get_weather(city: str = "New York") -> Optional[Dict]:
    """
    Get weather using wttr.in (free, no API key).
    
    Args:
        city: City name
        
    Returns:
        Dict with weather info or None
    """
    web_tools = get_web_tools()
    try:
        session = await web_tools._get_session()
        url = f"https://wttr.in/{city}?format=j1"
        
        async with session.get(url) as response:
            if response.status != 200:
                return None
            data = await response.json()
        
        current = data.get("current_condition", [{}])[0]
        return {
            "city": city,
            "temperature_f": current.get("temp_F", "N/A"),
            "temperature_c": current.get("temp_C", "N/A"),
            "condition": current.get("weatherDesc", [{}])[0].get("value", "Unknown"),
            "humidity": current.get("humidity", "N/A"),
            "wind_mph": current.get("windspeedMiles", "N/A")
        }
    except Exception as e:
        logger.error(f"Weather error: {e}")
        return None


async def get_wikipedia_summary(topic: str) -> Optional[str]:
    """
    Get Wikipedia summary for a topic.
    
    Args:
        topic: Topic to look up
        
    Returns:
        Summary string or None
    """
    web_tools = get_web_tools()
    try:
        session = await web_tools._get_session()
        url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + topic.replace(" ", "_")
        
        async with session.get(url) as response:
            if response.status != 200:
                return None
            data = await response.json()
        
        return data.get("extract", None)
    except Exception as e:
        logger.error(f"Wikipedia error: {e}")
        return None


def convert_units(value: float, from_unit: str, to_unit: str) -> Optional[float]:
    """
    Convert between common units.
    
    Args:
        value: Numeric value to convert
        from_unit: Source unit
        to_unit: Target unit
        
    Returns:
        Converted value or None if not supported
    """
    conversions = {
        # Length
        ("miles", "km"): lambda x: x * 1.60934,
        ("km", "miles"): lambda x: x / 1.60934,
        ("feet", "meters"): lambda x: x * 0.3048,
        ("meters", "feet"): lambda x: x / 0.3048,
        ("inches", "cm"): lambda x: x * 2.54,
        ("cm", "inches"): lambda x: x / 2.54,
        
        # Weight
        ("pounds", "kg"): lambda x: x * 0.453592,
        ("kg", "pounds"): lambda x: x / 0.453592,
        ("ounces", "grams"): lambda x: x * 28.3495,
        ("grams", "ounces"): lambda x: x / 28.3495,
        
        # Temperature
        ("fahrenheit", "celsius"): lambda x: (x - 32) * 5/9,
        ("celsius", "fahrenheit"): lambda x: x * 9/5 + 32,
        ("f", "c"): lambda x: (x - 32) * 5/9,
        ("c", "f"): lambda x: x * 9/5 + 32,
        
        # Volume
        ("gallons", "liters"): lambda x: x * 3.78541,
        ("liters", "gallons"): lambda x: x / 3.78541,
    }
    
    key = (from_unit.lower(), to_unit.lower())
    if key in conversions:
        return conversions[key](value)
    return None


async def get_exchange_rate(from_currency: str = "USD", to_currency: str = "EUR") -> Optional[float]:
    """
    Get currency exchange rate using free API.
    
    Args:
        from_currency: Source currency code
        to_currency: Target currency code
        
    Returns:
        Exchange rate or None
    """
    web_tools = get_web_tools()
    try:
        session = await web_tools._get_session()
        # Using exchangerate.host (free, no API key)
        url = f"https://api.exchangerate.host/latest?base={from_currency}&symbols={to_currency}"
        
        async with session.get(url) as response:
            if response.status != 200:
                return None
            data = await response.json()
        
        rates = data.get("rates", {})
        return rates.get(to_currency)
    except Exception as e:
        logger.error(f"Exchange rate error: {e}")
        return None


def parse_unit_conversion_query(query: str) -> Optional[Dict]:
    """
    Parse a unit conversion query.
    
    Args:
        query: Natural language query like "convert 100 miles to km"
        
    Returns:
        Dict with value, from_unit, to_unit or None
    """
    patterns = [
        r"convert\s+(\d+(?:\.\d+)?)\s+(\w+)\s+to\s+(\w+)",
        r"(\d+(?:\.\d+)?)\s+(\w+)\s+(?:to|in)\s+(\w+)",
        r"how many (\w+) (?:is|are|in)\s+(\d+(?:\.\d+)?)\s+(\w+)",
    ]
    
    query_lower = query.lower()
    
    for pattern in patterns:
        match = re.search(pattern, query_lower)
        if match:
            groups = match.groups()
            if pattern.startswith("how many"):
                return {
                    "value": float(groups[1]),
                    "from_unit": groups[2],
                    "to_unit": groups[0]
                }
            else:
                return {
                    "value": float(groups[0]),
                    "from_unit": groups[1],
                    "to_unit": groups[2]
                }
    return None

