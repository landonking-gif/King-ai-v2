"""
Web Tools for Agent Framework

Tools for web search and information retrieval.
"""

from typing import List, Dict, Any, Optional
from agent_framework.tools.base_tool import BaseTool
import json


class WebSearchTool(BaseTool):
    """Tool for searching the web"""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(
            name="web_search",
            description="Search the web for current information. Input should be a search query string."
        )
        self.api_key = api_key
    
    def run(self, query: str, num_results: int = 5) -> str:
        """
        Search the web for information.
        
        Args:
            query: Search query
            num_results: Number of results to return
            
        Returns:
            Formatted search results
        """
        # Placeholder implementation
        # In production, integrate with Google Search API, Bing API, or SerpAPI
        
        results = [
            {
                "title": f"Result {i+1} for: {query}",
                "url": f"https://example.com/result{i+1}",
                "snippet": f"This is a placeholder result for the query '{query}'. "
                          f"In production, this would contain actual search results."
            }
            for i in range(num_results)
        ]
        
        formatted = []
        for i, r in enumerate(results, 1):
            formatted.append(f"{i}. {r['title']}\n   URL: {r['url']}\n   {r['snippet']}")
        
        return "\n\n".join(formatted)


class WikipediaTool(BaseTool):
    """Tool for searching Wikipedia"""
    
    def __init__(self):
        super().__init__(
            name="wikipedia",
            description="Search Wikipedia for factual information. Input should be a topic or search term."
        )
    
    def run(self, query: str, sentences: int = 5) -> str:
        """
        Search Wikipedia for information.
        
        Args:
            query: Topic to search for
            sentences: Number of sentences to return
            
        Returns:
            Wikipedia summary
        """
        try:
            import wikipedia
            
            # Search for the topic
            search_results = wikipedia.search(query, results=3)
            
            if not search_results:
                return f"No Wikipedia articles found for '{query}'"
            
            # Get the summary of the first result
            try:
                summary = wikipedia.summary(search_results[0], sentences=sentences)
                return f"**{search_results[0]}**\n\n{summary}"
            except wikipedia.DisambiguationError as e:
                # Handle disambiguation pages
                return f"Multiple results found. Options: {', '.join(e.options[:5])}"
            except wikipedia.PageError:
                return f"No Wikipedia page found for '{query}'"
        
        except ImportError:
            return (
                f"Wikipedia search for '{query}': "
                "[Placeholder - install wikipedia package: pip install wikipedia]"
            )


class URLFetchTool(BaseTool):
    """Tool for fetching content from URLs"""
    
    def __init__(self):
        super().__init__(
            name="url_fetch",
            description="Fetch and extract text content from a URL. Input should be a valid URL."
        )
    
    def run(self, url: str, max_length: int = 5000) -> str:
        """
        Fetch content from a URL.
        
        Args:
            url: URL to fetch
            max_length: Maximum content length to return
            
        Returns:
            Extracted text content
        """
        try:
            import requests
            from bs4 import BeautifulSoup
            
            # Fetch the page
            headers = {
                'User-Agent': 'Mozilla/5.0 (compatible; AgentBot/1.0)'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # Get text
            text = soup.get_text(separator='\n', strip=True)
            
            # Truncate if needed
            if len(text) > max_length:
                text = text[:max_length] + "...[truncated]"
            
            return text
        
        except ImportError:
            return f"URL fetch for '{url}': [Install requests and beautifulsoup4]"
        except Exception as e:
            return f"Error fetching URL: {str(e)}"


class NewsSearchTool(BaseTool):
    """Tool for searching news articles"""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(
            name="news_search",
            description="Search for recent news articles. Input should be a topic or search query."
        )
        self.api_key = api_key
    
    def run(self, query: str, num_results: int = 5) -> str:
        """
        Search for news articles.
        
        Args:
            query: Search query
            num_results: Number of results
            
        Returns:
            Formatted news results
        """
        # Placeholder - integrate with NewsAPI or similar
        results = [
            {
                "title": f"News about {query} - Article {i+1}",
                "source": "News Source",
                "date": "2025-01-02",
                "snippet": f"Latest news regarding {query}..."
            }
            for i in range(num_results)
        ]
        
        formatted = []
        for r in results:
            formatted.append(f"📰 {r['title']}\n   Source: {r['source']} | {r['date']}\n   {r['snippet']}")
        
        return "\n\n".join(formatted)
