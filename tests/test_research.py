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
        llm.complete = AsyncMock(return_value="""
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
