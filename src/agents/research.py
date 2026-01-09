"""
Research Agent - Comprehensive research capabilities.
Handles web research, market analysis, and competitor tracking.
"""

import asyncio
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import hashlib
import os
import httpx
import inspect

from config.settings import settings
from src.agents.base import BaseAgent, AgentCapability, AgentResult
from src.utils.web_scraper import WebScraper, ScrapedPage, ContentExtractor
from src.utils.search_client import UnifiedSearchClient, SearchResponse, SearchResult, SearchEngine
from src.utils.ollama_client import OllamaClient
from src.database.vector_store import VectorStore
from src.utils.structured_logging import get_logger

logger = get_logger("research_agent")


class SerpAPIClient:
    """SerpAPI client for web search."""
    
    def __init__(self):
        self.api_key = settings.serpapi_key
        self.base_url = "https://serpapi.com/search"
    
    async def search(
        self,
        query: str,
        num_results: int = 10,
        search_type: str = "google"
    ) -> List[Dict[str, Any]]:
        """
        Search the web using SerpAPI.
        
        Args:
            query: Search query
            num_results: Number of results
            search_type: Engine type
            
        Returns:
            List of search results
        """
        if not self.api_key:
            logger.warning("SerpAPI key not configured, using mock data")
            return self._mock_results(query)
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                self.base_url,
                params={
                    "q": query,
                    "api_key": self.api_key,
                    "engine": search_type,
                    "num": num_results
                }
            )
            response.raise_for_status()
            data = response.json()
            
            results = []
            for item in data.get("organic_results", []):
                results.append({
                    "title": item.get("title"),
                    "url": item.get("link"),
                    "snippet": item.get("snippet"),
                    "position": item.get("position")
                })
            
            return results
    
    def _mock_results(self, query: str) -> List[Dict[str, Any]]:
        """Return mock results for development."""
        return [
            {
                "title": f"Result for: {query}",
                "url": "https://example.com",
                "snippet": "Mock search result",
                "position": 1
            }
        ]


class ResearchType(str, Enum):
    """Types of research tasks."""
    WEB_SEARCH = "web_search"
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
    
    # Function calling schema for LLM integration
    FUNCTION_SCHEMA = {
        "name": "research",
        "description": "Perform web research and market analysis. Supports web search, market research, competitor analysis, and trend analysis.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The research query or topic to investigate"
                },
                "research_type": {
                    "type": "string",
                    "enum": ["web_search", "market_research", "competitor_analysis", "trend_analysis", "product_research", "general_search"],
                    "description": "Type of research to perform"
                },
                "depth": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 3,
                    "default": 1,
                    "description": "Research depth: 1=shallow, 2=medium, 3=deep"
                },
                "max_sources": {
                    "type": "integer",
                    "default": 10,
                    "description": "Maximum number of sources to analyze"
                }
            },
            "required": ["query"]
        }
    }
    
    def __init__(
        self,
        llm_client: Optional[OllamaClient] = None,
        vector_store: Optional[VectorStore] = None
    ):
        """
        Initialize research agent.
        
        Args:
            llm_client: LLM client for analysis (optional, creates default if None)
            vector_store: Vector store for caching (optional, uses singleton if None)
        """
        super().__init__("research", llm_client)
        # Use provided vector_store or import singleton
        if vector_store is None:
            from src.database.vector_store import vector_store as default_vector_store
            self.vector_store = default_vector_store
        else:
            self.vector_store = vector_store
        self.scraper = WebScraper(rate_limit=0.5)
        self.search_client = UnifiedSearchClient()
        self.serp_client = SerpAPIClient()
        self.extractor = ContentExtractor()
    
    async def execute(self, task: Dict[str, Any]) -> dict:
        """Execute a research task."""
        try:
            # Get the research type, defaulting to general_search if invalid
            task_type = task.get("type", "general_search")
            
            # Map common task type names to valid ResearchTypes
            type_mapping = {
                "research": ResearchType.GENERAL_SEARCH,
                "web_search": ResearchType.WEB_SEARCH,
                "market_research": ResearchType.MARKET_RESEARCH,
                "competitor_analysis": ResearchType.COMPETITOR_ANALYSIS,
                "trend_analysis": ResearchType.TREND_ANALYSIS,
                "product_research": ResearchType.PRODUCT_RESEARCH,
                "general_search": ResearchType.GENERAL_SEARCH,
            }
            
            # Try to map the type, fallback to general_search
            if task_type in type_mapping:
                research_type = type_mapping[task_type]
            else:
                try:
                    research_type = ResearchType(task_type)
                except ValueError:
                    logger.warning(f"Unknown research type '{task_type}', defaulting to general_search")
                    research_type = ResearchType.GENERAL_SEARCH
            
            query = ResearchQuery(
                query=task.get("query", task.get("description", task.get("name", ""))),
                research_type=research_type,
                depth=task.get("depth", 1),
                max_sources=task.get("max_sources", 10)
            )

            report = await self.research(query)

            report_dict = report.to_dict()
            if inspect.isawaitable(report_dict):
                report_dict = await report_dict
            
            return {
                "success": True,
                "output": report_dict,
                "error": None,
                "metadata": {"sources_count": len(report.sources)}
            }
        except Exception as e:
            logger.error(f"Research failed: {e}")
            return {
                "success": False,
                "output": None,
                "error": str(e),
                "metadata": {}
            }
    
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
        """Search for sources using SerpAPI."""
        serp_results = await self.serp_client.search(
            query.query,
            num_results=query.max_sources
        )
        
        # Convert to SearchResult objects
        results = []
        for item in serp_results:
            result = SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("snippet", ""),
                position=item.get("position", 0),
                engine=SearchEngine.GOOGLE  # Default to Google
            )
            results.append(result)
        
        return results
    
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
        # Sanitize content to prevent prompt injection
        sanitized_title = page.title.replace("```", "").replace("\n", " ")[:200]
        sanitized_content = page.text_content.replace("```", "")[:3000]
        
        prompt = f"""Summarize the following content in relation to: "{query}"

Content from {sanitized_title}:
{sanitized_content}

Provide a 2-3 sentence summary focusing on the most relevant information."""

        response = await self.llm.complete(prompt)
        return response[:500]
    
    async def _generate_report(
        self,
        query: ResearchQuery,
        sources: List[ResearchSource]
    ) -> ResearchReport:
        """Generate final research report."""
        # Prepare source summaries with sanitization
        sanitized_sources = []
        for s in sources[:10]:
            sanitized_title = s.title.replace("```", "").replace("\n", " ")[:200]
            sanitized_summary = s.content_summary.replace("```", "")[:500]
            sanitized_sources.append(f"Source: {sanitized_title}\nSummary: {sanitized_summary}\n")
        
        source_text = "\n".join(sanitized_sources)
        
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

        response = await self.llm.complete(prompt)
        
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
    
    def _parse_report_response(self, response: str) -> Tuple[str, List[str]]:
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
            # Use deterministic hash for consistent IDs
            query_hash = hashlib.md5(report.query.encode()).hexdigest()
            research_id = f"research_{query_hash}_{int(datetime.now().timestamp())}"
            
            await self.vector_store.upsert(
                id=research_id,
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

