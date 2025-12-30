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

        response = await self.llm.complete(prompt)
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

