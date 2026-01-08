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
router = APIRouter(tags=["research"])


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
