"""
Content API Routes - REST endpoints for content generation.
"""

from typing import Optional, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.agents.content import (
    ContentAgent,
    ContentRequest,
    ContentType,
    ContentTone
)
from src.utils.seo_utils import SEOAnalyzer
from src.utils.logging import logger

router = APIRouter(tags=["content"])


class GenerateContentRequest(BaseModel):
    """Request for content generation."""
    topic: str
    content_type: str = "blog_post"
    keywords: List[str] = []
    tone: str = "professional"
    word_count: int = 1500
    audience: str = "general"
    cta: Optional[str] = None


class BlogPostRequest(BaseModel):
    """Request for blog post generation."""
    topic: str
    keywords: List[str] = []
    word_count: int = 1500
    sections: List[str] = []


class ProductDescriptionRequest(BaseModel):
    """Request for product description."""
    product_name: str
    features: List[str]
    benefits: List[str]
    keywords: List[str] = []


class SEOAnalysisRequest(BaseModel):
    """Request for SEO analysis."""
    content: str
    title: str = ""
    meta_description: str = ""
    keywords: List[str] = []


class ContentResponse(BaseModel):
    """Response with generated content."""
    title: str
    meta_description: str
    content: str
    word_count: int
    content_type: str
    seo_score: Optional[dict] = None
    social_snippets: List[str] = []


class SEOResponse(BaseModel):
    """Response with SEO analysis."""
    overall_score: float
    title_score: float
    meta_score: float
    content_score: float
    keyword_score: float
    readability_score: float
    issues: List[str]
    recommendations: List[str]


# Global agent instance
_content_agent: Optional[ContentAgent] = None


def get_content_agent() -> ContentAgent:
    """Get or create content agent."""
    global _content_agent
    if _content_agent is None:
        _content_agent = ContentAgent()
    return _content_agent


@router.post("/generate", response_model=ContentResponse)
async def generate_content(request: GenerateContentRequest):
    """Generate content based on request."""
    try:
        agent = get_content_agent()
        
        content_request = ContentRequest(
            topic=request.topic,
            content_type=ContentType(request.content_type),
            target_keywords=request.keywords,
            tone=ContentTone(request.tone),
            target_word_count=request.word_count,
            target_audience=request.audience,
            call_to_action=request.cta
        )
        
        result = await agent.generate(content_request)
        
        return ContentResponse(
            title=result.title,
            meta_description=result.meta_description,
            content=result.content,
            word_count=result.word_count,
            content_type=result.content_type.value,
            seo_score=result.seo_score.to_dict() if result.seo_score else None,
            social_snippets=result.social_snippets
        )
        
    except Exception as e:
        logger.error(f"Content generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/blog", response_model=ContentResponse)
async def generate_blog_post(request: BlogPostRequest):
    """Generate a blog post."""
    try:
        agent = get_content_agent()
        
        result = await agent.generate_blog_post(
            topic=request.topic,
            keywords=request.keywords,
            word_count=request.word_count
        )
        
        return ContentResponse(
            title=result.title,
            meta_description=result.meta_description,
            content=result.content,
            word_count=result.word_count,
            content_type=result.content_type.value,
            seo_score=result.seo_score.to_dict() if result.seo_score else None,
            social_snippets=result.social_snippets
        )
        
    except Exception as e:
        logger.error(f"Blog generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/product", response_model=ContentResponse)
async def generate_product_description(request: ProductDescriptionRequest):
    """Generate a product description."""
    try:
        agent = get_content_agent()
        
        result = await agent.generate_product_description(
            product_name=request.product_name,
            features=request.features,
            benefits=request.benefits,
            keywords=request.keywords
        )
        
        return ContentResponse(
            title=result.title,
            meta_description=result.meta_description,
            content=result.content,
            word_count=result.word_count,
            content_type=result.content_type.value,
            seo_score=result.seo_score.to_dict() if result.seo_score else None,
            social_snippets=[]
        )
        
    except Exception as e:
        logger.error(f"Product description generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze-seo", response_model=SEOResponse)
async def analyze_seo(request: SEOAnalysisRequest):
    """Analyze content for SEO."""
    try:
        analyzer = SEOAnalyzer()
        
        result = analyzer.analyze(
            content=request.content,
            title=request.title,
            meta_description=request.meta_description,
            target_keywords=request.keywords
        )
        
        return SEOResponse(
            overall_score=result.overall_score,
            title_score=result.title_score,
            meta_score=result.meta_score,
            content_score=result.content_score,
            keyword_score=result.keyword_score,
            readability_score=result.readability_score,
            issues=result.issues,
            recommendations=result.recommendations
        )
        
    except Exception as e:
        logger.error(f"SEO analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/extract-keywords")
async def extract_keywords(
    content: str,
    top_n: int = 10
):
    """Extract keywords from content."""
    analyzer = SEOAnalyzer()
    keywords = analyzer.extract_keywords(content, top_n)
    return {"keywords": keywords}


@router.get("/content-types")
async def list_content_types():
    """List available content types."""
    return {
        "content_types": [ct.value for ct in ContentType],
        "tones": [t.value for t in ContentTone]
    }
