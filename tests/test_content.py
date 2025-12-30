"""Tests for content agent and SEO utilities."""

import pytest
from unittest.mock import AsyncMock

from src.agents.content import (
    ContentAgent,
    ContentRequest,
    ContentType,
    ContentTone,
    GeneratedContent
)
from src.utils.seo_utils import SEOAnalyzer, ReadabilityScore


class TestSEOAnalyzer:
    """Tests for SEO analyzer."""
    
    @pytest.fixture
    def analyzer(self):
        return SEOAnalyzer()
    
    def test_analyze_complete_content(self, analyzer):
        """Test complete SEO analysis."""
        content = """
This is a test article about SEO optimization. SEO is important for visibility.

## Why SEO Matters

SEO helps websites rank higher in search engines. Good SEO practices include
keyword optimization, quality content, and proper structure.

## Best Practices

Follow these SEO tips to improve your rankings. SEO is not just about keywords.
"""
        
        result = analyzer.analyze(
            content=content,
            title="Complete Guide to SEO Optimization",
            meta_description="Learn the best SEO practices for higher rankings.",
            target_keywords=["SEO", "optimization"]
        )
        
        assert result.overall_score > 0
        assert result.title_score > 0
        assert result.meta_score > 0
    
    def test_title_length_check(self, analyzer):
        """Test title length analysis."""
        # Too short
        score1, issues1 = analyzer._analyze_title("Short")
        assert score1 < 100
        assert any("short" in i.lower() for i in issues1)
        
        # Good length
        score2, issues2 = analyzer._analyze_title(
            "A Good SEO Title That Is The Right Length"
        )
        assert score2 >= 80
    
    def test_calculate_readability(self, analyzer):
        """Test readability calculation."""
        simple_text = "The cat sat on the mat. It was a nice day. The sun was shining."
        complex_text = "The implementation of sophisticated algorithmic methodologies necessitates comprehensive understanding of computational paradigms."
        
        simple_score = analyzer.calculate_readability(simple_text)
        complex_score = analyzer.calculate_readability(complex_text)
        
        assert simple_score.flesch_reading_ease > complex_score.flesch_reading_ease
    
    def test_extract_keywords(self, analyzer):
        """Test keyword extraction."""
        text = """
Python is a programming language. Python is great for data science.
Data science uses Python extensively. Machine learning with Python is popular.
"""
        
        keywords = analyzer.extract_keywords(text, top_n=5)
        
        assert len(keywords) <= 5
        assert any(kw["keyword"] == "python" for kw in keywords)
    
    def test_keyword_density_check(self, analyzer):
        """Test keyword density analysis."""
        content = "keyword " * 100  # 100% density
        
        score, issues = analyzer._analyze_keywords(
            content, "keyword", "keyword", ["keyword"]
        )
        
        assert score < 100
        assert any("over-optimized" in i.lower() for i in issues)


class TestContentAgent:
    """Tests for content agent."""
    
    @pytest.fixture
    def mock_llm_response(self):
        return """
TITLE: 10 Tips for Better SEO in 2024
META: Learn the top SEO strategies to improve your search rankings and drive more organic traffic to your website.
CONTENT:
# 10 Tips for Better SEO in 2024

Search engine optimization remains crucial for online success.

## 1. Quality Content

Create valuable, original content that serves your audience.

## 2. Keyword Research

Use tools to find the right keywords for your niche.
"""
    
    @pytest.fixture
    def agent(self, mock_llm_response):
        agent = ContentAgent()
        agent._ask_llm = AsyncMock(return_value=mock_llm_response)
        return agent
    
    def test_parse_response(self, agent):
        """Test response parsing."""
        response = """
TITLE: Test Title
META: Test meta description here
CONTENT:
This is the main content.

## Section One

Some text here.
"""
        
        result = agent._parse_response(response, ContentType.BLOG_POST)
        
        assert result.title == "Test Title"
        assert result.meta_description == "Test meta description here"
        assert "main content" in result.content
    
    @pytest.mark.asyncio
    async def test_generate_blog_post(self, agent):
        """Test blog post generation."""
        result = await agent.generate_blog_post(
            topic="SEO Tips",
            keywords=["SEO", "optimization"],
            word_count=500
        )
        
        assert result.title
        assert result.content
        assert result.content_type == ContentType.BLOG_POST
    
    @pytest.mark.asyncio
    async def test_generate_with_seo_score(self, agent):
        """Test that SEO score is calculated."""
        request = ContentRequest(
            topic="Test Topic",
            content_type=ContentType.BLOG_POST,
            target_keywords=["test", "topic"]
        )
        
        result = await agent.generate(request)
        
        assert result.seo_score is not None
        assert result.seo_score.overall_score >= 0


class TestReadabilityScore:
    """Tests for readability scoring."""
    
    def test_reading_level_easy(self):
        """Test easy reading level."""
        score = ReadabilityScore(
            flesch_reading_ease=85,
            flesch_kincaid_grade=5,
            avg_sentence_length=10,
            avg_word_length=4,
            complex_word_percentage=5
        )
        
        assert "Easy" in score.reading_level
    
    def test_reading_level_difficult(self):
        """Test difficult reading level."""
        score = ReadabilityScore(
            flesch_reading_ease=30,
            flesch_kincaid_grade=16,
            avg_sentence_length=25,
            avg_word_length=6,
            complex_word_percentage=30
        )
        
        assert "Difficult" in score.reading_level
