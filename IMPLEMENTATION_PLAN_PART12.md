# King AI v2 - Implementation Plan Part 12
## Sub-Agent: Content (Blog/SEO)

**Target Timeline:** Week 8-9
**Objective:** Implement a content generation agent capable of creating SEO-optimized blog posts, marketing copy, and social media content.

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
| 10 | Sub-Agent: Research (Web/API) | ✅ Complete |
| 11 | Sub-Agent: Code Generator | ✅ Complete |
| **12** | **Sub-Agent: Content (Blog/SEO)** | 🔄 Current |
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

## Part 12 Scope

This part focuses on:
1. Blog post generation with SEO optimization
2. Marketing copy generation
3. Social media content creation
4. Content scheduling and management
5. SEO analysis and keyword research
6. Content performance tracking

---

## Task 12.1: Create SEO Utilities

**File:** `src/utils/seo_utils.py` (CREATE NEW FILE)

```python
"""
SEO Utilities - Tools for SEO analysis and optimization.
Provides keyword analysis, readability scoring, and SEO recommendations.
"""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from collections import Counter
import math


@dataclass
class KeywordAnalysis:
    """Analysis of keyword usage in content."""
    keyword: str
    count: int
    density: float  # Percentage
    in_title: bool
    in_headings: bool
    in_first_paragraph: bool
    in_meta_description: bool


@dataclass
class ReadabilityScore:
    """Readability metrics for content."""
    flesch_reading_ease: float  # 0-100, higher is easier
    flesch_kincaid_grade: float  # Grade level
    avg_sentence_length: float
    avg_word_length: float
    complex_word_percentage: float
    
    @property
    def reading_level(self) -> str:
        """Get human-readable reading level."""
        if self.flesch_reading_ease >= 80:
            return "Easy (6th grade)"
        elif self.flesch_reading_ease >= 60:
            return "Standard (8th-9th grade)"
        elif self.flesch_reading_ease >= 40:
            return "Difficult (College)"
        else:
            return "Very Difficult (Professional)"


@dataclass
class SEOScore:
    """Complete SEO analysis score."""
    overall_score: float  # 0-100
    title_score: float
    meta_score: float
    content_score: float
    keyword_score: float
    readability_score: float
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_score": self.overall_score,
            "title_score": self.title_score,
            "meta_score": self.meta_score,
            "content_score": self.content_score,
            "keyword_score": self.keyword_score,
            "readability_score": self.readability_score,
            "issues": self.issues,
            "recommendations": self.recommendations
        }


class SEOAnalyzer:
    """
    Analyzes content for SEO optimization.
    """
    
    # Ideal ranges for SEO metrics
    IDEAL_TITLE_LENGTH = (50, 60)
    IDEAL_META_LENGTH = (150, 160)
    IDEAL_KEYWORD_DENSITY = (1.0, 3.0)
    IDEAL_CONTENT_LENGTH = 1500
    
    def __init__(self):
        """Initialize SEO analyzer."""
        # Common stop words to exclude from keyword analysis
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were',
            'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this',
            'that', 'these', 'those', 'it', 'its', 'as', 'if', 'when', 'than'
        }
    
    def analyze(
        self,
        content: str,
        title: str = "",
        meta_description: str = "",
        target_keywords: List[str] = None
    ) -> SEOScore:
        """
        Perform complete SEO analysis.
        
        Args:
            content: Main content text
            title: Page/post title
            meta_description: Meta description
            target_keywords: Target keywords to optimize for
            
        Returns:
            Complete SEO score
        """
        issues = []
        recommendations = []
        
        # Title analysis
        title_score, title_issues = self._analyze_title(title, target_keywords)
        issues.extend(title_issues)
        
        # Meta description analysis
        meta_score, meta_issues = self._analyze_meta(meta_description, target_keywords)
        issues.extend(meta_issues)
        
        # Content analysis
        content_score, content_issues = self._analyze_content(content)
        issues.extend(content_issues)
        
        # Keyword analysis
        keyword_score = 100
        if target_keywords:
            keyword_score, keyword_issues = self._analyze_keywords(
                content, title, meta_description, target_keywords
            )
            issues.extend(keyword_issues)
        
        # Readability
        readability = self.calculate_readability(content)
        readability_score = min(100, readability.flesch_reading_ease)
        
        if readability.flesch_reading_ease < 50:
            recommendations.append("Consider simplifying language for better readability")
        
        # Calculate overall score
        overall_score = (
            title_score * 0.15 +
            meta_score * 0.15 +
            content_score * 0.30 +
            keyword_score * 0.25 +
            readability_score * 0.15
        )
        
        # Generate recommendations
        recommendations.extend(self._generate_recommendations(
            title_score, meta_score, content_score, keyword_score, content
        ))
        
        return SEOScore(
            overall_score=overall_score,
            title_score=title_score,
            meta_score=meta_score,
            content_score=content_score,
            keyword_score=keyword_score,
            readability_score=readability_score,
            issues=issues,
            recommendations=recommendations
        )
    
    def _analyze_title(
        self,
        title: str,
        keywords: List[str] = None
    ) -> tuple[float, List[str]]:
        """Analyze title for SEO."""
        score = 100
        issues = []
        
        if not title:
            return 0, ["Missing title"]
        
        title_len = len(title)
        
        if title_len < self.IDEAL_TITLE_LENGTH[0]:
            score -= 20
            issues.append(f"Title too short ({title_len} chars, ideal: 50-60)")
        elif title_len > self.IDEAL_TITLE_LENGTH[1]:
            score -= 15
            issues.append(f"Title too long ({title_len} chars, ideal: 50-60)")
        
        # Check for keyword in title
        if keywords:
            title_lower = title.lower()
            has_keyword = any(kw.lower() in title_lower for kw in keywords)
            if not has_keyword:
                score -= 25
                issues.append("Target keyword not in title")
        
        return max(0, score), issues
    
    def _analyze_meta(
        self,
        meta: str,
        keywords: List[str] = None
    ) -> tuple[float, List[str]]:
        """Analyze meta description for SEO."""
        score = 100
        issues = []
        
        if not meta:
            return 0, ["Missing meta description"]
        
        meta_len = len(meta)
        
        if meta_len < self.IDEAL_META_LENGTH[0]:
            score -= 20
            issues.append(f"Meta description too short ({meta_len} chars)")
        elif meta_len > self.IDEAL_META_LENGTH[1]:
            score -= 15
            issues.append(f"Meta description too long ({meta_len} chars)")
        
        # Check for keyword in meta
        if keywords:
            meta_lower = meta.lower()
            has_keyword = any(kw.lower() in meta_lower for kw in keywords)
            if not has_keyword:
                score -= 20
                issues.append("Target keyword not in meta description")
        
        return max(0, score), issues
    
    def _analyze_content(self, content: str) -> tuple[float, List[str]]:
        """Analyze content quality for SEO."""
        score = 100
        issues = []
        
        word_count = len(content.split())
        
        if word_count < 300:
            score -= 40
            issues.append(f"Content too short ({word_count} words, minimum 300)")
        elif word_count < self.IDEAL_CONTENT_LENGTH:
            score -= 20
            issues.append(f"Content could be longer ({word_count} words, ideal: 1500+)")
        
        # Check for headings
        heading_pattern = r'^#{1,6}\s|<h[1-6]>'
        has_headings = bool(re.search(heading_pattern, content, re.MULTILINE | re.IGNORECASE))
        if not has_headings and word_count > 500:
            score -= 15
            issues.append("Content lacks headings for structure")
        
        # Check for paragraphs (content structure)
        paragraphs = [p for p in content.split('\n\n') if p.strip()]
        if len(paragraphs) < 3 and word_count > 300:
            score -= 10
            issues.append("Content needs more paragraph breaks")
        
        return max(0, score), issues
    
    def _analyze_keywords(
        self,
        content: str,
        title: str,
        meta: str,
        keywords: List[str]
    ) -> tuple[float, List[str]]:
        """Analyze keyword usage."""
        score = 100
        issues = []
        
        content_lower = content.lower()
        word_count = len(content.split())
        
        for keyword in keywords:
            kw_lower = keyword.lower()
            count = content_lower.count(kw_lower)
            density = (count / word_count) * 100 if word_count > 0 else 0
            
            if count == 0:
                score -= 30
                issues.append(f"Keyword '{keyword}' not found in content")
            elif density < self.IDEAL_KEYWORD_DENSITY[0]:
                score -= 10
                issues.append(f"Keyword '{keyword}' density too low ({density:.1f}%)")
            elif density > self.IDEAL_KEYWORD_DENSITY[1]:
                score -= 15
                issues.append(f"Keyword '{keyword}' may be over-optimized ({density:.1f}%)")
        
        return max(0, score), issues
    
    def calculate_readability(self, text: str) -> ReadabilityScore:
        """Calculate readability metrics."""
        # Clean and split text
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        words = re.findall(r'\b\w+\b', text.lower())
        
        if not sentences or not words:
            return ReadabilityScore(0, 0, 0, 0, 0)
        
        # Count syllables (simplified)
        def count_syllables(word: str) -> int:
            word = word.lower()
            count = 0
            vowels = 'aeiouy'
            prev_vowel = False
            
            for char in word:
                is_vowel = char in vowels
                if is_vowel and not prev_vowel:
                    count += 1
                prev_vowel = is_vowel
            
            if word.endswith('e'):
                count -= 1
            
            return max(1, count)
        
        total_syllables = sum(count_syllables(w) for w in words)
        total_words = len(words)
        total_sentences = len(sentences)
        
        # Calculate metrics
        avg_sentence_length = total_words / total_sentences
        avg_syllables_per_word = total_syllables / total_words
        avg_word_length = sum(len(w) for w in words) / total_words
        
        # Complex words (3+ syllables)
        complex_words = sum(1 for w in words if count_syllables(w) >= 3)
        complex_percentage = (complex_words / total_words) * 100
        
        # Flesch Reading Ease
        flesch_reading_ease = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        flesch_reading_ease = max(0, min(100, flesch_reading_ease))
        
        # Flesch-Kincaid Grade Level
        flesch_kincaid = (0.39 * avg_sentence_length) + (11.8 * avg_syllables_per_word) - 15.59
        flesch_kincaid = max(0, flesch_kincaid)
        
        return ReadabilityScore(
            flesch_reading_ease=flesch_reading_ease,
            flesch_kincaid_grade=flesch_kincaid,
            avg_sentence_length=avg_sentence_length,
            avg_word_length=avg_word_length,
            complex_word_percentage=complex_percentage
        )
    
    def extract_keywords(
        self,
        text: str,
        top_n: int = 10
    ) -> List[Dict[str, Any]]:
        """Extract potential keywords from text."""
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        words = [w for w in words if w not in self.stop_words]
        
        # Count frequencies
        word_freq = Counter(words)
        
        # Get top keywords
        keywords = []
        for word, count in word_freq.most_common(top_n):
            density = (count / len(words)) * 100 if words else 0
            keywords.append({
                "keyword": word,
                "count": count,
                "density": round(density, 2)
            })
        
        return keywords
    
    def _generate_recommendations(
        self,
        title_score: float,
        meta_score: float,
        content_score: float,
        keyword_score: float,
        content: str
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        if title_score < 80:
            recommendations.append("Optimize title: include target keyword, keep 50-60 characters")
        
        if meta_score < 80:
            recommendations.append("Improve meta description: include keyword, use 150-160 characters")
        
        if content_score < 80:
            word_count = len(content.split())
            if word_count < 1500:
                recommendations.append(f"Expand content to at least 1500 words (currently {word_count})")
            recommendations.append("Add subheadings (H2, H3) to structure content")
        
        if keyword_score < 80:
            recommendations.append("Increase keyword usage naturally throughout content")
        
        # Check for internal/external links
        if 'http' not in content and '[' not in content:
            recommendations.append("Add relevant internal and external links")
        
        # Check for images
        if '![' not in content and '<img' not in content:
            recommendations.append("Add images with alt text for better engagement")
        
        return recommendations
```

---

## Task 12.2: Create Content Generator Agent

**File:** `src/agents/content.py` (REPLACE EXISTING FILE)

```python
"""
Content Agent - AI-powered content generation and optimization.
Creates SEO-optimized blog posts, marketing copy, and social media content.
"""

import re
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from src.agents.base import BaseAgent, AgentCapability, AgentResult
from src.utils.ollama_client import OllamaClient
from src.utils.seo_utils import SEOAnalyzer, SEOScore
from src.utils.structured_logging import get_logger

logger = get_logger("content_agent")


class ContentType(str, Enum):
    """Types of content to generate."""
    BLOG_POST = "blog_post"
    LANDING_PAGE = "landing_page"
    PRODUCT_DESCRIPTION = "product_description"
    EMAIL_MARKETING = "email_marketing"
    SOCIAL_MEDIA = "social_media"
    AD_COPY = "ad_copy"
    PRESS_RELEASE = "press_release"


class ContentTone(str, Enum):
    """Tone of content."""
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    FRIENDLY = "friendly"
    AUTHORITATIVE = "authoritative"
    HUMOROUS = "humorous"
    URGENT = "urgent"
    INSPIRATIONAL = "inspirational"


@dataclass
class ContentRequest:
    """Request for content generation."""
    topic: str
    content_type: ContentType
    target_keywords: List[str] = field(default_factory=list)
    tone: ContentTone = ContentTone.PROFESSIONAL
    target_word_count: int = 1500
    target_audience: str = "general"
    call_to_action: Optional[str] = None
    include_sections: List[str] = field(default_factory=list)
    brand_guidelines: Optional[str] = None


@dataclass
class GeneratedContent:
    """Generated content result."""
    title: str
    meta_description: str
    content: str
    word_count: int
    content_type: ContentType
    seo_score: Optional[SEOScore] = None
    sections: List[Dict[str, str]] = field(default_factory=list)
    social_snippets: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "meta_description": self.meta_description,
            "content": self.content,
            "word_count": self.word_count,
            "content_type": self.content_type.value,
            "seo_score": self.seo_score.to_dict() if self.seo_score else None,
            "sections": self.sections,
            "social_snippets": self.social_snippets
        }


class ContentAgent(BaseAgent):
    """
    Agent for generating SEO-optimized content.
    Creates various types of marketing and blog content.
    """
    
    CAPABILITIES = [
        AgentCapability.CONTENT_GENERATION,
        AgentCapability.SEO_OPTIMIZATION,
        AgentCapability.COPYWRITING
    ]
    
    # Content generation prompts
    PROMPTS = {
        ContentType.BLOG_POST: """Write a comprehensive, SEO-optimized blog post about: {topic}

Target Keywords: {keywords}
Tone: {tone}
Target Audience: {audience}
Word Count Target: {word_count} words

Requirements:
1. Create an engaging, keyword-optimized title (50-60 characters)
2. Write a compelling meta description (150-160 characters)
3. Structure with clear H2 and H3 headings
4. Include the main keyword in the first paragraph
5. Use keywords naturally throughout (1-3% density)
6. Include actionable tips and insights
7. End with a strong call-to-action

{additional_instructions}

Format:
TITLE: [Your title]
META: [Meta description]
CONTENT:
[Full blog post content with markdown formatting]
""",
        ContentType.PRODUCT_DESCRIPTION: """Write a compelling product description for: {topic}

Target Keywords: {keywords}
Tone: {tone}
Target Audience: {audience}

Requirements:
1. Highlight key benefits (not just features)
2. Use sensory and emotional language
3. Include social proof elements
4. Address potential objections
5. Include a strong call-to-action
6. Keep it scannable with bullet points

{additional_instructions}

Format:
TITLE: [Product headline]
META: [Meta description]
CONTENT:
[Product description]
""",
        ContentType.EMAIL_MARKETING: """Write a marketing email about: {topic}

Target Keywords: {keywords}
Tone: {tone}
Target Audience: {audience}
Call to Action: {cta}

Requirements:
1. Compelling subject line (under 50 characters)
2. Preview text (90 characters)
3. Personalized opening
4. Clear value proposition
5. Single focused CTA
6. P.S. line for urgency

{additional_instructions}

Format:
SUBJECT: [Subject line]
PREVIEW: [Preview text]
CONTENT:
[Email body]
""",
        ContentType.SOCIAL_MEDIA: """Create social media content about: {topic}

Target Keywords: {keywords}
Tone: {tone}
Target Audience: {audience}

Create posts for:
1. Twitter/X (280 characters max)
2. LinkedIn (longer, professional)
3. Instagram caption
4. Facebook post

{additional_instructions}

Format:
TWITTER: [Tweet]
LINKEDIN: [LinkedIn post]
INSTAGRAM: [Instagram caption with hashtags]
FACEBOOK: [Facebook post]
""",
        ContentType.AD_COPY: """Write advertising copy for: {topic}

Target Keywords: {keywords}
Tone: {tone}
Target Audience: {audience}
Call to Action: {cta}

Create:
1. Google Ads (3 headlines 30 chars each, 2 descriptions 90 chars each)
2. Facebook Ad (primary text, headline, description)
3. Display Ad (headline + tagline)

{additional_instructions}

Format:
GOOGLE_HEADLINES:
- [Headline 1]
- [Headline 2]
- [Headline 3]
GOOGLE_DESCRIPTIONS:
- [Description 1]
- [Description 2]
FACEBOOK:
Primary: [Primary text]
Headline: [Headline]
Description: [Description]
DISPLAY:
Headline: [Headline]
Tagline: [Tagline]
"""
    }
    
    def __init__(self, llm_client: OllamaClient):
        """
        Initialize content agent.
        
        Args:
            llm_client: LLM client for generation
        """
        super().__init__("content", llm_client)
        self.seo_analyzer = SEOAnalyzer()
    
    async def execute(self, task: Dict[str, Any]) -> AgentResult:
        """Execute a content generation task."""
        try:
            request = ContentRequest(
                topic=task.get("topic", ""),
                content_type=ContentType(task.get("content_type", "blog_post")),
                target_keywords=task.get("keywords", []),
                tone=ContentTone(task.get("tone", "professional")),
                target_word_count=task.get("word_count", 1500),
                target_audience=task.get("audience", "general"),
                call_to_action=task.get("cta"),
                include_sections=task.get("sections", []),
                brand_guidelines=task.get("brand_guidelines")
            )
            
            content = await self.generate(request)
            
            return AgentResult(
                success=True,
                data=content.to_dict(),
                message=f"Generated {content.content_type.value}: {content.word_count} words"
            )
            
        except Exception as e:
            logger.error(f"Content generation failed: {e}")
            return AgentResult(
                success=False,
                error=str(e),
                message="Content generation failed"
            )
    
    async def generate(self, request: ContentRequest) -> GeneratedContent:
        """
        Generate content based on request.
        
        Args:
            request: Content generation request
            
        Returns:
            Generated content
        """
        logger.info(
            f"Generating content",
            type=request.content_type.value,
            topic=request.topic[:50]
        )
        
        # Build prompt
        prompt = self._build_prompt(request)
        
        # Generate content
        response = await self.llm.generate(prompt)
        
        # Parse response
        content = self._parse_response(response, request.content_type)
        
        # Analyze SEO if applicable
        if request.content_type in [ContentType.BLOG_POST, ContentType.LANDING_PAGE, ContentType.PRODUCT_DESCRIPTION]:
            seo_score = self.seo_analyzer.analyze(
                content=content.content,
                title=content.title,
                meta_description=content.meta_description,
                target_keywords=request.target_keywords
            )
            content.seo_score = seo_score
            
            # If SEO score is low, try to improve
            if seo_score.overall_score < 70:
                content = await self._improve_seo(content, request, seo_score)
        
        # Generate social snippets
        if request.content_type == ContentType.BLOG_POST:
            content.social_snippets = await self._generate_social_snippets(content)
        
        logger.info(
            f"Content generated",
            type=request.content_type.value,
            words=content.word_count,
            seo_score=content.seo_score.overall_score if content.seo_score else None
        )
        
        return content
    
    def _build_prompt(self, request: ContentRequest) -> str:
        """Build generation prompt from request."""
        template = self.PROMPTS.get(
            request.content_type,
            self.PROMPTS[ContentType.BLOG_POST]
        )
        
        additional = []
        if request.brand_guidelines:
            additional.append(f"Brand Guidelines: {request.brand_guidelines}")
        if request.include_sections:
            additional.append(f"Include sections: {', '.join(request.include_sections)}")
        
        return template.format(
            topic=request.topic,
            keywords=", ".join(request.target_keywords) if request.target_keywords else "N/A",
            tone=request.tone.value,
            audience=request.target_audience,
            word_count=request.target_word_count,
            cta=request.call_to_action or "Learn more",
            additional_instructions="\n".join(additional) if additional else ""
        )
    
    def _parse_response(
        self,
        response: str,
        content_type: ContentType
    ) -> GeneratedContent:
        """Parse LLM response into structured content."""
        title = ""
        meta = ""
        content = ""
        sections = []
        
        # Extract title
        title_match = re.search(r'TITLE:\s*(.+?)(?:\n|$)', response)
        if title_match:
            title = title_match.group(1).strip()
        
        # Extract meta
        meta_match = re.search(r'META:\s*(.+?)(?:\n|$)', response)
        if meta_match:
            meta = meta_match.group(1).strip()
        
        # Extract content
        content_match = re.search(r'CONTENT:\s*(.+)', response, re.DOTALL)
        if content_match:
            content = content_match.group(1).strip()
        else:
            # If no CONTENT marker, use everything after META
            if meta:
                parts = response.split(meta)
                if len(parts) > 1:
                    content = parts[1].strip()
            else:
                content = response
        
        # Extract sections from headings
        heading_pattern = r'^##\s+(.+?)$'
        for match in re.finditer(heading_pattern, content, re.MULTILINE):
            section_title = match.group(1)
            sections.append({"heading": section_title})
        
        word_count = len(content.split())
        
        return GeneratedContent(
            title=title or "Untitled",
            meta_description=meta or "",
            content=content,
            word_count=word_count,
            content_type=content_type,
            sections=sections
        )
    
    async def _improve_seo(
        self,
        content: GeneratedContent,
        request: ContentRequest,
        seo_score: SEOScore
    ) -> GeneratedContent:
        """Attempt to improve content SEO."""
        improvements = []
        
        for issue in seo_score.issues[:3]:
            improvements.append(issue)
        
        prompt = f"""Improve the following content to address these SEO issues:

Issues:
{chr(10).join(f'- {issue}' for issue in improvements)}

Current Title: {content.title}
Current Meta: {content.meta_description}
Target Keywords: {', '.join(request.target_keywords)}

Current Content:
{content.content[:3000]}

Provide improved:
TITLE: [Improved title]
META: [Improved meta description]
FIRST_PARAGRAPH: [Improved first paragraph with keyword]
"""
        
        response = await self.llm.generate(prompt)
        
        # Parse improvements
        new_title = re.search(r'TITLE:\s*(.+?)(?:\n|$)', response)
        new_meta = re.search(r'META:\s*(.+?)(?:\n|$)', response)
        new_first = re.search(r'FIRST_PARAGRAPH:\s*(.+?)(?:\n\n|$)', response, re.DOTALL)
        
        if new_title:
            content.title = new_title.group(1).strip()
        if new_meta:
            content.meta_description = new_meta.group(1).strip()
        if new_first:
            # Replace first paragraph
            paragraphs = content.content.split('\n\n')
            if paragraphs:
                paragraphs[0] = new_first.group(1).strip()
                content.content = '\n\n'.join(paragraphs)
        
        # Recalculate SEO score
        content.seo_score = self.seo_analyzer.analyze(
            content=content.content,
            title=content.title,
            meta_description=content.meta_description,
            target_keywords=request.target_keywords
        )
        
        return content
    
    async def _generate_social_snippets(
        self,
        content: GeneratedContent
    ) -> List[str]:
        """Generate social media snippets from content."""
        prompt = f"""Create 3 social media posts promoting this blog post:

Title: {content.title}
Summary: {content.content[:500]}

Create:
1. A Twitter post (under 280 chars)
2. A LinkedIn post (2-3 sentences)
3. A casual Facebook post

Format each on a new line starting with the platform name.
"""
        
        response = await self.llm.generate(prompt)
        
        snippets = []
        for line in response.split('\n'):
            line = line.strip()
            if line and len(line) > 20:
                snippets.append(line)
        
        return snippets[:3]
    
    async def generate_blog_post(
        self,
        topic: str,
        keywords: List[str],
        word_count: int = 1500
    ) -> GeneratedContent:
        """Convenience method for blog post generation."""
        request = ContentRequest(
            topic=topic,
            content_type=ContentType.BLOG_POST,
            target_keywords=keywords,
            target_word_count=word_count
        )
        return await self.generate(request)
    
    async def generate_product_description(
        self,
        product_name: str,
        features: List[str],
        benefits: List[str],
        keywords: List[str] = None
    ) -> GeneratedContent:
        """Generate product description."""
        topic = f"{product_name}\n\nFeatures:\n" + "\n".join(f"- {f}" for f in features)
        topic += "\n\nBenefits:\n" + "\n".join(f"- {b}" for b in benefits)
        
        request = ContentRequest(
            topic=topic,
            content_type=ContentType.PRODUCT_DESCRIPTION,
            target_keywords=keywords or [],
            tone=ContentTone.FRIENDLY
        )
        return await self.generate(request)
    
    async def generate_email_sequence(
        self,
        topic: str,
        num_emails: int = 3,
        goal: str = "conversion"
    ) -> List[GeneratedContent]:
        """Generate a sequence of marketing emails."""
        emails = []
        
        email_types = [
            ("Introduction and value proposition", "Learn more"),
            ("Social proof and benefits", "See results"),
            ("Urgency and final offer", "Get started now")
        ]
        
        for i in range(min(num_emails, len(email_types))):
            focus, cta = email_types[i]
            
            request = ContentRequest(
                topic=f"{topic} - Email {i+1}: {focus}",
                content_type=ContentType.EMAIL_MARKETING,
                tone=ContentTone.FRIENDLY,
                call_to_action=cta
            )
            
            email = await self.generate(request)
            emails.append(email)
        
        return emails
```

---

## Task 12.3: Create Content API Routes

**File:** `src/api/routes/content.py` (CREATE NEW FILE)

```python
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
from src.utils.ollama_client import OllamaClient
from src.utils.structured_logging import get_logger

logger = get_logger("content_api")
router = APIRouter(prefix="/content", tags=["content"])


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
        llm = OllamaClient()
        _content_agent = ContentAgent(llm)
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
```

---

## Testing Requirements

**File:** `tests/test_content.py` (CREATE NEW FILE)

```python
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
    def mock_llm(self):
        llm = AsyncMock()
        llm.generate = AsyncMock(return_value="""
TITLE: 10 Tips for Better SEO in 2024
META: Learn the top SEO strategies to improve your search rankings and drive more organic traffic to your website.
CONTENT:
# 10 Tips for Better SEO in 2024

Search engine optimization remains crucial for online success.

## 1. Quality Content

Create valuable, original content that serves your audience.

## 2. Keyword Research

Use tools to find the right keywords for your niche.
""")
        return llm
    
    @pytest.fixture
    def agent(self, mock_llm):
        return ContentAgent(mock_llm)
    
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
```

---

## Acceptance Criteria

- [ ] `src/utils/seo_utils.py` - Complete SEO analysis tools
- [ ] `src/agents/content.py` - Full content generation agent
- [ ] `src/api/routes/content.py` - REST API endpoints
- [ ] `tests/test_content.py` - All tests passing
- [ ] SEO scoring accurate and actionable
- [ ] Multiple content types supported
- [ ] Readability analysis working
- [ ] Social media snippet generation working

---

## File Summary

| Action | File Path |
|--------|-----------|
| CREATE | `src/utils/seo_utils.py` |
| REPLACE | `src/agents/content.py` |
| CREATE | `src/api/routes/content.py` |
| CREATE | `tests/test_content.py` |

---

*End of Part 12*
