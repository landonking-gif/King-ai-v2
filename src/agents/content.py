"""
Content Agent - AI-powered content generation and optimization.
Creates SEO-optimized blog posts, marketing copy, and social media content.
Integrates with DALL-E for AI-generated images.
"""

import re
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from src.agents.base import SubAgent
from src.utils.seo_utils import SEOAnalyzer, SEOScore
from src.utils.metrics import TASKS_EXECUTED
from src.utils.logging import logger

# DALL-E integration
try:
    from src.integrations.dalle_client import dalle_client, GeneratedImage
    DALLE_AVAILABLE = True
except ImportError:
    DALLE_AVAILABLE = False
    dalle_client = None


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
    generate_images: bool = True  # Auto-generate images with DALL-E
    image_style: str = "professional photography"


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
    featured_image_url: Optional[str] = None  # DALL-E generated image
    additional_images: List[str] = field(default_factory=list)  # Section images
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "meta_description": self.meta_description,
            "content": self.content,
            "word_count": self.word_count,
            "content_type": self.content_type.value,
            "seo_score": self.seo_score.to_dict() if self.seo_score else None,
            "sections": self.sections,
            "social_snippets": self.social_snippets,
            "featured_image_url": self.featured_image_url,
            "additional_images": self.additional_images,
        }


class ContentAgent(SubAgent):
    """
    Agent for generating SEO-optimized content with AI images.
    Creates various types of marketing and blog content.
    Integrates with DALL-E for automatic image generation.
    """
    
    name = "content"
    description = "Generates SEO-optimized blog posts, marketing copy, social media content, and AI-generated images."
    
    # Function calling schema for LLM integration
    FUNCTION_SCHEMA = {
        "name": "content",
        "description": "Generate SEO-optimized content including blog posts, product descriptions, marketing copy, and social media posts. Automatically generates images using DALL-E.",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "The main topic or subject of the content"
                },
                "content_type": {
                    "type": "string",
                    "enum": ["blog_post", "landing_page", "product_description", "email_marketing", "social_media", "ad_copy", "press_release"],
                    "default": "blog_post"
                },
                "keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Target SEO keywords"
                },
                "tone": {
                    "type": "string",
                    "enum": ["professional", "casual", "friendly", "authoritative", "humorous", "urgent", "inspirational"],
                    "default": "professional"
                },
                "word_count": {
                    "type": "integer",
                    "default": 1500,
                    "description": "Target word count"
                },
                "generate_images": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to generate images using DALL-E"
                },
                "image_style": {
                    "type": "string",
                    "default": "professional photography",
                    "description": "Style for DALL-E image generation"
                }
            },
            "required": ["topic"]
        }
    }
    
    # SEO optimization threshold
    MIN_SEO_SCORE_THRESHOLD = 70  # Minimum acceptable SEO score before improvement attempts
    
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
    
    def __init__(self):
        """Initialize content agent."""
        super().__init__()
        self.seo_analyzer = SEOAnalyzer()
        self.dalle_enabled = DALLE_AVAILABLE and dalle_client is not None
    
    async def execute(self, task: dict) -> dict:
        """Execute a content generation task."""
        try:
            # Handle both old and new task formats
            if "input" in task:
                task_data = task.get("input", {})
            else:
                task_data = task
            
            request = ContentRequest(
                topic=task_data.get("topic", ""),
                content_type=ContentType(task_data.get("content_type", "blog_post")),
                target_keywords=task_data.get("keywords", []),
                tone=ContentTone(task_data.get("tone", "professional")),
                target_word_count=task_data.get("word_count", 1500),
                target_audience=task_data.get("audience", "general"),
                call_to_action=task_data.get("cta"),
                include_sections=task_data.get("sections", []),
                brand_guidelines=task_data.get("brand_guidelines"),
                generate_images=task_data.get("generate_images", True),
                image_style=task_data.get("image_style", "professional photography"),
            )
            
            content = await self.generate(request)
            
            TASKS_EXECUTED.labels(agent=self.name, status="success").inc()
            
            return {
                "success": True,
                "output": content.to_dict(),
                "metadata": {
                    "type": "content_generation",
                    "word_count": content.word_count,
                    "seo_score": content.seo_score.overall_score if content.seo_score else None,
                    "has_images": content.featured_image_url is not None,
                }
            }
            
        except Exception as e:
            logger.error(f"Content generation failed: {e}")
            TASKS_EXECUTED.labels(agent=self.name, status="failed").inc()
            return {
                "success": False,
                "error": str(e)
            }
    
    async def generate(self, request: ContentRequest) -> GeneratedContent:
        """
        Generate content based on request.
        
        Args:
            request: Content generation request
            
        Returns:
            Generated content with optional AI-generated images
        """
        logger.info(
            f"Generating {request.content_type.value} content",
            topic=request.topic[:50]
        )
        
        # Build prompt
        prompt = self._build_prompt(request)
        
        # Generate content
        response = await self._ask_llm(prompt)
        
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
            if seo_score.overall_score < self.MIN_SEO_SCORE_THRESHOLD:
                content = await self._improve_seo(content, request, seo_score)
        
        # Generate social snippets
        if request.content_type == ContentType.BLOG_POST:
            content.social_snippets = await self._generate_social_snippets(content)
        
        # Generate images using DALL-E
        if request.generate_images and self.dalle_enabled:
            await self._generate_images(content, request)
        
        logger.info(
            f"Content generated",
            type=request.content_type.value,
            words=content.word_count,
            seo_score=content.seo_score.overall_score if content.seo_score else None,
            has_images=content.featured_image_url is not None
        )
        
        return content

    async def _generate_images(self, content: GeneratedContent, request: ContentRequest) -> None:
        """
        Generate images for content using DALL-E.
        
        Args:
            content: The generated content to add images to
            request: Original content request with image style preferences
        """
        if not self.dalle_enabled or dalle_client is None:
            return
        
        try:
            # Generate featured image based on content type
            if request.content_type == ContentType.BLOG_POST:
                image = await dalle_client.generate_blog_illustration(
                    content.title,
                    style=request.image_style
                )
                content.featured_image_url = image.url if image else None
                
            elif request.content_type == ContentType.PRODUCT_DESCRIPTION:
                image = await dalle_client.generate_product_image(
                    request.topic,
                    style=request.image_style
                )
                content.featured_image_url = image.url if image else None
                
            elif request.content_type == ContentType.SOCIAL_MEDIA:
                image = await dalle_client.generate(
                    f"Social media post visual for: {request.topic}. Style: {request.image_style}. Vibrant, eye-catching, shareable.",
                    size="1024x1024"
                )
                content.featured_image_url = image.url if image else None
                
            elif request.content_type == ContentType.LANDING_PAGE:
                # Generate hero image for landing page
                image = await dalle_client.generate(
                    f"Hero banner image for landing page about: {request.topic}. Style: {request.image_style}. Professional, modern, inspiring trust.",
                    size="1792x1024"  # Wide format for hero
                )
                content.featured_image_url = image.url if image else None
                
            logger.info(f"Generated featured image for {request.content_type.value}")
            
        except Exception as e:
            logger.warning(f"Failed to generate images: {e}")
            # Don't fail the whole content generation if images fail
    
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
        
        issues_list = '\n'.join(f'- {issue}' for issue in improvements)
        
        prompt = f"""Improve the following content to address these SEO issues:

Issues:
{issues_list}

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
        
        response = await self._ask_llm(prompt)
        
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
        
        response = await self._ask_llm(prompt)
        
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
