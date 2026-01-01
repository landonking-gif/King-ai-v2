"""
Multi-Modal Agent - Extended content processing with audio/video/image capabilities.

Extends the Content Agent with multi-modal processing for:
- Audio transcription and analysis (podcasts, interviews, voice notes)
- Image analysis and product matching
- Video content generation planning
- Social media multi-format content
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import asyncio

from src.agents.base import BaseAgent, AgentResult
from src.agents.content import ContentAgent
from src.integrations.whisper_client import whisper_client, TranscriptionResult, AudioAnalysis
from src.integrations.clip_client import clip_client, ImageAnalysis
from src.integrations.dalle_client import dalle_client
from src.utils.structured_logging import get_logger

logger = get_logger("multimodal_agent")


class MediaType(str, Enum):
    """Types of media content."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    MIXED = "mixed"


class ContentFormat(str, Enum):
    """Output content formats."""
    BLOG_POST = "blog_post"
    SOCIAL_POST = "social_post"
    VIDEO_SCRIPT = "video_script"
    PODCAST_OUTLINE = "podcast_outline"
    PRODUCT_LISTING = "product_listing"
    AD_CREATIVE = "ad_creative"
    EMAIL_CAMPAIGN = "email_campaign"


@dataclass
class MultiModalContent:
    """Multi-modal content package."""
    primary_format: ContentFormat
    text_content: str
    images: List[str] = field(default_factory=list)
    audio_transcription: Optional[str] = None
    video_script: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class VideoAdPlan:
    """Plan for a video advertisement."""
    title: str
    duration_seconds: int
    scenes: List[Dict[str, Any]]
    script: str
    music_suggestions: List[str]
    target_platforms: List[str]
    estimated_production_cost: Optional[float] = None


class MultiModalAgent(BaseAgent):
    """
    Multi-modal content agent with audio, image, and video capabilities.
    
    Extends basic content generation to handle:
    - Audio transcription → blog posts, social content
    - Image analysis → product descriptions, SEO
    - Video planning → ad scripts, social videos
    - Cross-platform content adaptation
    """
    
    name = "MultiModalAgent"
    description = "Handles multi-modal content creation including audio, images, and video"
    
    def __init__(self, llm=None):
        super().__init__(llm)
        self.whisper = whisper_client
        self.clip = clip_client
        self.dalle = dalle_client
        self._content_agent = ContentAgent(llm)
    
    async def execute(self, task: str, context: dict = None) -> AgentResult:
        """
        Execute a multi-modal content task.
        
        Handles various multi-modal operations based on task content.
        """
        context = context or {}
        task_lower = task.lower()
        
        try:
            # Route to appropriate handler
            if "transcribe" in task_lower or "audio" in task_lower:
                result = await self._handle_audio_task(task, context)
            elif "analyze image" in task_lower or "product image" in task_lower:
                result = await self._handle_image_task(task, context)
            elif "video" in task_lower or "ad" in task_lower:
                result = await self._handle_video_task(task, context)
            elif "multi-modal" in task_lower or "content package" in task_lower:
                result = await self._create_content_package(task, context)
            else:
                # Fall back to standard content agent
                return await self._content_agent.execute(task, context)
            
            return AgentResult(
                success=True,
                output=result,
                agent_name=self.name,
                metadata={"task_type": "multi_modal"}
            )
            
        except Exception as e:
            logger.error(f"Multi-modal task failed: {e}")
            return AgentResult(
                success=False,
                output=str(e),
                agent_name=self.name,
                error=str(e)
            )
    
    async def _handle_audio_task(self, task: str, context: dict) -> Dict[str, Any]:
        """Process audio-related tasks."""
        audio_path = context.get("audio_path") or context.get("file_path")
        
        if not audio_path:
            return {"error": "No audio file path provided in context"}
        
        if not self.whisper.is_configured:
            return {"error": "Whisper not configured - OPENAI_API_KEY required"}
        
        # Transcribe audio
        analysis = await self.whisper.analyze_audio(
            audio_path=audio_path,
            llm_client=self.llm,
            include_summary=True,
            include_topics=True,
            include_action_items="action" in task.lower()
        )
        
        result = {
            "transcription": analysis.transcription.text,
            "duration_seconds": analysis.transcription.duration,
            "word_count": analysis.transcription.word_count,
            "language": analysis.transcription.language,
            "summary": analysis.summary,
            "key_topics": analysis.key_topics,
            "sentiment": analysis.sentiment
        }
        
        # Generate derived content if requested
        if "blog" in task.lower():
            blog_content = await self._generate_blog_from_audio(analysis)
            result["blog_post"] = blog_content
        
        if "social" in task.lower():
            social_content = await self._generate_social_from_audio(analysis)
            result["social_posts"] = social_content
        
        return result
    
    async def _handle_image_task(self, task: str, context: dict) -> Dict[str, Any]:
        """Process image-related tasks."""
        image_path = context.get("image_path") or context.get("file_path")
        
        if not image_path:
            return {"error": "No image file path provided in context"}
        
        if not self.clip.is_configured:
            return {"error": "CLIP not configured - OPENAI_API_KEY required"}
        
        # Analyze image
        analysis = await self.clip.analyze_image(image_path)
        
        result = {
            "description": analysis.description,
            "objects": analysis.objects,
            "colors": analysis.colors,
            "text_detected": analysis.text_detected,
            "suitable_for": analysis.suitable_for,
            "quality_score": analysis.quality_score
        }
        
        # Generate product description if requested
        if "product" in task.lower():
            product_description = await self._generate_product_description(analysis, context)
            result["product_description"] = product_description
        
        # Classify into categories if provided
        if context.get("categories"):
            scores = await self.clip.classify_image(
                image_path,
                context["categories"]
            )
            result["category_scores"] = scores
        
        return result
    
    async def _handle_video_task(self, task: str, context: dict) -> Dict[str, Any]:
        """Process video-related tasks."""
        # Video generation planning (actual video creation would use external services)
        
        product_name = context.get("product_name", "Product")
        target_audience = context.get("target_audience", "general consumers")
        duration = context.get("duration_seconds", 30)
        platforms = context.get("platforms", ["instagram", "tiktok", "youtube"])
        
        # Generate video ad plan
        video_plan = await self._create_video_ad_plan(
            product_name=product_name,
            target_audience=target_audience,
            duration=duration,
            platforms=platforms,
            context=context
        )
        
        result = {
            "title": video_plan.title,
            "duration_seconds": video_plan.duration_seconds,
            "scenes": video_plan.scenes,
            "script": video_plan.script,
            "music_suggestions": video_plan.music_suggestions,
            "target_platforms": video_plan.target_platforms
        }
        
        # Generate thumbnail concepts
        if "thumbnail" in task.lower():
            thumbnails = await self._generate_thumbnail_concepts(video_plan)
            result["thumbnail_concepts"] = thumbnails
        
        return result
    
    async def _create_content_package(self, task: str, context: dict) -> Dict[str, Any]:
        """Create a complete multi-modal content package."""
        topic = context.get("topic", task)
        formats = context.get("formats", [ContentFormat.BLOG_POST, ContentFormat.SOCIAL_POST])
        
        package = {
            "topic": topic,
            "created_at": datetime.utcnow().isoformat(),
            "content": {}
        }
        
        # Generate text content
        if ContentFormat.BLOG_POST in formats:
            blog_prompt = f"Write a comprehensive blog post about: {topic}"
            blog_result = await self._content_agent.execute(blog_prompt, context)
            package["content"]["blog_post"] = blog_result.output
        
        if ContentFormat.SOCIAL_POST in formats:
            social_prompt = f"Create social media posts about: {topic}"
            social_result = await self._content_agent.execute(social_prompt, context)
            package["content"]["social_posts"] = social_result.output
        
        # Generate images if DALL-E is available
        if self.dalle and hasattr(self.dalle, 'generate'):
            try:
                image_prompt = f"Professional image for: {topic}"
                image_result = await self.dalle.generate(image_prompt)
                package["content"]["generated_images"] = [image_result]
            except Exception as e:
                logger.warning(f"Image generation failed: {e}")
        
        if ContentFormat.VIDEO_SCRIPT in formats:
            video_result = await self._handle_video_task(
                f"Create video script for {topic}",
                {**context, "product_name": topic}
            )
            package["content"]["video_plan"] = video_result
        
        return package
    
    async def _generate_blog_from_audio(self, analysis: AudioAnalysis) -> str:
        """Generate a blog post from audio transcription."""
        prompt = f"""Convert this audio transcription into a well-structured blog post.

Transcription:
{analysis.transcription.text[:3000]}

Summary: {analysis.summary}
Key Topics: {', '.join(analysis.key_topics or [])}

Requirements:
1. Create an engaging title
2. Write an introduction that hooks readers
3. Organize content into logical sections with headers
4. Add a conclusion with key takeaways
5. Maintain the original insights while improving readability

Format as Markdown."""

        if self.llm:
            response = await self.llm.generate(prompt, temperature=0.7)
            return response
        return f"# Blog Post\n\n{analysis.transcription.text}"
    
    async def _generate_social_from_audio(self, analysis: AudioAnalysis) -> List[Dict[str, str]]:
        """Generate social media posts from audio content."""
        prompt = f"""Create social media posts from this audio content.

Summary: {analysis.summary}
Key Topics: {', '.join(analysis.key_topics or [])}

Generate:
1. One Twitter/X post (280 chars max)
2. One LinkedIn post (professional tone)
3. One Instagram caption (with emoji suggestions)

Format as JSON array with keys: platform, content, hashtags"""

        posts = []
        
        if self.llm:
            response = await self.llm.generate(prompt, temperature=0.7)
            try:
                import json
                import re
                json_match = re.search(r'\[.*\]', response, re.DOTALL)
                if json_match:
                    posts = json.loads(json_match.group())
            except Exception:
                pass
        
        return posts
    
    async def _generate_product_description(
        self,
        analysis: ImageAnalysis,
        context: dict
    ) -> Dict[str, str]:
        """Generate product description from image analysis."""
        product_name = context.get("product_name", "Product")
        
        prompt = f"""Create a product listing based on this image analysis:

Image Description: {analysis.description}
Objects Detected: {', '.join(analysis.objects)}
Colors: {', '.join(analysis.colors)}
Any Text: {analysis.text_detected or 'None'}

Product Name: {product_name}

Generate:
1. A compelling product title (60 chars max)
2. A persuasive product description (150 words)
3. Key features/bullet points (5 items)
4. SEO keywords (10 relevant terms)

Format as JSON with keys: title, description, features, keywords"""

        result = {
            "title": product_name,
            "description": analysis.description,
            "features": [],
            "keywords": []
        }
        
        if self.llm:
            response = await self.llm.generate(prompt, temperature=0.6)
            try:
                import json
                import re
                json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
            except Exception:
                pass
        
        return result
    
    async def _create_video_ad_plan(
        self,
        product_name: str,
        target_audience: str,
        duration: int,
        platforms: List[str],
        context: dict
    ) -> VideoAdPlan:
        """Create a video advertisement plan."""
        prompt = f"""Create a video ad plan for:

Product: {product_name}
Target Audience: {target_audience}
Duration: {duration} seconds
Platforms: {', '.join(platforms)}

Generate a JSON with:
1. title: Catchy ad title
2. scenes: Array of scene objects with:
   - scene_number
   - duration_seconds
   - visual_description
   - text_overlay
   - action
3. script: Full voiceover script
4. music_suggestions: 3 music style suggestions
5. target_platforms: optimized platform list

Make it engaging and platform-appropriate."""

        default_plan = VideoAdPlan(
            title=f"{product_name} Ad",
            duration_seconds=duration,
            scenes=[{"scene_number": 1, "description": "Product showcase", "duration_seconds": duration}],
            script=f"Introducing {product_name}...",
            music_suggestions=["Upbeat pop", "Electronic", "Acoustic"],
            target_platforms=platforms
        )
        
        if self.llm:
            response = await self.llm.generate(prompt, temperature=0.7)
            try:
                import json
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                    default_plan = VideoAdPlan(
                        title=data.get("title", default_plan.title),
                        duration_seconds=duration,
                        scenes=data.get("scenes", default_plan.scenes),
                        script=data.get("script", default_plan.script),
                        music_suggestions=data.get("music_suggestions", default_plan.music_suggestions),
                        target_platforms=data.get("target_platforms", platforms)
                    )
            except Exception as e:
                logger.warning(f"Failed to parse video plan: {e}")
        
        return default_plan
    
    async def _generate_thumbnail_concepts(self, video_plan: VideoAdPlan) -> List[str]:
        """Generate thumbnail image concepts for video."""
        prompt = f"""Suggest 3 thumbnail concepts for this video:

Title: {video_plan.title}
Script preview: {video_plan.script[:200]}

For each thumbnail, describe:
- Visual composition
- Text overlay (if any)
- Color scheme
- Emotional appeal"""

        concepts = []
        
        if self.llm:
            response = await self.llm.generate(prompt, temperature=0.7)
            # Split response into concepts
            concepts = [c.strip() for c in response.split("\n\n") if c.strip()][:3]
        
        return concepts
    
    async def transcribe_and_repurpose(
        self,
        audio_path: str,
        output_formats: List[ContentFormat] = None
    ) -> MultiModalContent:
        """
        Transcribe audio and repurpose into multiple content formats.
        
        Args:
            audio_path: Path to audio file
            output_formats: Desired output formats
            
        Returns:
            MultiModalContent with all generated content
        """
        if output_formats is None:
            output_formats = [ContentFormat.BLOG_POST, ContentFormat.SOCIAL_POST]
        
        # Transcribe
        analysis = await self.whisper.analyze_audio(
            audio_path=audio_path,
            llm_client=self.llm,
            include_summary=True,
            include_topics=True
        )
        
        content = MultiModalContent(
            primary_format=output_formats[0],
            text_content=analysis.transcription.text,
            audio_transcription=analysis.transcription.text,
            metadata={
                "source_audio": audio_path,
                "duration": analysis.transcription.duration,
                "summary": analysis.summary,
                "topics": analysis.key_topics
            }
        )
        
        # Generate additional formats
        if ContentFormat.BLOG_POST in output_formats:
            content.metadata["blog_post"] = await self._generate_blog_from_audio(analysis)
        
        if ContentFormat.SOCIAL_POST in output_formats:
            content.metadata["social_posts"] = await self._generate_social_from_audio(analysis)
        
        return content


# Export agent
multimodal_agent = MultiModalAgent()
