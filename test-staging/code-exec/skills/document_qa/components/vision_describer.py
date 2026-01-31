"""
Vision Image Describer Component for Haystack 2.x.

Module: code-exec/skills/document_qa/components/vision_describer.py

Describes complex images using Vision LLM (GPT-4o, Claude 3, Gemini, etc.)
configured via environment. Uses smart filtering to reduce costs by ~95%.

Uses LLM adapters for provider-agnostic LLM calls where possible.
Vision currently requires OpenAI-compatible vision API or native provider API.
"""

import base64
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from haystack import component

# Add adapters to path for import
_repo_root = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from adapters.llm import get_default_model, get_default_provider, LLMProvider

from ..utils.image_filter import ImageData, ImageFilter

logger = logging.getLogger(__name__)


@dataclass
class ImageDescription:
    """Represents an image with its description from Vision LLM."""

    image: ImageData
    description: str
    source_id: Optional[str] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}


@component
class VisionImageDescriber:
    """
    Describe complex images using Vision LLM.

    Uses smart filtering to skip ~95% of images (icons, logos, decorative).
    Only sends complex diagrams and charts to Vision LLM.

    Configuration via environment:
    - OPENAI_API_KEY: API key for OpenAI
    - OPENAI_MODEL: Model to use (default: gpt-4o)
    - DOCUMENT_QA_MAX_IMAGES: Max images to process (default: 10)
    """

    VISION_PROMPT = """Describe this image in detail for document Q&A purposes.

Focus on:
1. Type of image (chart, diagram, photo, screenshot, etc.)
2. Key information displayed
3. Any text, labels, or annotations visible
4. Data points, trends, or relationships shown
5. Overall purpose and what it illustrates

Be concise but comprehensive. If this is a chart/graph, describe the data it represents."""

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        vision_model: Optional[str] = None,
        image_filter: Optional[ImageFilter] = None,
        max_concurrent: int = 5,
    ):
        """
        Initialize the Vision Image Describer.

        Args:
            llm_client: Optional LLM client with vision capability
            vision_model: Model to use (default from OPENAI_MODEL env)
            image_filter: Custom ImageFilter (default creates new one)
            max_concurrent: Max concurrent API calls
        """
        self.llm_client = llm_client
        self.vision_model = vision_model or os.getenv("OPENAI_MODEL", "gpt-4o")
        self.image_filter = image_filter or ImageFilter()
        self.max_concurrent = max_concurrent

        # Statistics
        self._processed_count = 0
        self._skipped_count = 0

    @component.output_types(descriptions=List[ImageDescription])
    def run(
        self,
        images: List[ImageData],
        custom_prompt: Optional[str] = None,
    ) -> Dict[str, List[ImageDescription]]:
        """
        Describe images using Vision LLM.

        Args:
            images: List of ImageData to process
            custom_prompt: Optional custom prompt for descriptions

        Returns:
            Dictionary with 'descriptions' key
        """
        # Filter images
        original_count = len(images)
        filtered_images = self.image_filter.filter_images(images)
        self._skipped_count = original_count - len(filtered_images)

        stats = self.image_filter.get_statistics(original_count, len(filtered_images))
        logger.info(
            f"Image filtering: {original_count} -> {len(filtered_images)} "
            f"({stats['reduction_percent']}% reduction)"
        )

        if not filtered_images:
            return {"descriptions": []}

        # Process images
        descriptions = []
        prompt = custom_prompt or self.VISION_PROMPT

        for image in filtered_images:
            try:
                description = self._describe_image(image, prompt)
                if description:
                    descriptions.append(description)
                    self._processed_count += 1

            except Exception as e:
                logger.error(f"Failed to describe image {image.file_path}: {e}")
                continue

        logger.info(f"Described {len(descriptions)} images")
        return {"descriptions": descriptions}

    def _describe_image(self, image: ImageData, prompt: str) -> Optional[ImageDescription]:
        """
        Describe a single image using Vision LLM.

        Args:
            image: ImageData to describe
            prompt: Prompt for the Vision LLM

        Returns:
            ImageDescription or None if failed
        """
        # Encode image to base64
        image_b64 = base64.b64encode(image.image_bytes).decode("utf-8")
        media_type = f"image/{image.image_format}"

        # Use provided LLM client or create OpenAI client
        if self.llm_client:
            return self._describe_with_client(image, image_b64, media_type, prompt)
        else:
            return self._describe_with_openai(image, image_b64, media_type, prompt)

    def _describe_with_client(
        self,
        image: ImageData,
        image_b64: str,
        media_type: str,
        prompt: str,
    ) -> Optional[ImageDescription]:
        """Use provided LLM client for description."""
        try:
            # Assume client has a vision-capable method
            if hasattr(self.llm_client, "describe_image"):
                description = self.llm_client.describe_image(
                    image_b64=image_b64,
                    media_type=media_type,
                    prompt=prompt,
                )
            elif hasattr(self.llm_client, "chat"):
                # Generic chat with vision message format
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{media_type};base64,{image_b64}"
                                },
                            },
                        ],
                    }
                ]
                response = self.llm_client.chat(messages)
                description = response.get("content", "") if isinstance(response, dict) else str(response)
            else:
                logger.warning("LLM client doesn't support vision")
                return None

            return ImageDescription(
                image=image,
                description=description,
                metadata={"model": self.vision_model},
            )

        except Exception as e:
            logger.error(f"LLM client vision call failed: {e}")
            return None

    def _describe_with_openai(
        self,
        image: ImageData,
        image_b64: str,
        media_type: str,
        prompt: str,
    ) -> Optional[ImageDescription]:
        """
        Use adapter-configured provider for vision description.

        Detects provider from .env and uses appropriate vision API.
        Supports: OpenAI (gpt-4o, etc.), Anthropic (claude-3+), Gemini.
        """
        provider = get_default_provider()
        model = self.vision_model or get_default_model(provider)

        logger.info(f"Vision describer using provider: {provider.value}, model: {model}")

        if provider == LLMProvider.OPENAI:
            return self._call_openai_vision(image, image_b64, media_type, prompt, model)
        elif provider == LLMProvider.ANTHROPIC:
            return self._call_anthropic_vision(image, image_b64, media_type, prompt, model)
        elif provider == LLMProvider.GEMINI:
            return self._call_gemini_vision(image, image_b64, media_type, prompt, model)
        else:
            logger.warning(f"Provider {provider.value} may not support vision, trying OpenAI format")
            return self._call_openai_vision(image, image_b64, media_type, prompt, model)

    def _call_openai_vision(
        self,
        image: ImageData,
        image_b64: str,
        media_type: str,
        prompt: str,
        model: str,
    ) -> Optional[ImageDescription]:
        """Call OpenAI Vision API."""
        try:
            from openai import OpenAI

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.error("OPENAI_API_KEY not set")
                return None

            client = OpenAI(api_key=api_key)

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{media_type};base64,{image_b64}",
                                    "detail": "auto",
                                },
                            },
                        ],
                    }
                ],
                max_tokens=500,
            )

            description = response.choices[0].message.content

            return ImageDescription(
                image=image,
                description=description,
                metadata={
                    "model": model,
                    "provider": "openai",
                    "tokens_used": response.usage.total_tokens if response.usage else 0,
                },
            )

        except ImportError:
            logger.error("OpenAI package not installed. Run: pip install openai")
            return None
        except Exception as e:
            logger.error(f"OpenAI vision call failed: {e}")
            return None

    def _call_anthropic_vision(
        self,
        image: ImageData,
        image_b64: str,
        media_type: str,
        prompt: str,
        model: str,
    ) -> Optional[ImageDescription]:
        """Call Anthropic Vision API (Claude 3+)."""
        try:
            import httpx

            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                logger.error("ANTHROPIC_API_KEY not set")
                return None

            # Anthropic uses different media type format
            anthropic_media_type = media_type.replace("image/", "")
            if anthropic_media_type == "jpg":
                anthropic_media_type = "jpeg"

            response = httpx.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": model,
                    "max_tokens": 500,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": f"image/{anthropic_media_type}",
                                        "data": image_b64,
                                    },
                                },
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ],
                },
                timeout=60.0,
            )
            response.raise_for_status()
            data = response.json()

            content_blocks = data.get("content", [])
            description = content_blocks[0].get("text", "") if content_blocks else ""

            return ImageDescription(
                image=image,
                description=description,
                metadata={
                    "model": model,
                    "provider": "anthropic",
                    "tokens_used": (
                        data.get("usage", {}).get("input_tokens", 0)
                        + data.get("usage", {}).get("output_tokens", 0)
                    ),
                },
            )

        except ImportError:
            logger.error("httpx package not installed. Run: pip install httpx")
            return None
        except Exception as e:
            logger.error(f"Anthropic vision call failed: {e}")
            return None

    def _call_gemini_vision(
        self,
        image: ImageData,
        image_b64: str,
        media_type: str,
        prompt: str,
        model: str,
    ) -> Optional[ImageDescription]:
        """Call Google Gemini Vision API."""
        try:
            import httpx

            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                logger.error("GEMINI_API_KEY not set")
                return None

            response = httpx.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                params={"key": api_key},
                headers={"content-type": "application/json"},
                json={
                    "contents": [
                        {
                            "parts": [
                                {"text": prompt},
                                {
                                    "inline_data": {
                                        "mime_type": media_type,
                                        "data": image_b64,
                                    }
                                },
                            ]
                        }
                    ]
                },
                timeout=60.0,
            )
            response.raise_for_status()
            data = response.json()

            candidates = data.get("candidates", [])
            description = ""
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    description = parts[0].get("text", "")

            return ImageDescription(
                image=image,
                description=description,
                metadata={
                    "model": model,
                    "provider": "gemini",
                },
            )

        except ImportError:
            logger.error("httpx package not installed. Run: pip install httpx")
            return None
        except Exception as e:
            logger.error(f"Gemini vision call failed: {e}")
            return None

    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            "processed": self._processed_count,
            "skipped": self._skipped_count,
            "model": self.vision_model,
        }


def describe_single_image(
    image_path: str,
    prompt: Optional[str] = None,
) -> Optional[str]:
    """
    Convenience function to describe a single image.

    Args:
        image_path: Path to image file
        prompt: Optional custom prompt

    Returns:
        Description string or None
    """
    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        from pathlib import Path

        ext = Path(image_path).suffix.lower().lstrip(".")

        image_data = ImageData(
            image_bytes=image_bytes,
            file_path=image_path,
            source_file=image_path,
            image_format=ext,
        )

        describer = VisionImageDescriber()
        # Bypass filter for single image
        describer.image_filter.min_size_kb = 0

        result = describer.run([image_data], custom_prompt=prompt)

        if result["descriptions"]:
            return result["descriptions"][0].description
        return None

    except Exception as e:
        logger.error(f"Failed to describe image: {e}")
        return None
