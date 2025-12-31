# FILE: src/integrations/dalle_client.py (CREATE NEW FILE)
"""
DALL-E Client - Image generation for content creation.
"""

import os
import httpx
from typing import Optional, List
from dataclasses import dataclass
from enum import Enum

from src.utils.logging import get_logger

logger = get_logger("dalle")


class ImageSize(str, Enum):
    """Available image sizes."""
    SMALL = "256x256"
    MEDIUM = "512x512"
    LARGE = "1024x1024"
    WIDE = "1792x1024"
    TALL = "1024x1792"


class ImageQuality(str, Enum):
    """Image quality options."""
    STANDARD = "standard"
    HD = "hd"


@dataclass
class GeneratedImage:
    """A generated image result."""
    url: str
    revised_prompt: str
    size: str
    created_at: str


class DALLEClient:
    """
    OpenAI DALL-E API client for image generation.
    
    Used by the Content Agent for:
    - Product images
    - Blog post illustrations
    - Marketing materials
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize DALL-E client.
        
        Args:
            api_key: OpenAI API key (defaults to env var)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = "https://api.openai.com/v1"
        self.model = "dall-e-3"
        
        if not self.api_key:
            logger.warning("OpenAI API key not configured")
    
    async def generate_image(
        self,
        prompt: str,
        size: ImageSize = ImageSize.LARGE,
        quality: ImageQuality = ImageQuality.STANDARD,
        style: str = "vivid",
        n: int = 1
    ) -> List[GeneratedImage]:
        """
        Generate images from a text prompt.
        
        Args:
            prompt: Text description of the image
            size: Image dimensions
            quality: Standard or HD
            style: "vivid" or "natural"
            n: Number of images (DALL-E 3 only supports 1)
            
        Returns:
            List of generated images
        """
        if not self.api_key:
            raise ValueError("OpenAI API key not configured")
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/images/generations",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "size": size.value,
                    "quality": quality.value,
                    "style": style,
                    "n": 1  # DALL-E 3 limitation
                },
                timeout=60.0
            )
            
            response.raise_for_status()
            data = response.json()
            
            images = []
            for item in data.get("data", []):
                images.append(GeneratedImage(
                    url=item["url"],
                    revised_prompt=item.get("revised_prompt", prompt),
                    size=size.value,
                    created_at=str(data.get("created", ""))
                ))
            
            logger.info(f"Generated {len(images)} image(s)")
            return images
    
    async def generate_product_image(
        self,
        product_name: str,
        product_description: str,
        style: str = "product photography"
    ) -> GeneratedImage:
        """
        Generate a product image for e-commerce.
        
        Args:
            product_name: Name of the product
            product_description: Description of the product
            style: Photography style
            
        Returns:
            Generated product image
        """
        prompt = f"""
        Professional {style} of {product_name}.
        {product_description}
        Clean white background, studio lighting, high quality product shot.
        E-commerce ready, centered composition.
        """
        
        images = await self.generate_image(
            prompt=prompt.strip(),
            size=ImageSize.LARGE,
            quality=ImageQuality.HD,
            style="natural"
        )
        
        return images[0] if images else None
    
    async def generate_blog_illustration(
        self,
        topic: str,
        style: str = "modern digital illustration"
    ) -> GeneratedImage:
        """
        Generate a blog post illustration.
        
        Args:
            topic: Blog post topic
            style: Illustration style
            
        Returns:
            Generated illustration
        """
        prompt = f"""
        {style} representing the concept of: {topic}
        Professional, clean design suitable for a business blog.
        Abstract and modern aesthetic.
        """
        
        images = await self.generate_image(
            prompt=prompt.strip(),
            size=ImageSize.WIDE,
            quality=ImageQuality.STANDARD,
            style="vivid"
        )
        
        return images[0] if images else None


# Singleton
dalle_client = DALLEClient()