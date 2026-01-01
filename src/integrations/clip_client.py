"""
CLIP Image Analysis Client.

Provides image understanding and similarity search using OpenAI's CLIP model.
Enables multi-modal processing for product matching, visual search, and content analysis.
"""

import base64
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import httpx
import numpy as np

from config.settings import settings
from src.utils.structured_logging import get_logger
from src.utils.circuit_breaker import CircuitBreaker

logger = get_logger("clip_client")

# CLIP circuit breaker
clip_circuit = CircuitBreaker(
    "clip",
    failure_threshold=3,
    timeout=60.0,
    success_threshold=2
)


@dataclass
class ImageEmbedding:
    """Embedding vector for an image."""
    image_path: str
    embedding: List[float]
    dimensions: int
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass 
class SimilarityResult:
    """Result of image similarity comparison."""
    query_image: str
    matches: List[Tuple[str, float]]  # (image_path, similarity_score)
    top_match: Optional[str] = None
    top_score: Optional[float] = None
    
    def __post_init__(self):
        if self.matches:
            self.top_match = self.matches[0][0]
            self.top_score = self.matches[0][1]


@dataclass
class ImageAnalysis:
    """Analysis results for an image."""
    image_path: str
    description: str
    labels: List[str]
    colors: List[str]
    objects: List[str]
    text_detected: Optional[str] = None
    suitable_for: List[str] = None  # e.g., ["product photo", "social media"]
    quality_score: Optional[float] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.suitable_for is None:
            self.suitable_for = []
        if self.metadata is None:
            self.metadata = {}


class CLIPClient:
    """
    CLIP-based image analysis client using OpenAI Vision API.
    
    Features:
    - Image embedding generation
    - Image similarity search
    - Zero-shot image classification
    - Product image analysis
    - Visual content understanding
    
    Note: Uses OpenAI's Vision API which incorporates CLIP-like capabilities.
    For true CLIP embeddings, would need to run CLIP locally or use a dedicated service.
    """
    
    API_URL = "https://api.openai.com/v1"
    MAX_IMAGE_SIZE = 20 * 1024 * 1024  # 20 MB
    SUPPORTED_FORMATS = [".jpg", ".jpeg", ".png", ".gif", ".webp"]
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or settings.openai_api_key
        self._configured = bool(self.api_key)
        self._embedding_cache: Dict[str, List[float]] = {}
        
        if not self._configured:
            logger.warning("CLIP client not configured - OPENAI_API_KEY not set")
    
    @property
    def is_configured(self) -> bool:
        return self._configured
    
    def _validate_image(self, image_path: Path) -> None:
        """Validate image file before processing."""
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        if image_path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format: {image_path.suffix}. "
                f"Supported: {', '.join(self.SUPPORTED_FORMATS)}"
            )
        
        file_size = image_path.stat().st_size
        if file_size > self.MAX_IMAGE_SIZE:
            raise ValueError(
                f"Image too large: {file_size / 1024 / 1024:.1f}MB. "
                f"Maximum: {self.MAX_IMAGE_SIZE / 1024 / 1024}MB"
            )
    
    def _encode_image(self, image_path: Path) -> str:
        """Encode image to base64."""
        with open(image_path, "rb") as f:
            return base64.standard_b64encode(f.read()).decode("utf-8")
    
    def _get_media_type(self, image_path: Path) -> str:
        """Get media type for image."""
        suffix = image_path.suffix.lower()
        media_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp"
        }
        return media_types.get(suffix, "image/jpeg")
    
    @clip_circuit.protect
    async def analyze_image(
        self,
        image_path: str,
        prompt: str = None,
        detail: str = "auto"
    ) -> ImageAnalysis:
        """
        Analyze an image using GPT-4 Vision.
        
        Args:
            image_path: Path to the image file or URL
            prompt: Custom analysis prompt
            detail: Detail level ("low", "high", "auto")
            
        Returns:
            ImageAnalysis with description and extracted information
        """
        if not self._configured:
            raise RuntimeError("CLIP client not configured")
        
        # Handle URL or file path
        if image_path.startswith("http"):
            image_content = {"type": "image_url", "image_url": {"url": image_path, "detail": detail}}
        else:
            path = Path(image_path)
            self._validate_image(path)
            base64_image = self._encode_image(path)
            media_type = self._get_media_type(path)
            image_content = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{media_type};base64,{base64_image}",
                    "detail": detail
                }
            }
        
        analysis_prompt = prompt or """Analyze this image and provide:
1. A brief description (1-2 sentences)
2. Main objects or subjects visible
3. Dominant colors
4. Any text visible in the image
5. What this image would be suitable for (e.g., product photo, social media, marketing)
6. Image quality assessment (1-10)

Format your response as JSON with keys: description, objects, colors, text_detected, suitable_for, quality_score"""
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.API_URL}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "gpt-4o",
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": analysis_prompt},
                                    image_content
                                ]
                            }
                        ],
                        "max_tokens": 1000
                    }
                )
                response.raise_for_status()
                
                result = response.json()
                content = result["choices"][0]["message"]["content"]
            
            # Parse JSON response
            import json
            import re
            
            # Extract JSON from response
            json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                # Fallback parsing
                data = {
                    "description": content[:200],
                    "objects": [],
                    "colors": [],
                    "text_detected": None,
                    "suitable_for": [],
                    "quality_score": None
                }
            
            analysis = ImageAnalysis(
                image_path=image_path,
                description=data.get("description", ""),
                labels=[],  # Would need separate classification
                colors=data.get("colors", []),
                objects=data.get("objects", []),
                text_detected=data.get("text_detected"),
                suitable_for=data.get("suitable_for", []),
                quality_score=data.get("quality_score"),
                metadata={
                    "timestamp": datetime.utcnow().isoformat(),
                    "model": "gpt-4o"
                }
            )
            
            logger.info(
                "Image analyzed successfully",
                image=image_path,
                objects_found=len(analysis.objects)
            )
            
            return analysis
            
        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            raise
    
    @clip_circuit.protect
    async def classify_image(
        self,
        image_path: str,
        categories: List[str],
        return_scores: bool = True
    ) -> Dict[str, float]:
        """
        Zero-shot classify an image into provided categories.
        
        Args:
            image_path: Path to image file or URL
            categories: List of category labels
            return_scores: Whether to return confidence scores
            
        Returns:
            Dictionary mapping categories to confidence scores
        """
        if not self._configured:
            raise RuntimeError("CLIP client not configured")
        
        # Handle URL or file path
        if image_path.startswith("http"):
            image_content = {"type": "image_url", "image_url": {"url": image_path}}
        else:
            path = Path(image_path)
            self._validate_image(path)
            base64_image = self._encode_image(path)
            media_type = self._get_media_type(path)
            image_content = {
                "type": "image_url",
                "image_url": {"url": f"data:{media_type};base64,{base64_image}"}
            }
        
        categories_str = ", ".join(categories)
        prompt = f"""Classify this image into one or more of these categories: {categories_str}

For each applicable category, provide a confidence score from 0.0 to 1.0.
Return ONLY a JSON object with category names as keys and scores as values.
Only include categories with score > 0.1."""
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.API_URL}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "gpt-4o",
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    image_content
                                ]
                            }
                        ],
                        "max_tokens": 500
                    }
                )
                response.raise_for_status()
                
                result = response.json()
                content = result["choices"][0]["message"]["content"]
            
            # Parse JSON response
            import json
            import re
            
            json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
            if json_match:
                scores = json.loads(json_match.group())
            else:
                scores = {}
            
            # Normalize and filter
            scores = {k: float(v) for k, v in scores.items() if k in categories}
            
            logger.info(
                "Image classified",
                image=image_path,
                top_category=max(scores, key=scores.get) if scores else None
            )
            
            return scores
            
        except Exception as e:
            logger.error(f"Image classification failed: {e}")
            raise
    
    @clip_circuit.protect
    async def compare_images(
        self,
        image1_path: str,
        image2_path: str
    ) -> float:
        """
        Compare two images and return a similarity score.
        
        Args:
            image1_path: Path to first image
            image2_path: Path to second image
            
        Returns:
            Similarity score from 0.0 to 1.0
        """
        if not self._configured:
            raise RuntimeError("CLIP client not configured")
        
        # Prepare both images
        images = []
        for img_path in [image1_path, image2_path]:
            if img_path.startswith("http"):
                images.append({"type": "image_url", "image_url": {"url": img_path}})
            else:
                path = Path(img_path)
                self._validate_image(path)
                base64_image = self._encode_image(path)
                media_type = self._get_media_type(path)
                images.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{media_type};base64,{base64_image}"}
                })
        
        prompt = """Compare these two images and rate their similarity on a scale from 0.0 to 1.0, where:
- 1.0 = identical or nearly identical images
- 0.8-0.9 = same subject/product from different angles
- 0.5-0.7 = similar category or style
- 0.2-0.4 = loosely related
- 0.0-0.1 = completely different

Return ONLY a JSON object: {"similarity": <score>, "reason": "<brief explanation>"}"""
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.API_URL}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "gpt-4o",
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    images[0],
                                    images[1]
                                ]
                            }
                        ],
                        "max_tokens": 200
                    }
                )
                response.raise_for_status()
                
                result = response.json()
                content = result["choices"][0]["message"]["content"]
            
            # Parse response
            import json
            import re
            
            json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                similarity = float(data.get("similarity", 0.5))
            else:
                similarity = 0.5
            
            logger.info(
                "Images compared",
                similarity=similarity
            )
            
            return max(0.0, min(1.0, similarity))
            
        except Exception as e:
            logger.error(f"Image comparison failed: {e}")
            raise
    
    async def find_similar_products(
        self,
        query_image: str,
        product_images: List[str],
        top_k: int = 5
    ) -> SimilarityResult:
        """
        Find products most similar to a query image.
        
        Args:
            query_image: Path to query image
            product_images: List of product image paths
            top_k: Number of top matches to return
            
        Returns:
            SimilarityResult with ranked matches
        """
        if not product_images:
            return SimilarityResult(query_image=query_image, matches=[])
        
        # Compare query to each product
        similarities = []
        for product_path in product_images:
            try:
                score = await self.compare_images(query_image, product_path)
                similarities.append((product_path, score))
            except Exception as e:
                logger.warning(f"Comparison failed for {product_path}: {e}")
                continue
        
        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return SimilarityResult(
            query_image=query_image,
            matches=similarities[:top_k]
        )


# Global client instance
clip_client = CLIPClient()
