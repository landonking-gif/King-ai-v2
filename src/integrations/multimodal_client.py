"""
Multi-Modal AI Client.

Provides unified access to multi-modal AI capabilities:
- Whisper: Audio transcription and translation
- CLIP: Image similarity search and classification
- Vision models: Image analysis and understanding

Enables advanced business use cases like:
- Analyzing product images for quality
- Transcribing customer calls for insights
- Generating video ad scripts from audio
- Visual similarity search for products
"""

import base64
import os
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import httpx
import tempfile

from src.utils.structured_logging import get_logger
from src.utils.circuit_breaker import CircuitBreaker

logger = get_logger("multimodal_client")

# Circuit breakers for multi-modal services
whisper_circuit = CircuitBreaker("whisper", failure_threshold=3, timeout=60.0)
clip_circuit = CircuitBreaker("clip", failure_threshold=3, timeout=60.0)
vision_circuit = CircuitBreaker("vision", failure_threshold=3, timeout=60.0)


class AudioFormat(str, Enum):
    """Supported audio formats for Whisper."""
    MP3 = "mp3"
    MP4 = "mp4"
    WAV = "wav"
    WEBM = "webm"
    M4A = "m4a"
    OGG = "ogg"


class ImageFormat(str, Enum):
    """Supported image formats."""
    PNG = "png"
    JPEG = "jpeg"
    JPG = "jpg"
    WEBP = "webp"
    GIF = "gif"


@dataclass
class TranscriptionResult:
    """Result from audio transcription."""
    text: str
    language: Optional[str] = None
    duration_seconds: Optional[float] = None
    segments: Optional[List[Dict[str, Any]]] = None
    confidence: Optional[float] = None


@dataclass
class ImageAnalysisResult:
    """Result from image analysis."""
    description: str
    labels: List[str]
    objects: List[Dict[str, Any]]
    colors: List[str]
    confidence: float
    metadata: Dict[str, Any] = None


@dataclass
class SimilarityResult:
    """Result from similarity search."""
    score: float
    id: str
    metadata: Dict[str, Any] = None


class WhisperClient:
    """
    OpenAI Whisper API client for audio transcription.
    
    Supports:
    - Speech-to-text transcription
    - Language detection
    - Translation to English
    - Timestamp generation
    """
    
    API_URL = "https://api.openai.com/v1/audio"
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._configured = bool(self.api_key)
        
        if not self._configured:
            logger.warning("Whisper not configured - audio features disabled")
    
    @property
    def is_configured(self) -> bool:
        return self._configured
    
    @whisper_circuit.protect
    async def transcribe(
        self,
        audio_data: Union[bytes, str],
        language: str = None,
        prompt: str = None,
        response_format: str = "json",
        temperature: float = 0.0
    ) -> TranscriptionResult:
        """
        Transcribe audio to text.
        
        Args:
            audio_data: Audio bytes or file path
            language: Optional language code (e.g., 'en', 'es')
            prompt: Optional context prompt to guide transcription
            response_format: Output format (json, text, srt, verbose_json, vtt)
            temperature: Sampling temperature (0-1)
            
        Returns:
            TranscriptionResult with text and metadata
        """
        if not self._configured:
            raise RuntimeError("Whisper API key not configured")
        
        # Handle file path vs bytes
        if isinstance(audio_data, str):
            with open(audio_data, "rb") as f:
                audio_bytes = f.read()
            filename = os.path.basename(audio_data)
        else:
            audio_bytes = audio_data
            filename = "audio.mp3"
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            files = {"file": (filename, audio_bytes)}
            data = {
                "model": "whisper-1",
                "response_format": response_format,
                "temperature": temperature
            }
            
            if language:
                data["language"] = language
            if prompt:
                data["prompt"] = prompt
            
            response = await client.post(
                f"{self.API_URL}/transcriptions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                files=files,
                data=data
            )
            response.raise_for_status()
            
            result = response.json()
            
            logger.info(
                "Audio transcribed",
                text_length=len(result.get("text", "")),
                language=result.get("language")
            )
            
            return TranscriptionResult(
                text=result.get("text", ""),
                language=result.get("language"),
                duration_seconds=result.get("duration"),
                segments=result.get("segments"),
                confidence=None
            )
    
    @whisper_circuit.protect
    async def translate(
        self,
        audio_data: Union[bytes, str],
        prompt: str = None
    ) -> TranscriptionResult:
        """
        Translate audio to English text.
        
        Args:
            audio_data: Audio bytes or file path
            prompt: Optional context prompt
            
        Returns:
            TranscriptionResult with English translation
        """
        if not self._configured:
            raise RuntimeError("Whisper API key not configured")
        
        if isinstance(audio_data, str):
            with open(audio_data, "rb") as f:
                audio_bytes = f.read()
            filename = os.path.basename(audio_data)
        else:
            audio_bytes = audio_data
            filename = "audio.mp3"
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            files = {"file": (filename, audio_bytes)}
            data = {"model": "whisper-1"}
            
            if prompt:
                data["prompt"] = prompt
            
            response = await client.post(
                f"{self.API_URL}/translations",
                headers={"Authorization": f"Bearer {self.api_key}"},
                files=files,
                data=data
            )
            response.raise_for_status()
            
            result = response.json()
            
            return TranscriptionResult(
                text=result.get("text", ""),
                language="en"
            )


class CLIPClient:
    """
    CLIP (Contrastive Language-Image Pre-training) client.
    
    Uses OpenAI's CLIP model for:
    - Image-text similarity scoring
    - Zero-shot image classification
    - Image search by text description
    
    Can use local CLIP or cloud APIs.
    """
    
    def __init__(self, model_name: str = "ViT-B/32"):
        self.model_name = model_name
        self._model = None
        self._preprocess = None
        self._device = "cpu"
        self._initialized = False
    
    async def _ensure_initialized(self):
        """Lazy initialization of CLIP model."""
        if self._initialized:
            return
        
        try:
            import torch
            import clip
            
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model, self._preprocess = clip.load(self.model_name, device=self._device)
            self._initialized = True
            
            logger.info(f"CLIP initialized on {self._device}")
        except ImportError:
            logger.warning("CLIP not available - install with: pip install git+https://github.com/openai/CLIP.git")
            raise RuntimeError("CLIP not installed")
    
    @clip_circuit.protect
    async def compute_similarity(
        self,
        image_data: Union[bytes, str],
        text_queries: List[str]
    ) -> List[Tuple[str, float]]:
        """
        Compute similarity between an image and text queries.
        
        Args:
            image_data: Image bytes or file path
            text_queries: List of text descriptions to compare
            
        Returns:
            List of (query, similarity_score) tuples, sorted by score
        """
        await self._ensure_initialized()
        
        import torch
        from PIL import Image
        import clip
        import io
        
        # Load image
        if isinstance(image_data, str):
            image = Image.open(image_data)
        else:
            image = Image.open(io.BytesIO(image_data))
        
        # Preprocess
        image_input = self._preprocess(image).unsqueeze(0).to(self._device)
        text_inputs = clip.tokenize(text_queries).to(self._device)
        
        # Compute features
        with torch.no_grad():
            image_features = self._model.encode_image(image_input)
            text_features = self._model.encode_text(text_inputs)
            
            # Normalize
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            # Compute similarity
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        
        scores = similarity[0].cpu().numpy()
        results = [(query, float(score)) for query, score in zip(text_queries, scores)]
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    @clip_circuit.protect
    async def classify_image(
        self,
        image_data: Union[bytes, str],
        categories: List[str]
    ) -> Dict[str, float]:
        """
        Zero-shot image classification.
        
        Args:
            image_data: Image bytes or file path
            categories: List of category labels
            
        Returns:
            Dict mapping category to confidence score
        """
        results = await self.compute_similarity(image_data, categories)
        return {category: score for category, score in results}
    
    @clip_circuit.protect
    async def get_embedding(
        self,
        image_data: Union[bytes, str]
    ) -> List[float]:
        """
        Get CLIP embedding for an image (for similarity search).
        
        Args:
            image_data: Image bytes or file path
            
        Returns:
            Embedding vector as list of floats
        """
        await self._ensure_initialized()
        
        import torch
        from PIL import Image
        import io
        
        if isinstance(image_data, str):
            image = Image.open(image_data)
        else:
            image = Image.open(io.BytesIO(image_data))
        
        image_input = self._preprocess(image).unsqueeze(0).to(self._device)
        
        with torch.no_grad():
            features = self._model.encode_image(image_input)
            features /= features.norm(dim=-1, keepdim=True)
        
        return features[0].cpu().numpy().tolist()


class VisionClient:
    """
    Vision AI client for image analysis.
    
    Uses GPT-4 Vision or similar models for:
    - Detailed image description
    - Object detection
    - Text extraction (OCR)
    - Visual question answering
    """
    
    API_URL = "https://api.openai.com/v1/chat/completions"
    
    def __init__(self, api_key: str = None, model: str = "gpt-4o"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self._configured = bool(self.api_key)
    
    @property
    def is_configured(self) -> bool:
        return self._configured
    
    def _encode_image(self, image_data: Union[bytes, str]) -> str:
        """Encode image to base64."""
        if isinstance(image_data, str):
            with open(image_data, "rb") as f:
                image_bytes = f.read()
        else:
            image_bytes = image_data
        
        return base64.b64encode(image_bytes).decode("utf-8")
    
    @vision_circuit.protect
    async def analyze_image(
        self,
        image_data: Union[bytes, str],
        prompt: str = "Describe this image in detail, including any text, objects, colors, and notable features."
    ) -> ImageAnalysisResult:
        """
        Analyze an image using vision AI.
        
        Args:
            image_data: Image bytes or file path
            prompt: Analysis prompt
            
        Returns:
            ImageAnalysisResult with description and metadata
        """
        if not self._configured:
            raise RuntimeError("Vision API key not configured")
        
        base64_image = self._encode_image(image_data)
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                self.API_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    "max_tokens": 1000
                }
            )
            response.raise_for_status()
            
            result = response.json()
            description = result["choices"][0]["message"]["content"]
            
            logger.info("Image analyzed", description_length=len(description))
            
            return ImageAnalysisResult(
                description=description,
                labels=[],
                objects=[],
                colors=[],
                confidence=1.0
            )
    
    @vision_circuit.protect
    async def extract_text(
        self,
        image_data: Union[bytes, str]
    ) -> str:
        """
        Extract text from an image (OCR).
        
        Args:
            image_data: Image bytes or file path
            
        Returns:
            Extracted text
        """
        result = await self.analyze_image(
            image_data,
            prompt="Extract all text visible in this image. Return only the extracted text, preserving the layout as much as possible."
        )
        return result.description
    
    @vision_circuit.protect
    async def analyze_product(
        self,
        image_data: Union[bytes, str]
    ) -> Dict[str, Any]:
        """
        Analyze a product image for e-commerce.
        
        Args:
            image_data: Product image
            
        Returns:
            Product analysis including quality, features, suggested categories
        """
        result = await self.analyze_image(
            image_data,
            prompt="""Analyze this product image for e-commerce. Provide:
1. Product type/category
2. Key features visible
3. Quality assessment (1-10)
4. Suggested product title
5. Suggested keywords/tags
6. Any issues with the image (lighting, angle, background)

Format as JSON."""
        )
        
        # Try to parse JSON from response
        import json
        try:
            # Find JSON in response
            text = result.description
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
        
        return {"raw_analysis": result.description}


class MultiModalClient:
    """
    Unified multi-modal AI client.
    
    Provides a single interface for all multi-modal capabilities:
    - Audio transcription (Whisper)
    - Image similarity (CLIP)
    - Image analysis (Vision)
    """
    
    def __init__(self):
        self.whisper = WhisperClient()
        self.clip = CLIPClient()
        self.vision = VisionClient()
    
    async def transcribe_audio(
        self,
        audio_data: Union[bytes, str],
        **kwargs
    ) -> TranscriptionResult:
        """Transcribe audio to text."""
        return await self.whisper.transcribe(audio_data, **kwargs)
    
    async def translate_audio(
        self,
        audio_data: Union[bytes, str],
        **kwargs
    ) -> TranscriptionResult:
        """Translate audio to English."""
        return await self.whisper.translate(audio_data, **kwargs)
    
    async def analyze_image(
        self,
        image_data: Union[bytes, str],
        **kwargs
    ) -> ImageAnalysisResult:
        """Analyze an image."""
        return await self.vision.analyze_image(image_data, **kwargs)
    
    async def classify_image(
        self,
        image_data: Union[bytes, str],
        categories: List[str]
    ) -> Dict[str, float]:
        """Classify an image into categories."""
        return await self.clip.classify_image(image_data, categories)
    
    async def extract_text_from_image(
        self,
        image_data: Union[bytes, str]
    ) -> str:
        """Extract text from an image (OCR)."""
        return await self.vision.extract_text(image_data)
    
    async def analyze_product_image(
        self,
        image_data: Union[bytes, str]
    ) -> Dict[str, Any]:
        """Analyze a product image for e-commerce."""
        return await self.vision.analyze_product(image_data)
    
    async def get_image_embedding(
        self,
        image_data: Union[bytes, str]
    ) -> List[float]:
        """Get embedding for image similarity search."""
        return await self.clip.get_embedding(image_data)
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all multi-modal services."""
        return {
            "whisper": {
                "configured": self.whisper.is_configured,
                "circuit_state": whisper_circuit.state.value
            },
            "clip": {
                "initialized": self.clip._initialized,
                "circuit_state": clip_circuit.state.value
            },
            "vision": {
                "configured": self.vision.is_configured,
                "circuit_state": vision_circuit.state.value
            }
        }


# Global instance
multimodal_client = MultiModalClient()
