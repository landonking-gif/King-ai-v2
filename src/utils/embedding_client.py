"""
Embedding generation client for vector operations.
Supports multiple providers: Ollama, OpenAI, or Sentence Transformers.
"""

import httpx
import asyncio
from typing import List, Optional
from abc import ABC, abstractmethod
import numpy as np

from config.settings import settings
from src.utils.structured_logging import get_logger

logger = get_logger("embeddings")


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        pass
    
    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        pass


class OllamaEmbeddingProvider(EmbeddingProvider):
    """
    Embedding generation using Ollama's embedding models.
    Uses models like nomic-embed-text or mxbai-embed-large.
    """
    
    def __init__(
        self,
        base_url: str = None,
        model: str = "nomic-embed-text"
    ):
        self.base_url = (base_url or settings.ollama_url).rstrip("/")
        self.model = model
        self.client = httpx.AsyncClient(timeout=60.0)
        self._dimension = 768  # Default for nomic-embed-text
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    async def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text using Ollama."""
        try:
            response = await self.client.post(
                f"{self.base_url}/api/embeddings",
                json={
                    "model": self.model,
                    "prompt": text
                }
            )
            response.raise_for_status()
            data = response.json()
            return data["embedding"]
        except Exception as e:
            logger.error("Embedding generation failed", error=str(e))
            raise
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        # Ollama doesn't support batch embedding natively, so we parallelize
        tasks = [self.embed(text) for text in texts]
        return await asyncio.gather(*tasks)


class SentenceTransformerProvider(EmbeddingProvider):
    """
    Local embedding using Sentence Transformers.
    Fallback when Ollama embeddings aren't available.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._dimension = 384  # Default for MiniLM
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    def _get_model(self):
        """Lazy load the model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
                self._dimension = self._model.get_sentence_embedding_dimension()
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )
        return self._model
    
    async def embed(self, text: str) -> List[float]:
        """Generate embedding using Sentence Transformers."""
        model = self._get_model()
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None,
            lambda: model.encode(text, convert_to_numpy=True)
        )
        return embedding.tolist()
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        model = self._get_model()
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: model.encode(texts, convert_to_numpy=True)
        )
        return embeddings.tolist()


class EmbeddingClient:
    """
    Unified embedding client with provider fallback.
    Tries Ollama first, falls back to local Sentence Transformers.
    """
    
    def __init__(self):
        self.primary: Optional[EmbeddingProvider] = None
        self.fallback: Optional[EmbeddingProvider] = None
        
        # Try to initialize Ollama provider
        try:
            self.primary = OllamaEmbeddingProvider()
            logger.info("Initialized Ollama embedding provider")
        except Exception as e:
            logger.warning(f"Ollama embeddings unavailable: {e}")
        
        # Initialize fallback
        try:
            self.fallback = SentenceTransformerProvider()
            logger.info("Initialized Sentence Transformer fallback")
        except Exception as e:
            logger.warning(f"Sentence Transformers unavailable: {e}")
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension from active provider."""
        if self.primary:
            return self.primary.dimension
        if self.fallback:
            return self.fallback.dimension
        return 768  # Default
    
    async def embed(self, text: str) -> List[float]:
        """Generate embedding with fallback."""
        if self.primary:
            try:
                return await self.primary.embed(text)
            except Exception as e:
                logger.warning(f"Primary embedding failed: {e}")
        
        if self.fallback:
            return await self.fallback.embed(text)
        
        raise RuntimeError("No embedding providers available")
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate batch embeddings with fallback."""
        if self.primary:
            try:
                return await self.primary.embed_batch(texts)
            except Exception as e:
                logger.warning(f"Primary batch embedding failed: {e}")
        
        if self.fallback:
            return await self.fallback.embed_batch(texts)
        
        raise RuntimeError("No embedding providers available")
    
    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two embeddings."""
        a_np = np.array(a)
        b_np = np.array(b)
        return float(np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np)))


# Singleton instance
embedding_client = EmbeddingClient()
