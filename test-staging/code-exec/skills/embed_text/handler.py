"""
Text embedding skill handler.
Module: code-exec/skills/embed_text/handler.py
"""

from typing import Any, Dict, List

# Global model cache to avoid reloading
_MODEL_CACHE: Dict[str, Any] = {}


def embed_text(text: str, model: str = "all-MiniLM-L6-v2", normalize: bool = True) -> Dict[str, Any]:
    """
    Generate embeddings for text using sentence-transformers.

    Args:
        text: Text to embed
        model: Model to use (one of: all-MiniLM-L6-v2, all-mpnet-base-v2, paraphrase-MiniLM-L6-v2)
        normalize: Whether to normalize embeddings to unit length

    Returns:
        Dictionary with embedding vector and metadata
    """
    # Map short names to full model names
    model_map = {
        "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
        "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",
        "paraphrase-MiniLM-L6-v2": "sentence-transformers/paraphrase-MiniLM-L6-v2",
    }

    full_model_name = model_map.get(model, model_map["all-MiniLM-L6-v2"])

    # Load model (cached)
    embedder = _load_model(full_model_name)

    # Generate embedding
    embedding = embedder.encode(text, normalize_embeddings=normalize)

    # Convert to list (from numpy array)
    embedding_list: List[float] = embedding.tolist()

    return {
        "embedding": embedding_list,
        "dimension": len(embedding_list),
        "model_used": model,
        "text_length": len(text),
        "normalized": normalize,
    }


def _load_model(model_name: str) -> Any:
    """
    Load sentence-transformer model with caching.

    Args:
        model_name: Full model name

    Returns:
        SentenceTransformer model
    """
    if model_name not in _MODEL_CACHE:
        try:
            from sentence_transformers import SentenceTransformer

            _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
        except ImportError:
            # Fallback: return a mock embedder for testing without sentence-transformers
            import hashlib
            import numpy as np

            class MockEmbedder:
                """Mock embedder for testing without sentence-transformers."""

                def encode(self, text: str, normalize_embeddings: bool = True) -> Any:
                    """Generate deterministic pseudo-embedding from text hash."""
                    # Create deterministic embedding based on text hash
                    text_hash = hashlib.sha256(text.encode()).digest()
                    # Convert to 384-dim vector (same as all-MiniLM-L6-v2)
                    values = np.frombuffer(text_hash, dtype=np.uint8).astype(np.float32)
                    # Repeat and truncate to 384 dimensions
                    embedding = np.tile(values, (384 // len(values)) + 1)[:384]

                    if normalize_embeddings:
                        # Normalize to unit length
                        norm = np.linalg.norm(embedding)
                        if norm > 0:
                            embedding = embedding / norm

                    return embedding

            _MODEL_CACHE[model_name] = MockEmbedder()

    return _MODEL_CACHE[model_name]
