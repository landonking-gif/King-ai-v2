"""
Indexing Pipeline for Document Q&A.

Module: code-exec/skills/document_qa/pipelines/indexing.py

Haystack 2.x pipeline for document indexing:
1. Embedding generation (OpenAI or sentence-transformers)
2. Document store writing (in-memory or persistent)

Configuration:
- OPENAI_EMBEDDING_MODEL: If set, uses OpenAI embeddings (e.g., text-embedding-3-large)
- Otherwise falls back to sentence-transformers
"""

import logging
import os
from typing import Any, Dict, List, Optional

from haystack import Document, Pipeline
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy

logger = logging.getLogger(__name__)


def _get_embedding_model() -> tuple[str, str]:
    """
    Get embedding model configuration from environment.

    Returns:
        Tuple of (model_name, provider) where provider is 'openai' or 'sentence-transformers'
    """
    openai_model = os.getenv("OPENAI_EMBEDDING_MODEL")
    if openai_model:
        return openai_model, "openai"
    return "sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers"


def create_indexing_pipeline(
    document_store: Optional[InMemoryDocumentStore] = None,
    embedding_model: Optional[str] = None,
    duplicate_policy: DuplicatePolicy = DuplicatePolicy.OVERWRITE,
) -> tuple[Pipeline, InMemoryDocumentStore]:
    """
    Create the document indexing pipeline.

    Pipeline flow:
    documents -> Embedder (OpenAI or SentenceTransformers) -> DocumentWriter -> document_store

    Args:
        document_store: Optional document store (creates InMemory if not provided)
        embedding_model: Embedding model (auto-detected from env if not provided)
        duplicate_policy: How to handle duplicate documents

    Returns:
        Tuple of (Pipeline, DocumentStore)
    """
    # Create or use provided document store
    if document_store is None:
        document_store = InMemoryDocumentStore()

    pipeline = Pipeline()

    # Get embedding configuration from environment or use provided
    if embedding_model is None:
        model_name, provider = _get_embedding_model()
    else:
        # Detect provider from model name
        if "text-embedding" in embedding_model or embedding_model.startswith("openai"):
            model_name, provider = embedding_model, "openai"
        else:
            model_name, provider = embedding_model, "sentence-transformers"

    logger.info(f"Using embedding model: {model_name} (provider: {provider})")

    # Add embedding component based on provider
    if provider == "openai":
        try:
            from haystack.components.embedders import OpenAIDocumentEmbedder

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not set for OpenAI embeddings")

            pipeline.add_component(
                "embedder",
                OpenAIDocumentEmbedder(model=model_name),
            )
            logger.info(f"Using OpenAI embeddings: {model_name}")
        except ImportError:
            logger.warning("OpenAI embedder not available, falling back to sentence-transformers")
            provider = "sentence-transformers"
            model_name = "sentence-transformers/all-MiniLM-L6-v2"

    if provider == "sentence-transformers":
        try:
            from haystack.components.embedders import SentenceTransformersDocumentEmbedder

            pipeline.add_component(
                "embedder",
                SentenceTransformersDocumentEmbedder(model=model_name),
            )
            logger.info(f"Using SentenceTransformers embeddings: {model_name}")
        except ImportError:
            logger.warning(
                "sentence-transformers not available, using fallback embedder"
            )
            # Fallback: use a simple hash-based pseudo-embedder for testing
            pipeline.add_component("embedder", SimpleEmbedder())

    # Add document writer
    pipeline.add_component(
        "writer",
        DocumentWriter(
            document_store=document_store,
            policy=duplicate_policy,
        ),
    )

    # Connect components
    pipeline.connect("embedder", "writer")

    return pipeline, document_store


def run_indexing(
    documents: List[Document],
    document_store: Optional[InMemoryDocumentStore] = None,
    embedding_model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the indexing pipeline on documents.

    Args:
        documents: List of Haystack Documents to index
        document_store: Optional document store
        embedding_model: Embedding model to use

    Returns:
        Dictionary with 'document_store', 'documents_written', and 'statistics'
    """
    if not documents:
        return {
            "document_store": document_store or InMemoryDocumentStore(),
            "documents_written": 0,
            "statistics": {"total_documents": 0},
        }

    pipeline, store = create_indexing_pipeline(
        document_store=document_store,
        embedding_model=embedding_model,
    )

    # Run pipeline
    result = pipeline.run({"embedder": {"documents": documents}})

    documents_written = result.get("writer", {}).get("documents_written", 0)

    statistics = {
        "total_documents": len(documents),
        "documents_written": documents_written,
        "embedding_model": embedding_model,
        "store_type": type(store).__name__,
    }

    logger.info(f"Indexing complete: {documents_written} documents indexed")

    return {
        "document_store": store,
        "documents_written": documents_written,
        "statistics": statistics,
    }


class SimpleEmbedder:
    """
    Fallback embedder when sentence-transformers is not available.

    Uses simple hash-based vectors for testing/development only.
    NOT suitable for production - always install sentence-transformers.
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        logger.warning(
            "Using SimpleEmbedder fallback. Install sentence-transformers for proper embeddings."
        )

    def run(self, documents: List[Document]) -> Dict[str, List[Document]]:
        """Generate pseudo-embeddings for documents."""
        import hashlib

        for doc in documents:
            # Generate deterministic pseudo-embedding from content hash
            content_hash = hashlib.md5(doc.content.encode()).hexdigest()
            embedding = []

            for i in range(self.dimension):
                # Use hash bytes to generate float values
                byte_idx = i % 16
                value = int(content_hash[byte_idx * 2 : byte_idx * 2 + 2], 16)
                normalized = (value / 255.0) * 2 - 1  # Range [-1, 1]
                embedding.append(normalized)

            doc.embedding = embedding

        return {"documents": documents}


def create_query_embedding(
    query: str,
    embedding_model: Optional[str] = None,
) -> List[float]:
    """
    Create embedding for a query string.

    Uses OpenAI embeddings if OPENAI_EMBEDDING_MODEL is set, otherwise uses sentence-transformers.

    Args:
        query: Query text
        embedding_model: Embedding model to use (auto-detected from env if not provided)

    Returns:
        Embedding vector as list of floats
    """
    # Get embedding configuration from environment or use provided
    if embedding_model is None:
        model_name, provider = _get_embedding_model()
    else:
        if "text-embedding" in embedding_model or embedding_model.startswith("openai"):
            model_name, provider = embedding_model, "openai"
        else:
            model_name, provider = embedding_model, "sentence-transformers"

    logger.info(f"Creating query embedding with: {model_name} (provider: {provider})")

    if provider == "openai":
        try:
            from haystack.components.embedders import OpenAITextEmbedder

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not set for OpenAI embeddings")

            embedder = OpenAITextEmbedder(model=model_name)
            result = embedder.run(text=query)
            return result["embedding"]
        except ImportError:
            logger.warning("OpenAI embedder not available, falling back to sentence-transformers")
            provider = "sentence-transformers"
            model_name = "sentence-transformers/all-MiniLM-L6-v2"

    if provider == "sentence-transformers":
        try:
            from haystack.components.embedders import SentenceTransformersTextEmbedder

            embedder = SentenceTransformersTextEmbedder(model=model_name)
            embedder.warm_up()
            result = embedder.run(text=query)
            return result["embedding"]
        except ImportError:
            logger.warning("Using fallback query embedding")
            import hashlib

            content_hash = hashlib.md5(query.encode()).hexdigest()
            embedding = []
            for i in range(384):
                byte_idx = i % 16
                value = int(content_hash[byte_idx * 2 : byte_idx * 2 + 2], 16)
                normalized = (value / 255.0) * 2 - 1
                embedding.append(normalized)
            return embedding

    # Default fallback
    logger.warning("Using hash-based fallback embedding")
    import hashlib
    content_hash = hashlib.md5(query.encode()).hexdigest()
    embedding = []
    for i in range(384):
        byte_idx = i % 16
        value = int(content_hash[byte_idx * 2 : byte_idx * 2 + 2], 16)
        normalized = (value / 255.0) * 2 - 1
        embedding.append(normalized)
    return embedding
