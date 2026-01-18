"""
Document Q&A Skill Handler.

Module: code-exec/skills/document_qa/handler.py

Provides deterministic implementation for comprehensive document Q&A:
1. Extract content from documents (PDF, DOCX, XLSX, PPTX)
2. Process images with Vision LLM
3. Chunk semantically
4. Index with embeddings
5. Retrieve with hybrid search (Vector + BM25)
6. Rerank with LLM-as-judge
7. Assemble context strategically
8. Generate answer with source citations

Configuration via environment variables:
- OPENAI_API_KEY: API key for OpenAI LLM
- OPENAI_MODEL: Model to use (default: gpt-4o-mini)
- DOCUMENT_QA_MAX_IMAGES: Max images for Vision LLM (default: 10)
- DOCUMENT_QA_CHUNK_SIZE: Chunk size in tokens (default: 512)
"""

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from haystack.document_stores.in_memory import InMemoryDocumentStore

from .components.vision_describer import ImageDescription
from .pipelines.extraction import run_extraction
from .pipelines.generation import run_generation
from .pipelines.indexing import run_indexing
from .utils.source_tracker import SourceTracker

logger = logging.getLogger(__name__)


def document_qa(
    documents: List[str],
    query: str,
    include_sources: bool = True,
    max_context_tokens: int = 6000,
    rerank_top_k: int = 5,
    process_images: bool = True,
    chunk_size: int = 512,
    llm_client: Optional[Any] = None,
    tool_executor: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Process documents and answer questions with source citations.

    This handler orchestrates the document Q&A pipeline:
    1. Extract content from documents using MarkItDown
    2. Process complex images with Vision LLM (optional)
    3. Chunk content semantically
    4. Index chunks with embeddings
    5. Retrieve relevant chunks with hybrid search
    6. Rerank with LLM-as-judge
    7. Assemble context with strategic positioning
    8. Generate answer with source citations

    Args:
        documents: List of file paths (PDF, DOCX, XLSX, PPTX, HTML, CSV)
        query: User question to answer
        include_sources: Include source citations in response (default: True)
        max_context_tokens: Maximum tokens for context window (default: 6000)
        rerank_top_k: Number of chunks after reranking (default: 5)
        process_images: Process images with Vision LLM (default: True)
        chunk_size: Target chunk size in tokens (default: 512)
        llm_client: Optional LLM client (uses OpenAI if not provided)
        tool_executor: Optional ToolExecutor instance (not used directly)

    Returns:
        Dictionary with:
        - success: bool
        - answer: str (with [src_XXX] citations)
        - sources: list of source metadata
        - chunks_retrieved: int
        - confidence: float (0-1)
        - metadata: dict with execution stats
        - error: str (if failed)

    Example:
        result = document_qa(
            documents=["report.pdf", "slides.pptx"],
            query="What was the Q3 revenue?"
        )
        # Output:
        # {
        #   "success": True,
        #   "answer": "The Q3 revenue was $2.3M [src_001]...",
        #   "sources": [
        #     {"id": "src_001", "file": "report.pdf", "page": 3}
        #   ],
        #   "chunks_retrieved": 5,
        #   "confidence": 0.92
        # }
    """
    start_time = time.time()

    # Validate inputs
    if not documents:
        return {
            "success": False,
            "answer": "",
            "sources": [],
            "chunks_retrieved": 0,
            "confidence": 0.0,
            "error": "No documents provided",
            "metadata": {"execution_time_seconds": 0},
        }

    if not query or not query.strip():
        return {
            "success": False,
            "answer": "",
            "sources": [],
            "chunks_retrieved": 0,
            "confidence": 0.0,
            "error": "No query provided",
            "metadata": {"execution_time_seconds": 0},
        }

    # Validate file paths
    valid_documents = []
    for doc_path in documents:
        path = Path(doc_path)
        if path.exists():
            valid_documents.append(str(path))
        else:
            logger.warning(f"File not found: {doc_path}")

    if not valid_documents:
        return {
            "success": False,
            "answer": "",
            "sources": [],
            "chunks_retrieved": 0,
            "confidence": 0.0,
            "error": "No valid document files found",
            "metadata": {"execution_time_seconds": 0},
        }

    logger.info(f"Starting document Q&A: {len(valid_documents)} documents, query='{query[:50]}...'")

    # Initialize result structure
    result = {
        "success": False,
        "answer": "",
        "sources": [],
        "chunks_retrieved": 0,
        "confidence": 0.0,
        "metadata": {
            "documents_processed": len(valid_documents),
            "execution_time_seconds": 0,
            "chunks_created": 0,
            "images_processed": 0,
        },
    }

    try:
        # Initialize source tracker
        source_tracker = SourceTracker()

        # Step 1: Extract content from documents
        logger.info("Step 1: Extracting content from documents...")
        extraction_result = run_extraction(
            file_paths=valid_documents,
            llm_client=llm_client,
            extract_images=process_images,
            chunk_size=chunk_size,
            source_tracker=source_tracker,
        )

        chunks = extraction_result["chunks"]
        image_descriptions = extraction_result.get("image_descriptions", [])

        if not chunks:
            result["error"] = "No content extracted from documents"
            return result

        result["metadata"]["chunks_created"] = len(chunks)
        result["metadata"]["images_processed"] = len(image_descriptions)

        # Step 2: Add image descriptions as additional chunks
        if image_descriptions:
            chunks = _add_image_chunks(chunks, image_descriptions, source_tracker)

        # Step 3: Index chunks
        logger.info(f"Step 2: Indexing {len(chunks)} chunks...")
        indexing_result = run_indexing(documents=chunks)
        document_store = indexing_result["document_store"]

        # Step 4: Generate answer
        logger.info("Step 3: Generating answer...")
        generation_result = run_generation(
            query=query,
            document_store=document_store,
            llm_client=llm_client,
            rerank_top_k=rerank_top_k,
            max_context_tokens=max_context_tokens,
            source_tracker=source_tracker,
        )

        # Build final result
        result["success"] = True
        result["answer"] = generation_result["answer"]
        result["sources"] = generation_result["sources"] if include_sources else []
        result["chunks_retrieved"] = rerank_top_k
        result["confidence"] = generation_result["confidence"]

    except Exception as e:
        logger.error(f"Document Q&A failed: {e}")
        result["error"] = str(e)

    result["metadata"]["execution_time_seconds"] = round(time.time() - start_time, 2)
    return result


def _add_image_chunks(
    chunks: List[Any],
    image_descriptions: List[ImageDescription],
    source_tracker: SourceTracker,
) -> List[Any]:
    """Add image descriptions as searchable chunks."""
    from haystack import Document

    from .utils.source_tracker import SourceType

    for img_desc in image_descriptions:
        if not img_desc.description:
            continue

        # Create source entry for image
        source_id = source_tracker.add_source(
            source_type=SourceType.IMAGE,
            file_path=img_desc.image.source_file,
            page=img_desc.image.page,
            slide=img_desc.image.slide,
            content=img_desc.description[:200],
        )

        # Create document for image description
        doc = Document(
            content=f"[Image Description] {img_desc.description}",
            meta={
                "source_id": source_id,
                "file_path": img_desc.image.source_file,
                "chunk_type": "image",
                "is_image": True,
            },
        )
        chunks.append(doc)

    return chunks


def extract_only(
    documents: List[str],
    process_images: bool = True,
    chunk_size: int = 512,
) -> Dict[str, Any]:
    """
    Extract content from documents without Q&A.

    Useful for pre-processing documents before multiple queries.

    Args:
        documents: List of file paths
        process_images: Process images with Vision LLM
        chunk_size: Target chunk size

    Returns:
        Dictionary with 'chunks', 'images', and 'source_tracker'
    """
    source_tracker = SourceTracker()

    extraction_result = run_extraction(
        file_paths=documents,
        extract_images=process_images,
        chunk_size=chunk_size,
        source_tracker=source_tracker,
    )

    return {
        "chunks": extraction_result["chunks"],
        "images": extraction_result.get("image_descriptions", []),
        "source_tracker": source_tracker,
        "statistics": extraction_result["statistics"],
    }


def query_indexed_documents(
    query: str,
    document_store: InMemoryDocumentStore,
    source_tracker: Optional[SourceTracker] = None,
    max_context_tokens: int = 6000,
    rerank_top_k: int = 5,
) -> Dict[str, Any]:
    """
    Query pre-indexed documents.

    Useful when documents are already extracted and indexed.

    Args:
        query: User question
        document_store: Pre-populated document store
        source_tracker: Source tracker from extraction
        max_context_tokens: Maximum context tokens
        rerank_top_k: Number of chunks after reranking

    Returns:
        Dictionary with 'answer', 'sources', 'confidence'
    """
    return run_generation(
        query=query,
        document_store=document_store,
        rerank_top_k=rerank_top_k,
        max_context_tokens=max_context_tokens,
        source_tracker=source_tracker,
    )
