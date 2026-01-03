"""
Extraction Pipeline for Document Q&A.

Module: code-exec/skills/document_qa/pipelines/extraction.py

Haystack 2.x pipeline for document extraction:
1. MarkItDown conversion
2. Vision LLM image description
3. Semantic chunking
"""

import logging
from typing import Any, Dict, List, Optional

from haystack import Pipeline

from ..components.markitdown_converter import MarkItDownConverter
from ..components.semantic_chunker import SemanticChunker
from ..components.vision_describer import VisionImageDescriber
from ..utils.source_tracker import SourceTracker

logger = logging.getLogger(__name__)


def create_extraction_pipeline(
    llm_client: Optional[Any] = None,
    extract_images: bool = True,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> Pipeline:
    """
    Create the document extraction pipeline.

    Pipeline flow:
    file_paths -> MarkItDownConverter -> [documents, images]
                                          |          |
                                          v          v
                              SemanticChunker  VisionImageDescriber
                                          |          |
                                          v          v
                                      chunks    image_descriptions

    Args:
        llm_client: Optional LLM client for vision descriptions
        extract_images: Whether to extract and describe images
        chunk_size: Target chunk size in tokens
        chunk_overlap: Overlap between chunks

    Returns:
        Configured Haystack Pipeline
    """
    pipeline = Pipeline()

    # Add components
    pipeline.add_component(
        "converter",
        MarkItDownConverter(extract_images=extract_images),
    )

    pipeline.add_component(
        "chunker",
        SemanticChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap),
    )

    if extract_images:
        pipeline.add_component(
            "vision",
            VisionImageDescriber(llm_client=llm_client),
        )

    # Connect components
    # Converter outputs documents and images
    pipeline.connect("converter.documents", "chunker.documents")

    if extract_images:
        pipeline.connect("converter.images", "vision.images")

    return pipeline


def run_extraction(
    file_paths: List[str],
    llm_client: Optional[Any] = None,
    extract_images: bool = True,
    chunk_size: int = 512,
    source_tracker: Optional[SourceTracker] = None,
) -> Dict[str, Any]:
    """
    Run the extraction pipeline on documents.

    Args:
        file_paths: List of file paths to process
        llm_client: Optional LLM client for vision
        extract_images: Whether to process images
        chunk_size: Target chunk size
        source_tracker: Optional source tracker for citations

    Returns:
        Dictionary with 'chunks', 'image_descriptions', and 'statistics'
    """
    pipeline = create_extraction_pipeline(
        llm_client=llm_client,
        extract_images=extract_images,
        chunk_size=chunk_size,
    )

    tracker = source_tracker or SourceTracker()

    # Build input
    input_data = {
        "converter": {
            "file_paths": file_paths,
            "source_tracker": tracker,
        },
    }

    # Run pipeline
    result = pipeline.run(input_data)

    # Collect results
    chunks = result.get("chunker", {}).get("chunks", [])
    image_descriptions = []

    if extract_images and "vision" in result:
        image_descriptions = result["vision"].get("descriptions", [])

    # Build statistics
    statistics = {
        "files_processed": len(file_paths),
        "chunks_created": len(chunks),
        "images_described": len(image_descriptions),
        "sources_tracked": tracker.count,
    }

    logger.info(
        f"Extraction complete: {len(file_paths)} files -> "
        f"{len(chunks)} chunks, {len(image_descriptions)} images"
    )

    return {
        "chunks": chunks,
        "image_descriptions": image_descriptions,
        "source_tracker": tracker,
        "statistics": statistics,
    }


def extract_documents_simple(file_paths: List[str]) -> Dict[str, Any]:
    """
    Simple extraction without vision processing.

    Args:
        file_paths: List of file paths to process

    Returns:
        Dictionary with chunks and statistics
    """
    return run_extraction(
        file_paths=file_paths,
        extract_images=False,
    )
