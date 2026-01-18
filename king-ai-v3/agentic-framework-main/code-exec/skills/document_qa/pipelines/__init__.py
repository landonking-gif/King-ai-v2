"""
Haystack 2.x Pipelines for Document Q&A.

Module: code-exec/skills/document_qa/pipelines/__init__.py
"""

from .extraction import create_extraction_pipeline
from .indexing import create_indexing_pipeline
from .generation import create_generation_pipeline

__all__ = [
    "create_extraction_pipeline",
    "create_indexing_pipeline",
    "create_generation_pipeline",
]
