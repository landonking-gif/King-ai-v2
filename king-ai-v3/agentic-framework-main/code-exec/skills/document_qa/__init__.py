"""
Document Q&A Skill.

Module: code-exec/skills/document_qa/__init__.py

A comprehensive document extraction and Q&A skill using Haystack 2.x pipelines.
Supports PDF, Word, Excel, PowerPoint with source tracking and Vision LLM integration.
"""

from .handler import document_qa

__all__ = ["document_qa"]
__version__ = "1.0.0"
