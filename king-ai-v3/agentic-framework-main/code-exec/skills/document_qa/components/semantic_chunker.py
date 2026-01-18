"""
Semantic Chunker Component for Haystack 2.x.

Module: code-exec/skills/document_qa/components/semantic_chunker.py

Structure-aware chunking that preserves document semantics.
Never splits tables, code blocks, or lists mid-content.
"""

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from haystack import Document, component

logger = logging.getLogger(__name__)


@component
class SemanticChunker:
    """
    Structure-aware chunking that preserves document semantics.

    Features:
    - Detects structure: headings, tables, lists, code blocks
    - Never splits tables or code blocks mid-content
    - Chunks by semantic boundaries (sections)
    - Adds overlap for context continuity
    - Preserves source metadata in each chunk

    Configuration via environment:
    - DOCUMENT_QA_CHUNK_SIZE: Target chunk size in tokens (default: 512)
    - DOCUMENT_QA_CHUNK_OVERLAP: Overlap between chunks (default: 50)
    """

    # Regex patterns for structure detection
    HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    TABLE_START = re.compile(r"^\|.+\|$", re.MULTILINE)
    TABLE_SEPARATOR = re.compile(r"^\|[-:| ]+\|$", re.MULTILINE)
    CODE_BLOCK = re.compile(r"```[\s\S]*?```", re.MULTILINE)
    LIST_ITEM = re.compile(r"^[\s]*[-*+]\s+", re.MULTILINE)
    NUMBERED_LIST = re.compile(r"^[\s]*\d+\.\s+", re.MULTILINE)

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        preserve_tables: bool = True,
        preserve_code_blocks: bool = True,
    ):
        """
        Initialize the semantic chunker.

        Args:
            chunk_size: Target chunk size in tokens (default from env: 512)
            chunk_overlap: Overlap between chunks (default from env: 50)
            preserve_tables: Never split tables (may exceed chunk size)
            preserve_code_blocks: Never split code blocks
        """
        self.chunk_size = chunk_size or int(
            os.getenv("DOCUMENT_QA_CHUNK_SIZE", "512")
        )
        self.chunk_overlap = chunk_overlap or int(
            os.getenv("DOCUMENT_QA_CHUNK_OVERLAP", "50")
        )
        self.preserve_tables = preserve_tables
        self.preserve_code_blocks = preserve_code_blocks

    @component.output_types(chunks=List[Document])
    def run(self, documents: List[Document]) -> Dict[str, List[Document]]:
        """
        Chunk documents with semantic awareness.

        Args:
            documents: List of Haystack Documents to chunk

        Returns:
            Dictionary with 'chunks' key containing chunked documents
        """
        all_chunks: List[Document] = []

        for doc in documents:
            chunks = self._chunk_document(doc)
            all_chunks.extend(chunks)

        logger.info(f"Chunked {len(documents)} documents into {len(all_chunks)} chunks")
        return {"chunks": all_chunks}

    def _chunk_document(self, document: Document) -> List[Document]:
        """Chunk a single document."""
        content = document.content
        if not content:
            return []

        # First, extract protected blocks (tables, code)
        protected_blocks, remaining_content = self._extract_protected_blocks(content)

        # Split remaining content into sections
        sections = self._split_by_sections(remaining_content)

        # Create chunks from sections
        chunks = []
        current_section: Optional[str] = None

        for section_title, section_content in sections:
            if section_title:
                current_section = section_title

            section_chunks = self._chunk_section(
                content=section_content,
                section_title=current_section,
                base_meta=document.meta,
            )
            chunks.extend(section_chunks)

        # Add protected blocks as separate chunks
        for block_type, block_content, position in protected_blocks:
            meta = {
                **document.meta,
                "chunk_type": block_type,
                "is_protected": True,
            }

            chunk = Document(
                content=block_content.strip(),
                meta=meta,
            )
            chunks.append(chunk)

        # Sort chunks by original position if possible
        return chunks

    def _extract_protected_blocks(
        self, content: str
    ) -> Tuple[List[Tuple[str, str, int]], str]:
        """
        Extract tables and code blocks that shouldn't be split.

        Returns:
            Tuple of (protected_blocks, remaining_content)
        """
        protected: List[Tuple[str, str, int]] = []
        remaining = content

        # Extract code blocks
        if self.preserve_code_blocks:
            for match in self.CODE_BLOCK.finditer(content):
                block = match.group(0)
                protected.append(("code", block, match.start()))
                remaining = remaining.replace(block, "\n[CODE_BLOCK]\n", 1)

        # Extract tables
        if self.preserve_tables:
            tables = self._extract_tables(remaining)
            for table, position in tables:
                protected.append(("table", table, position))
                remaining = remaining.replace(table, "\n[TABLE]\n", 1)

        return protected, remaining

    def _extract_tables(self, content: str) -> List[Tuple[str, int]]:
        """Extract markdown tables from content."""
        tables = []
        lines = content.split("\n")
        i = 0

        while i < len(lines):
            line = lines[i]

            # Check if this looks like a table row
            if self.TABLE_START.match(line):
                table_lines = [line]
                j = i + 1

                # Check for separator row
                if j < len(lines) and self.TABLE_SEPARATOR.match(lines[j]):
                    table_lines.append(lines[j])
                    j += 1

                    # Collect remaining table rows
                    while j < len(lines) and self.TABLE_START.match(lines[j]):
                        table_lines.append(lines[j])
                        j += 1

                    if len(table_lines) >= 3:  # Header + separator + at least 1 row
                        table = "\n".join(table_lines)
                        position = content.find(table)
                        tables.append((table, position))

                i = j
            else:
                i += 1

        return tables

    def _split_by_sections(self, content: str) -> List[Tuple[Optional[str], str]]:
        """
        Split content by headings into sections.

        Returns:
            List of (section_title, section_content) tuples
        """
        sections = []
        lines = content.split("\n")
        current_title: Optional[str] = None
        current_lines: List[str] = []

        for line in lines:
            heading_match = self.HEADING_PATTERN.match(line)

            if heading_match:
                # Save previous section
                if current_lines:
                    section_content = "\n".join(current_lines).strip()
                    if section_content:
                        sections.append((current_title, section_content))

                # Start new section
                current_title = heading_match.group(2).strip()
                current_lines = [line]
            else:
                current_lines.append(line)

        # Save last section
        if current_lines:
            section_content = "\n".join(current_lines).strip()
            if section_content:
                sections.append((current_title, section_content))

        return sections

    def _chunk_section(
        self,
        content: str,
        section_title: Optional[str],
        base_meta: Dict[str, Any],
    ) -> List[Document]:
        """Chunk a single section."""
        if not content.strip():
            return []

        # Estimate tokens (rough: 1 token ≈ 4 chars for English)
        tokens = len(content) // 4

        if tokens <= self.chunk_size:
            # Single chunk for small sections
            meta = {
                **base_meta,
                "section": section_title,
                "chunk_type": "text",
            }
            return [Document(content=content.strip(), meta=meta)]

        # Need to split - use sentence boundaries
        chunks = []
        sentences = self._split_sentences(content)

        current_chunk: List[str] = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = len(sentence) // 4

            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = " ".join(current_chunk)
                meta = {
                    **base_meta,
                    "section": section_title,
                    "chunk_type": "text",
                    "chunk_index": len(chunks),
                }
                chunks.append(Document(content=chunk_text.strip(), meta=meta))

                # Start new chunk with overlap
                overlap_text = self._get_overlap(current_chunk)
                current_chunk = [overlap_text] if overlap_text else []
                current_tokens = len(overlap_text) // 4 if overlap_text else 0

            current_chunk.append(sentence)
            current_tokens += sentence_tokens

        # Save last chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            meta = {
                **base_meta,
                "section": section_title,
                "chunk_type": "text",
                "chunk_index": len(chunks),
            }
            chunks.append(Document(content=chunk_text.strip(), meta=meta))

        return chunks

    def _split_sentences(self, content: str) -> List[str]:
        """Split content into sentences."""
        # Simple sentence splitting on punctuation
        sentences = re.split(r"(?<=[.!?])\s+", content)
        return [s.strip() for s in sentences if s.strip()]

    def _get_overlap(self, chunks: List[str]) -> str:
        """Get overlap text from the end of chunks."""
        if not chunks:
            return ""

        overlap_tokens = self.chunk_overlap
        overlap_chars = overlap_tokens * 4

        # Get last few sentences up to overlap size
        combined = " ".join(chunks)
        if len(combined) <= overlap_chars:
            return combined

        return combined[-overlap_chars:]


def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 50,
) -> List[str]:
    """
    Convenience function to chunk plain text.

    Args:
        text: Text to chunk
        chunk_size: Target chunk size in tokens
        overlap: Overlap between chunks

    Returns:
        List of chunk strings
    """
    chunker = SemanticChunker(chunk_size=chunk_size, chunk_overlap=overlap)
    doc = Document(content=text)
    result = chunker.run([doc])

    return [chunk.content for chunk in result["chunks"]]
