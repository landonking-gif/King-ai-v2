"""
Context Assembler Component for Haystack 2.x.

Module: code-exec/skills/document_qa/components/context_assembler.py

Assembles context with strategic positioning for best LLM recall.
Addresses the "lost in the middle" problem by placing critical info at start/end.
"""

import logging
from typing import Any, Dict, List, Optional

from haystack import Document, component

from ..utils.source_tracker import SourceEntry, SourceTracker, SourceType

logger = logging.getLogger(__name__)


@component
class ContextAssembler:
    """
    Assemble context with strategic positioning for best LLM recall.

    Addresses the "lost in the middle" problem:
    - Most relevant content at START and END
    - Supporting content in the middle
    - Source citations injected inline

    Output Structure:
    [MOST RELEVANT - Position 1]
    [src_001] Content from report.pdf page 3...

    [SUPPORTING - Position 2-4]
    [src_002] Content from slides.pptx slide 5...

    [SECOND MOST RELEVANT - Position 5]
    [src_003] Content from data.xlsx...

    ---
    Sources:
    [src_001] report.pdf, Page 3, "Revenue Analysis"
    ...
    """

    def __init__(
        self,
        max_tokens: int = 6000,
        include_source_citations: bool = True,
        position_strategy: str = "edges",
    ):
        """
        Initialize the context assembler.

        Args:
            max_tokens: Maximum tokens for assembled context
            include_source_citations: Include [src_XXX] inline citations
            position_strategy: "edges" (important at start/end) or "linear"
        """
        self.max_tokens = max_tokens
        self.include_source_citations = include_source_citations
        self.position_strategy = position_strategy

    @component.output_types(context=str, sources=List[Dict[str, Any]])
    def run(
        self,
        documents: List[Document],
        source_tracker: Optional[SourceTracker] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Assemble context from ranked documents.

        Args:
            documents: Ranked documents (highest score first)
            source_tracker: Optional SourceTracker for citation management
            max_tokens: Override default max_tokens

        Returns:
            Dictionary with 'context' (str) and 'sources' (list)
        """
        if not documents:
            return {"context": "", "sources": []}

        max_tokens = max_tokens or self.max_tokens
        tracker = source_tracker or SourceTracker()

        # Reorder documents for strategic positioning
        positioned_docs = self._position_documents(documents)

        # Build context with citations
        context_parts: List[str] = []
        sources_used: List[Dict[str, Any]] = []
        current_tokens = 0

        for i, (doc, position_label) in enumerate(positioned_docs):
            # Estimate tokens for this document
            doc_tokens = len(doc.content) // 4

            # Check if we can fit this document
            if current_tokens + doc_tokens > max_tokens:
                # Try to truncate if it's the last important one
                available = max_tokens - current_tokens
                if available > 100:  # Only if we have meaningful space
                    truncated = self._truncate_content(doc.content, available)
                    if truncated:
                        part, source = self._format_document_part(
                            doc, truncated, tracker, position_label
                        )
                        context_parts.append(part)
                        sources_used.append(source)
                break

            # Add document to context
            part, source = self._format_document_part(
                doc, doc.content, tracker, position_label
            )
            context_parts.append(part)
            sources_used.append(source)
            current_tokens += doc_tokens

        # Assemble final context
        context = "\n\n".join(context_parts)

        # Add source citations block if enabled
        if self.include_source_citations and sources_used:
            source_block = self._format_sources_block(sources_used)
            context = f"{context}\n\n{source_block}"

        logger.info(
            f"Assembled context: {len(positioned_docs)} docs -> "
            f"~{current_tokens} tokens"
        )

        return {
            "context": context,
            "sources": sources_used,
        }

    def _position_documents(
        self, documents: List[Document]
    ) -> List[tuple[Document, str]]:
        """
        Reorder documents for strategic positioning.

        For "edges" strategy:
        - Position 1: Highest ranked (start)
        - Position 2-4: Middle ranked (middle)
        - Position 5: Second highest (end)

        This places the most important content at the edges where LLMs
        have better recall.
        """
        if self.position_strategy != "edges" or len(documents) <= 2:
            return [(doc, "linear") for doc in documents]

        positioned = []

        # First position: highest ranked
        if documents:
            positioned.append((documents[0], "primary"))

        # Last position: second highest (we'll add this at the end)
        second_highest = documents[1] if len(documents) > 1 else None

        # Middle positions: remaining documents
        middle_docs = documents[2:] if len(documents) > 2 else []
        for doc in middle_docs:
            positioned.append((doc, "supporting"))

        # Add second highest at the end
        if second_highest:
            positioned.append((second_highest, "secondary"))

        return positioned

    def _format_document_part(
        self,
        document: Document,
        content: str,
        tracker: SourceTracker,
        position_label: str,
    ) -> tuple[str, Dict[str, Any]]:
        """Format a document part with citation."""
        # Get or create source ID
        source_id = document.meta.get("source_id")

        if not source_id:
            # Create new source entry
            source_id = tracker.add_source(
                source_type=SourceType.DOCUMENT,
                file_path=document.meta.get("file_name", "unknown"),
                page=document.meta.get("page"),
                section=document.meta.get("section"),
                content=content[:200],
            )

        # Build source metadata
        source_meta = {
            "id": source_id,
            "file": document.meta.get("file_name", "unknown"),
            "page": document.meta.get("page"),
            "section": document.meta.get("section"),
            "position": position_label,
            "rerank_score": document.meta.get("rerank_score"),
        }

        # Format content with citation
        if self.include_source_citations:
            formatted = f"[{source_id}] {content}"
        else:
            formatted = content

        return formatted, source_meta

    def _format_sources_block(self, sources: List[Dict[str, Any]]) -> str:
        """Format the sources citation block."""
        lines = ["---", "Sources:"]

        for source in sources:
            parts = [source.get("file", "unknown")]

            if source.get("page"):
                parts.append(f"Page {source['page']}")
            if source.get("section"):
                parts.append(f'"{source["section"]}"')

            citation = ", ".join(parts)
            lines.append(f"[{source['id']}] {citation}")

        return "\n".join(lines)

    def _truncate_content(self, content: str, max_tokens: int) -> str:
        """Truncate content to fit within token limit."""
        max_chars = max_tokens * 4  # Rough estimate

        if len(content) <= max_chars:
            return content

        # Try to truncate at sentence boundary
        truncated = content[:max_chars]
        last_period = truncated.rfind(".")
        if last_period > max_chars * 0.7:  # At least 70% kept
            return truncated[: last_period + 1]

        return truncated + "..."


def assemble_context(
    documents: List[Document],
    max_tokens: int = 6000,
) -> Dict[str, Any]:
    """
    Convenience function to assemble context.

    Args:
        documents: Ranked documents
        max_tokens: Maximum tokens

    Returns:
        Dictionary with 'context' and 'sources'
    """
    assembler = ContextAssembler(max_tokens=max_tokens)
    return assembler.run(documents=documents)
