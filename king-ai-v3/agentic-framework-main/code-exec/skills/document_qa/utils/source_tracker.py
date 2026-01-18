"""
Unified Source Tracking for Document Q&A.

Module: code-exec/skills/document_qa/utils/source_tracker.py

Provides consistent source citation tracking across document extraction,
retrieval, and generation phases.
"""

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class SourceType(Enum):
    """Types of sources that can be tracked."""

    DOCUMENT = "document"
    WEB = "web"
    IMAGE = "image"
    TABLE = "table"
    CODE = "code"


@dataclass
class SourceEntry:
    """Represents a single source entry with full metadata."""

    source_id: str  # e.g., "src_001"
    source_type: SourceType
    file_path: str
    page: Optional[int] = None
    section: Optional[str] = None
    slide: Optional[int] = None
    sheet: Optional[str] = None
    row_range: Optional[str] = None
    content_hash: Optional[str] = None
    content_preview: Optional[str] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_citation(self) -> str:
        """Generate human-readable citation string."""
        parts = [self.file_path]

        if self.page is not None:
            parts.append(f"Page {self.page}")
        if self.slide is not None:
            parts.append(f"Slide {self.slide}")
        if self.sheet is not None:
            parts.append(f"Sheet '{self.sheet}'")
        if self.row_range is not None:
            parts.append(f"Rows {self.row_range}")
        if self.section:
            parts.append(f'"{self.section}"')

        return ", ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "id": self.source_id,
            "type": self.source_type.value,
            "file": self.file_path,
        }

        if self.page is not None:
            result["page"] = self.page
        if self.slide is not None:
            result["slide"] = self.slide
        if self.sheet is not None:
            result["sheet"] = self.sheet
        if self.row_range is not None:
            result["row_range"] = self.row_range
        if self.section:
            result["section"] = self.section
        if self.content_preview:
            result["preview"] = self.content_preview[:100]

        return result


class SourceTracker:
    """
    Tracks and manages source citations throughout the document Q&A pipeline.

    Usage:
        tracker = SourceTracker()
        source_id = tracker.add_source(
            source_type=SourceType.DOCUMENT,
            file_path="report.pdf",
            page=3,
            section="Revenue Analysis"
        )
        # Later: tracker.get_source(source_id)
        # Or: tracker.get_all_sources()
    """

    def __init__(self, prefix: str = "src"):
        """
        Initialize the source tracker.

        Args:
            prefix: Prefix for source IDs (default: "src")
        """
        self.prefix = prefix
        self._sources: Dict[str, SourceEntry] = {}
        self._counter = 0
        self._content_hashes: Dict[str, str] = {}  # For deduplication

    def add_source(
        self,
        source_type: SourceType,
        file_path: str,
        page: Optional[int] = None,
        section: Optional[str] = None,
        slide: Optional[int] = None,
        sheet: Optional[str] = None,
        row_range: Optional[str] = None,
        content: Optional[str] = None,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Add a new source and return its ID.

        Args:
            source_type: Type of source
            file_path: Path to the source file
            page: Page number (for PDFs, Word docs)
            section: Section or heading name
            slide: Slide number (for PowerPoint)
            sheet: Sheet name (for Excel)
            row_range: Row range string (for Excel, e.g., "15-30")
            content: Optional content for hashing/preview
            confidence: Confidence score (0-1)
            metadata: Additional metadata

        Returns:
            Source ID (e.g., "src_001")
        """
        # Generate content hash for deduplication
        content_hash = None
        if content:
            content_hash = hashlib.md5(content.encode()).hexdigest()[:12]

            # Check for duplicate content
            if content_hash in self._content_hashes:
                return self._content_hashes[content_hash]

        # Generate new source ID
        self._counter += 1
        source_id = f"{self.prefix}_{self._counter:03d}"

        # Create source entry
        entry = SourceEntry(
            source_id=source_id,
            source_type=source_type,
            file_path=file_path,
            page=page,
            section=section,
            slide=slide,
            sheet=sheet,
            row_range=row_range,
            content_hash=content_hash,
            content_preview=content[:200] if content else None,
            confidence=confidence,
            metadata=metadata or {},
        )

        self._sources[source_id] = entry

        if content_hash:
            self._content_hashes[content_hash] = source_id

        return source_id

    def get_source(self, source_id: str) -> Optional[SourceEntry]:
        """Get a source entry by ID."""
        return self._sources.get(source_id)

    def get_all_sources(self) -> List[SourceEntry]:
        """Get all tracked sources in order."""
        return list(self._sources.values())

    def get_sources_as_list(self) -> List[Dict[str, Any]]:
        """Get all sources as list of dictionaries."""
        return [source.to_dict() for source in self._sources.values()]

    def format_citations(self) -> str:
        """Format all sources as a citation block."""
        lines = ["---", "Sources:"]
        for source in self._sources.values():
            lines.append(f"[{source.source_id}] {source.to_citation()}")
        return "\n".join(lines)

    def inject_citation(self, text: str, source_id: str) -> str:
        """
        Inject a citation into text.

        Args:
            text: Text to append citation to
            source_id: Source ID to cite

        Returns:
            Text with citation appended
        """
        return f"{text} [{source_id}]"

    def clear(self) -> None:
        """Clear all tracked sources."""
        self._sources.clear()
        self._content_hashes.clear()
        self._counter = 0

    @property
    def count(self) -> int:
        """Return the number of tracked sources."""
        return len(self._sources)
