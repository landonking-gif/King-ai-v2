"""
MarkItDown Converter Component for Haystack 2.x.

Module: code-exec/skills/document_qa/components/markitdown_converter.py

Converts documents (PDF, DOCX, XLSX, PPTX) to Markdown using Microsoft's MarkItDown library.
Extracts embedded images for Vision LLM processing.
"""

import io
import logging
import os
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from haystack import Document, component

from ..utils.image_filter import ImageData
from ..utils.source_tracker import SourceTracker, SourceType

logger = logging.getLogger(__name__)


@component
class MarkItDownConverter:
    """
    Convert documents to Markdown with source tracking and image extraction.

    Supports:
    - PDF (.pdf)
    - Word (.docx)
    - Excel (.xlsx)
    - PowerPoint (.pptx)
    - HTML (.html, .htm)
    - CSV (.csv)

    Outputs:
    - documents: List[Document] with markdown content and source metadata
    - images: List[ImageData] for Vision LLM processing
    """

    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".pptx", ".html", ".htm", ".csv"}

    def __init__(self, extract_images: bool = True):
        """
        Initialize the converter.

        Args:
            extract_images: Whether to extract images from documents
        """
        self.extract_images = extract_images
        self._markitdown = None

    def _get_markitdown(self) -> Any:
        """Lazy load MarkItDown to avoid import errors if not installed."""
        if self._markitdown is None:
            try:
                from markitdown import MarkItDown

                self._markitdown = MarkItDown()
            except ImportError as e:
                logger.error("MarkItDown not installed. Run: pip install markitdown")
                raise ImportError(
                    "MarkItDown is required. Install with: pip install markitdown"
                ) from e
        return self._markitdown

    @component.output_types(documents=List[Document], images=List[ImageData])
    def run(
        self,
        file_paths: List[str],
        source_tracker: Optional[SourceTracker] = None,
    ) -> Dict[str, Any]:
        """
        Convert documents to Markdown.

        Args:
            file_paths: List of file paths to convert
            source_tracker: Optional SourceTracker for citation tracking

        Returns:
            Dictionary with 'documents' and 'images' keys
        """
        documents: List[Document] = []
        all_images: List[ImageData] = []

        tracker = source_tracker or SourceTracker()

        for file_path in file_paths:
            path = Path(file_path)

            if not path.exists():
                logger.warning(f"File not found: {file_path}")
                continue

            extension = path.suffix.lower()
            if extension not in self.SUPPORTED_EXTENSIONS:
                logger.warning(f"Unsupported file type: {extension}")
                continue

            try:
                # Convert document to markdown
                markdown_content, page_breaks = self._convert_file(path)

                if not markdown_content:
                    continue

                # Extract images if enabled
                images = []
                if self.extract_images and extension in {".docx", ".pptx", ".xlsx"}:
                    images = self._extract_images_from_ooxml(path)
                    all_images.extend(images)

                # Create documents with page/section metadata
                docs = self._create_documents(
                    content=markdown_content,
                    file_path=str(path),
                    extension=extension,
                    page_breaks=page_breaks,
                    tracker=tracker,
                )
                documents.extend(docs)

                logger.info(
                    f"Converted {path.name}: {len(docs)} sections, {len(images)} images"
                )

            except Exception as e:
                logger.error(f"Failed to convert {file_path}: {e}")
                continue

        return {"documents": documents, "images": all_images}

    def _convert_file(self, path: Path) -> Tuple[str, List[int]]:
        """
        Convert a file to markdown and detect page breaks.

        Returns:
            Tuple of (markdown_content, page_break_positions)
        """
        md = self._get_markitdown()

        try:
            result = md.convert(str(path))
            content = result.text_content if hasattr(result, "text_content") else str(result)

            # Detect page breaks (MarkItDown inserts <!-- Page Break --> markers)
            page_breaks = []
            for match in re.finditer(r"<!-- Page Break -->|---\s*Page \d+\s*---", content):
                page_breaks.append(match.start())

            return content, page_breaks

        except Exception as e:
            logger.error(f"MarkItDown conversion failed: {e}")
            return "", []

    def _create_documents(
        self,
        content: str,
        file_path: str,
        extension: str,
        page_breaks: List[int],
        tracker: SourceTracker,
    ) -> List[Document]:
        """
        Create Haystack Documents with proper metadata.

        For documents with page breaks, creates separate documents per page.
        Otherwise creates a single document.
        """
        documents = []
        file_name = Path(file_path).name

        # Determine source type
        source_type = SourceType.DOCUMENT
        if extension == ".xlsx":
            source_type = SourceType.TABLE
        elif extension == ".pptx":
            source_type = SourceType.DOCUMENT

        if not page_breaks:
            # Single document
            source_id = tracker.add_source(
                source_type=source_type,
                file_path=file_name,
                content=content[:500],
            )

            doc = Document(
                content=content,
                meta={
                    "source_id": source_id,
                    "file_path": file_path,
                    "file_name": file_name,
                    "extension": extension,
                    "page": 1,
                },
            )
            documents.append(doc)
        else:
            # Split by page breaks
            positions = [0] + page_breaks + [len(content)]
            for i in range(len(positions) - 1):
                page_content = content[positions[i] : positions[i + 1]].strip()
                if not page_content:
                    continue

                page_num = i + 1
                source_id = tracker.add_source(
                    source_type=source_type,
                    file_path=file_name,
                    page=page_num,
                    content=page_content[:500],
                )

                doc = Document(
                    content=page_content,
                    meta={
                        "source_id": source_id,
                        "file_path": file_path,
                        "file_name": file_name,
                        "extension": extension,
                        "page": page_num,
                    },
                )
                documents.append(doc)

        return documents

    def _extract_images_from_ooxml(self, path: Path) -> List[ImageData]:
        """
        Extract images from Office Open XML formats (docx, pptx, xlsx).

        These formats are ZIP archives containing media files.
        """
        images = []
        image_extensions = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff"}

        try:
            with zipfile.ZipFile(path, "r") as zf:
                for name in zf.namelist():
                    # Images are typically in media folders
                    if not any(
                        folder in name.lower()
                        for folder in ["media", "image", "pictures"]
                    ):
                        continue

                    ext = Path(name).suffix.lower()
                    if ext not in image_extensions:
                        continue

                    try:
                        image_bytes = zf.read(name)

                        # Try to get image dimensions
                        size = self._get_image_size(image_bytes)

                        image_data = ImageData(
                            image_bytes=image_bytes,
                            file_path=name,
                            source_file=str(path),
                            image_format=ext.lstrip("."),
                            size=size,
                            metadata={"archive_path": name},
                        )
                        images.append(image_data)

                    except Exception as e:
                        logger.debug(f"Failed to extract image {name}: {e}")
                        continue

        except zipfile.BadZipFile:
            logger.warning(f"Not a valid ZIP/OOXML file: {path}")
        except Exception as e:
            logger.error(f"Failed to extract images from {path}: {e}")

        return images

    def _get_image_size(self, image_bytes: bytes) -> Optional[Tuple[int, int]]:
        """Get image dimensions without loading full image."""
        try:
            from PIL import Image

            with Image.open(io.BytesIO(image_bytes)) as img:
                return img.size
        except ImportError:
            logger.debug("PIL not available for image size detection")
            return None
        except Exception:
            return None


def convert_single_file(file_path: str) -> Dict[str, Any]:
    """
    Convenience function to convert a single file.

    Args:
        file_path: Path to the file

    Returns:
        Dictionary with 'content' (markdown) and 'images' (list)
    """
    converter = MarkItDownConverter()
    result = converter.run(file_paths=[file_path])

    content = ""
    if result["documents"]:
        content = "\n\n".join(doc.content for doc in result["documents"])

    return {
        "content": content,
        "images": result["images"],
        "document_count": len(result["documents"]),
    }
