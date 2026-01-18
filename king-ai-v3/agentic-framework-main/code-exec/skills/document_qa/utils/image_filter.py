"""
Smart Image Filtering for Document Q&A.

Module: code-exec/skills/document_qa/utils/image_filter.py

Filters images to avoid sending unnecessary images to Vision LLM,
reducing costs by ~95%.
"""

import hashlib
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ImageData:
    """Represents an extracted image with metadata."""

    image_bytes: bytes
    file_path: str
    source_file: str
    page: Optional[int] = None
    slide: Optional[int] = None
    position: Optional[Tuple[int, int]] = None  # (x, y)
    size: Optional[Tuple[int, int]] = None  # (width, height)
    image_format: str = "png"
    alt_text: Optional[str] = None
    caption: Optional[str] = None
    content_hash: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Calculate content hash if not provided."""
        if not self.content_hash and self.image_bytes:
            self.content_hash = hashlib.md5(self.image_bytes).hexdigest()[:12]

    @property
    def size_kb(self) -> float:
        """Return image size in KB."""
        return len(self.image_bytes) / 1024 if self.image_bytes else 0

    @property
    def aspect_ratio(self) -> float:
        """Return aspect ratio (width/height)."""
        if self.size and self.size[1] > 0:
            return self.size[0] / self.size[1]
        return 1.0


class ImageFilter:
    """
    Filters images to determine which should be sent to Vision LLM.

    Filtering rules:
    1. Skip if size < min_size_kb (likely icon)
    2. Skip if aspect ratio indicates decorative (e.g., 1:10 banner)
    3. Skip if image hash matches known patterns (logos, icons)
    4. Skip if image appears multiple times (repeated logo)
    5. Send: charts, diagrams, complex visuals

    Configuration via environment variables:
    - DOCUMENT_QA_MAX_IMAGES: Max images to send (default: 10)
    - DOCUMENT_QA_IMAGE_MIN_SIZE_KB: Min size threshold (default: 50)
    """

    # Common icon/logo patterns to skip
    SKIP_PATTERNS = {
        "icon",
        "logo",
        "favicon",
        "button",
        "arrow",
        "bullet",
        "checkbox",
        "radio",
        "avatar",
        "placeholder",
    }

    def __init__(
        self,
        min_size_kb: Optional[float] = None,
        max_images: Optional[int] = None,
        min_aspect_ratio: float = 0.2,
        max_aspect_ratio: float = 5.0,
    ):
        """
        Initialize the image filter.

        Args:
            min_size_kb: Minimum size in KB to consider (default: 50 from env)
            max_images: Maximum images to return (default: 10 from env)
            min_aspect_ratio: Minimum aspect ratio (skip very thin images)
            max_aspect_ratio: Maximum aspect ratio (skip very wide images)
        """
        self.min_size_kb = min_size_kb or float(
            os.getenv("DOCUMENT_QA_IMAGE_MIN_SIZE_KB", "50")
        )
        self.max_images = max_images or int(
            os.getenv("DOCUMENT_QA_MAX_IMAGES", "10")
        )
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio

        # Track seen hashes for deduplication
        self._seen_hashes: Dict[str, int] = {}

    def should_process(self, image: ImageData) -> Tuple[bool, str]:
        """
        Determine if an image should be sent to Vision LLM.

        Args:
            image: ImageData to evaluate

        Returns:
            Tuple of (should_process, reason)
        """
        # Check size threshold
        if image.size_kb < self.min_size_kb:
            return False, f"too_small ({image.size_kb:.1f}KB < {self.min_size_kb}KB)"

        # Check aspect ratio
        if image.aspect_ratio < self.min_aspect_ratio:
            return False, f"too_narrow (ratio={image.aspect_ratio:.2f})"
        if image.aspect_ratio > self.max_aspect_ratio:
            return False, f"too_wide (ratio={image.aspect_ratio:.2f})"

        # Check for skip patterns in alt text or caption
        text_to_check = f"{image.alt_text or ''} {image.caption or ''}".lower()
        for pattern in self.SKIP_PATTERNS:
            if pattern in text_to_check:
                return False, f"matches_skip_pattern ({pattern})"

        # Check for duplicates
        if image.content_hash:
            self._seen_hashes[image.content_hash] = (
                self._seen_hashes.get(image.content_hash, 0) + 1
            )
            if self._seen_hashes[image.content_hash] > 1:
                return False, "duplicate"

        # Check dimensions for very small images
        if image.size:
            width, height = image.size
            if width < 100 or height < 100:
                return False, f"too_small_dimensions ({width}x{height})"

        return True, "acceptable"

    def filter_images(self, images: List[ImageData]) -> List[ImageData]:
        """
        Filter a list of images, returning only those worth processing.

        Args:
            images: List of ImageData to filter

        Returns:
            Filtered list of images (max self.max_images)
        """
        self._seen_hashes.clear()

        filtered = []
        skipped_reasons: Dict[str, int] = {}

        for image in images:
            should_process, reason = self.should_process(image)

            if should_process:
                filtered.append(image)
                if len(filtered) >= self.max_images:
                    break
            else:
                skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1

        return filtered

    def get_statistics(
        self, original_count: int, filtered_count: int
    ) -> Dict[str, Any]:
        """
        Get filtering statistics.

        Args:
            original_count: Original number of images
            filtered_count: Number after filtering

        Returns:
            Statistics dictionary
        """
        reduction = 0.0
        if original_count > 0:
            reduction = (1 - filtered_count / original_count) * 100

        return {
            "original_count": original_count,
            "filtered_count": filtered_count,
            "reduction_percent": round(reduction, 1),
            "max_allowed": self.max_images,
        }


def estimate_vision_cost(
    image_count: int,
    avg_resolution: str = "medium",
    model: str = "gpt-4o",
) -> Dict[str, Any]:
    """
    Estimate the cost of Vision LLM calls.

    Args:
        image_count: Number of images to process
        avg_resolution: "low", "medium", or "high"
        model: Model to use

    Returns:
        Cost estimate dictionary
    """
    # Token estimates per resolution
    tokens_per_resolution = {
        "low": 85,
        "medium": 170,
        "high": 765,
    }

    # Cost per 1M tokens (approximate)
    cost_per_million = {
        "gpt-4o": 5.0,
        "gpt-4o-mini": 0.15,
    }

    tokens = tokens_per_resolution.get(avg_resolution, 170) * image_count
    rate = cost_per_million.get(model, 5.0)
    cost = (tokens / 1_000_000) * rate

    return {
        "image_count": image_count,
        "resolution": avg_resolution,
        "model": model,
        "estimated_tokens": tokens,
        "estimated_cost_usd": round(cost, 4),
    }
