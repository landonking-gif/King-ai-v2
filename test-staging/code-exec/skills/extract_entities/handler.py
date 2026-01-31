"""
Entity extraction skill handler.
Module: code-exec/skills/extract_entities/handler.py
"""

import re
from typing import Any, Dict, List


def extract_entities(
    text: str, entity_types: List[str] = None, min_confidence: float = 0.5
) -> Dict[str, Any]:
    """
    Extract named entities from text using pattern matching.

    This is a simple regex-based NER. For production, use spaCy,
    transformers, or a dedicated NER model.

    Args:
        text: Text to extract entities from
        entity_types: Types of entities to extract (if empty, extract all)
        min_confidence: Minimum confidence threshold

    Returns:
        Dictionary with extracted entities and metadata
    """
    if entity_types is None:
        entity_types = []

    all_entities = []
    entity_counts: Dict[str, int] = {}

    # Define entity patterns and extractors
    extractors = {
        "EMAIL": (_extract_emails, 0.9),
        "PHONE": (_extract_phones, 0.8),
        "URL": (_extract_urls, 0.95),
        "DATE": (_extract_dates, 0.7),
        "MONEY": (_extract_money, 0.8),
        "PERSON": (_extract_persons, 0.6),
        "ORGANIZATION": (_extract_organizations, 0.5),
        "LOCATION": (_extract_locations, 0.5),
    }

    # Determine which entity types to extract
    types_to_extract = entity_types if entity_types else list(extractors.keys())

    # Extract entities for each type
    for entity_type in types_to_extract:
        if entity_type not in extractors:
            continue

        extractor_func, base_confidence = extractors[entity_type]
        entities = extractor_func(text)

        for entity in entities:
            # Assign confidence
            confidence = base_confidence

            # Filter by confidence
            if confidence < min_confidence:
                continue

            all_entities.append(
                {
                    "text": entity["text"],
                    "type": entity_type,
                    "confidence": confidence,
                    "start_pos": entity["start"],
                    "end_pos": entity["end"],
                }
            )

            # Update counts
            entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1

    # Sort entities by position
    all_entities.sort(key=lambda e: e["start_pos"])

    return {
        "entities": all_entities,
        "total_entities": len(all_entities),
        "entity_counts": entity_counts,
    }


def _extract_emails(text: str) -> List[Dict[str, Any]]:
    """Extract email addresses."""
    pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    matches = []
    for match in re.finditer(pattern, text):
        matches.append({"text": match.group(), "start": match.start(), "end": match.end()})
    return matches


def _extract_phones(text: str) -> List[Dict[str, Any]]:
    """Extract phone numbers."""
    # Simple pattern for US/international phone numbers
    pattern = r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    matches = []
    for match in re.finditer(pattern, text):
        matches.append({"text": match.group(), "start": match.start(), "end": match.end()})
    return matches


def _extract_urls(text: str) -> List[Dict[str, Any]]:
    """Extract URLs."""
    pattern = r"https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*)"
    matches = []
    for match in re.finditer(pattern, text):
        matches.append({"text": match.group(), "start": match.start(), "end": match.end()})
    return matches


def _extract_dates(text: str) -> List[Dict[str, Any]]:
    """Extract dates."""
    # Simple date patterns
    patterns = [
        r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",  # MM/DD/YYYY
        r"\b\d{4}-\d{2}-\d{2}\b",  # YYYY-MM-DD
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b",
    ]

    matches = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            matches.append({"text": match.group(), "start": match.start(), "end": match.end()})
    return matches


def _extract_money(text: str) -> List[Dict[str, Any]]:
    """Extract monetary amounts."""
    pattern = r"(?:\$|USD|EUR|GBP|£|€)\s*\d+(?:,\d{3})*(?:\.\d{2})?"
    matches = []
    for match in re.finditer(pattern, text):
        matches.append({"text": match.group(), "start": match.start(), "end": match.end()})
    return matches


def _extract_persons(text: str) -> List[Dict[str, Any]]:
    """Extract person names (simple capitalized words pattern)."""
    # Very simple pattern: Title + Capitalized words
    pattern = r"\b(?:Mr\.|Mrs\.|Ms\.|Dr\.|Prof\.)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b"
    matches = []
    for match in re.finditer(pattern, text):
        matches.append({"text": match.group(), "start": match.start(), "end": match.end()})

    # Also extract capitalized sequences (2-3 words)
    pattern2 = r"\b[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b"
    for match in re.finditer(pattern2, text):
        # Avoid duplicates
        if not any(m["start"] == match.start() for m in matches):
            matches.append({"text": match.group(), "start": match.start(), "end": match.end()})

    return matches


def _extract_organizations(text: str) -> List[Dict[str, Any]]:
    """Extract organization names."""
    # Simple pattern: Capitalized words + Inc/LLC/Corp/Ltd
    pattern = r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Inc\.?|LLC|Corp\.?|Ltd\.?|Company|Co\.)\b"
    matches = []
    for match in re.finditer(pattern, text):
        matches.append({"text": match.group(), "start": match.start(), "end": match.end()})
    return matches


def _extract_locations(text: str) -> List[Dict[str, Any]]:
    """Extract location names."""
    # Simple pattern: City, State/Country abbreviations
    pattern = r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?,\s+[A-Z]{2,}\b"
    matches = []
    for match in re.finditer(pattern, text):
        matches.append({"text": match.group(), "start": match.start(), "end": match.end()})
    return matches
