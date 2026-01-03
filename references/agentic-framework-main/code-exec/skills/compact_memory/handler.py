"""
Memory compaction skill handler.
Module: code-exec/skills/compact_memory/handler.py
"""

import json
from typing import Any, Dict, List


def compact_memory(
    artifacts: List[Dict[str, Any]],
    strategy: str = "summarize",
    max_tokens: int = 2000,
    preserve_fields: List[str] = None,
) -> Dict[str, Any]:
    """
    Compact artifacts to reduce token count while preserving key information.

    Args:
        artifacts: List of artifacts to compact
        strategy: Compaction strategy (summarize, truncate, merge, intelligent)
        max_tokens: Maximum tokens in compacted output
        preserve_fields: Fields to always preserve

    Returns:
        Dictionary with compacted artifacts and metadata
    """
    if preserve_fields is None:
        preserve_fields = ["id", "type", "created_at", "provenance_refs"]

    # Estimate original token count
    original_tokens = _estimate_tokens(artifacts)

    # Apply compaction strategy
    if strategy == "summarize":
        compacted = _compact_by_summarize(artifacts, max_tokens, preserve_fields)
    elif strategy == "truncate":
        compacted = _compact_by_truncate(artifacts, max_tokens, preserve_fields)
    elif strategy == "merge":
        compacted = _compact_by_merge(artifacts, max_tokens, preserve_fields)
    elif strategy == "intelligent":
        compacted = _compact_intelligently(artifacts, max_tokens, preserve_fields)
    else:
        # Default to summarize
        compacted = _compact_by_summarize(artifacts, max_tokens, preserve_fields)

    # Estimate compacted token count
    compacted_tokens = _estimate_tokens(compacted)

    # Calculate compression ratio
    compression_ratio = (
        compacted_tokens / original_tokens if original_tokens > 0 else 1.0
    )

    return {
        "compacted_artifacts": compacted,
        "original_count": len(artifacts),
        "compacted_count": len(compacted),
        "original_tokens_estimate": original_tokens,
        "compacted_tokens_estimate": compacted_tokens,
        "compression_ratio": round(compression_ratio, 3),
        "strategy_used": strategy,
        "preserved_artifact_ids": [a.get("id") for a in compacted if "id" in a],
    }


def _estimate_tokens(artifacts: List[Dict[str, Any]]) -> int:
    """
    Estimate token count for artifacts.

    Uses rough heuristic: ~4 characters per token.
    """
    total_chars = len(json.dumps(artifacts, default=str))
    return total_chars // 4


def _compact_by_summarize(
    artifacts: List[Dict[str, Any]], max_tokens: int, preserve_fields: List[str]
) -> List[Dict[str, Any]]:
    """
    Compact by summarizing content while preserving key fields.
    """
    compacted = []

    for artifact in artifacts:
        compacted_artifact: Dict[str, Any] = {}

        # Preserve specified fields
        for field in preserve_fields:
            if field in artifact:
                compacted_artifact[field] = artifact[field]

        # Summarize content
        content = artifact.get("content", {})
        if isinstance(content, dict):
            # Extract key fields and summarize
            summary = _summarize_dict(content)
            compacted_artifact["content_summary"] = summary
        elif isinstance(content, str):
            # Truncate long strings
            compacted_artifact["content_summary"] = content[:200] + "..." if len(content) > 200 else content
        else:
            compacted_artifact["content_summary"] = str(content)[:100]

        # Add metadata
        compacted_artifact["compacted"] = True
        compacted_artifact["original_size"] = len(json.dumps(artifact, default=str))

        compacted.append(compacted_artifact)

    return compacted


def _compact_by_truncate(
    artifacts: List[Dict[str, Any]], max_tokens: int, preserve_fields: List[str]
) -> List[Dict[str, Any]]:
    """
    Compact by truncating to most recent artifacts.
    """
    # Sort by creation time (most recent first)
    sorted_artifacts = sorted(
        artifacts,
        key=lambda a: a.get("created_at", ""),
        reverse=True,
    )

    compacted = []
    current_tokens = 0

    for artifact in sorted_artifacts:
        artifact_tokens = _estimate_tokens([artifact])

        if current_tokens + artifact_tokens <= max_tokens:
            compacted.append(artifact)
            current_tokens += artifact_tokens
        else:
            # Try to add a truncated version
            truncated = _truncate_artifact(artifact, preserve_fields)
            truncated_tokens = _estimate_tokens([truncated])

            if current_tokens + truncated_tokens <= max_tokens:
                compacted.append(truncated)
                current_tokens += truncated_tokens
            else:
                # Stop adding
                break

    return compacted


def _compact_by_merge(
    artifacts: List[Dict[str, Any]], max_tokens: int, preserve_fields: List[str]
) -> List[Dict[str, Any]]:
    """
    Compact by merging similar artifacts.
    """
    # Group by type
    by_type: Dict[str, List[Dict[str, Any]]] = {}

    for artifact in artifacts:
        artifact_type = artifact.get("type", "unknown")
        if artifact_type not in by_type:
            by_type[artifact_type] = []
        by_type[artifact_type].append(artifact)

    compacted = []

    for artifact_type, type_artifacts in by_type.items():
        # Merge artifacts of the same type
        merged = {
            "id": f"merged_{artifact_type}",
            "type": artifact_type,
            "merged_count": len(type_artifacts),
            "merged_from": [a.get("id") for a in type_artifacts],
        }

        # Preserve fields from first artifact
        for field in preserve_fields:
            if field in type_artifacts[0]:
                merged[field] = type_artifacts[0][field]

        # Aggregate content
        contents = [a.get("content", {}) for a in type_artifacts]
        merged["aggregated_content"] = _aggregate_contents(contents)
        merged["compacted"] = True

        compacted.append(merged)

    return compacted


def _compact_intelligently(
    artifacts: List[Dict[str, Any]], max_tokens: int, preserve_fields: List[str]
) -> List[Dict[str, Any]]:
    """
    Intelligent compaction combining multiple strategies.
    """
    # Score artifacts by importance
    scored = [(a, _calculate_importance(a)) for a in artifacts]
    scored.sort(key=lambda x: x[1], reverse=True)

    compacted = []
    current_tokens = 0

    for artifact, score in scored:
        artifact_tokens = _estimate_tokens([artifact])

        if current_tokens + artifact_tokens <= max_tokens:
            # Add full artifact if it fits
            compacted.append(artifact)
            current_tokens += artifact_tokens
        else:
            # Try summarized version
            summarized = _compact_by_summarize([artifact], max_tokens, preserve_fields)[0]
            summarized_tokens = _estimate_tokens([summarized])

            if current_tokens + summarized_tokens <= max_tokens:
                compacted.append(summarized)
                current_tokens += summarized_tokens
            else:
                # No more room
                break

    return compacted


def _truncate_artifact(artifact: Dict[str, Any], preserve_fields: List[str]) -> Dict[str, Any]:
    """Truncate an artifact to preserve only essential fields."""
    truncated: Dict[str, Any] = {"truncated": True}

    for field in preserve_fields:
        if field in artifact:
            truncated[field] = artifact[field]

    return truncated


def _summarize_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """Summarize a dictionary by keeping only key fields."""
    summary = {}

    # Priority fields to keep
    priority_fields = [
        "summary", "text", "claim_text", "verdict", "confidence",
        "tags", "source", "title", "key_points",
    ]

    for field in priority_fields:
        if field in data:
            value = data[field]
            # Truncate long strings
            if isinstance(value, str) and len(value) > 200:
                summary[field] = value[:200] + "..."
            else:
                summary[field] = value

    # Add field count
    summary["_original_fields"] = len(data)

    return summary


def _aggregate_contents(contents: List[Any]) -> Dict[str, Any]:
    """Aggregate multiple content objects."""
    aggregated = {
        "count": len(contents),
        "types": list(set(type(c).__name__ for c in contents)),
    }

    # Extract common fields
    if all(isinstance(c, dict) for c in contents):
        common_keys = set.intersection(*[set(c.keys()) for c in contents if isinstance(c, dict)])
        aggregated["common_fields"] = list(common_keys)

    return aggregated


def _calculate_importance(artifact: Dict[str, Any]) -> float:
    """
    Calculate importance score for an artifact.

    Higher score = more important.
    """
    score = 1.0

    # Boost score for certain types
    artifact_type = artifact.get("type", "")
    if artifact_type in ["claim_verification", "code_patch"]:
        score *= 2.0

    # Boost for high confidence
    content = artifact.get("content", {})
    if isinstance(content, dict):
        confidence = content.get("confidence", 0.0)
        score *= (1.0 + confidence)

    # Boost for recent artifacts
    # (Assumes created_at is ISO format string - would need proper parsing)
    # For now, just check if it exists
    if "created_at" in artifact:
        score *= 1.5

    return score
