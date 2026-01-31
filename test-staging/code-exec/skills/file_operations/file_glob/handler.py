"""
File Glob Skill Handler.

Module: code-exec/skills/prepackaged/file_operations/file_glob/handler.py

Find files matching glob patterns. Mirrors Claude Code's Glob tool behavior.
"""

from typing import Any, Dict, List, Optional
from pathlib import Path
import os


def glob_files(
    pattern: str,
    path: Optional[str] = None,
    max_results: int = 100,
) -> Dict[str, Any]:
    """
    Find files matching a glob pattern.

    Args:
        pattern: Glob pattern (e.g., **/*.py, src/**/*.ts)
        path: Base directory to search in
        max_results: Maximum number of results

    Returns:
        Dictionary with matching files
    """
    # Determine base path
    base_path = Path(path) if path else Path.cwd()

    if not base_path.exists():
        return {
            "files": [],
            "total_matches": 0,
            "truncated": False,
        }

    try:
        # Use glob to find matching files
        matches: List[str] = []
        all_matches = list(base_path.glob(pattern))

        # Sort by modification time (most recent first)
        all_matches.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)

        total_matches = len(all_matches)
        truncated = total_matches > max_results

        # Limit results
        for match in all_matches[:max_results]:
            # Return absolute paths
            matches.append(str(match.absolute()))

        return {
            "files": matches,
            "total_matches": total_matches,
            "truncated": truncated,
        }

    except PermissionError:
        return {
            "files": [],
            "total_matches": 0,
            "truncated": False,
        }
    except Exception as e:
        return {
            "files": [],
            "total_matches": 0,
            "truncated": False,
        }
