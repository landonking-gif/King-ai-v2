"""
File Read Skill Handler.

Module: code-exec/skills/prepackaged/file_operations/file_read/handler.py

Reads file contents with optional line range and line number formatting.
Mirrors Claude Code's Read tool behavior.
"""

from typing import Any, Dict, Optional
from pathlib import Path


def read_file(
    file_path: str,
    offset: Optional[int] = None,
    limit: int = 2000,
) -> Dict[str, Any]:
    """
    Read file contents with optional line range.

    Args:
        file_path: Absolute path to the file to read
        offset: Starting line number (1-based)
        limit: Maximum number of lines to read (default 2000)

    Returns:
        Dictionary with file contents and metadata
    """
    path = Path(file_path)

    # Check if file exists
    if not path.exists():
        return {
            "content": f"File not found: {file_path}",
            "file_exists": False,
            "total_lines": 0,
            "lines_read": 0,
            "file_size_bytes": 0,
            "truncated": False,
        }

    # Check if it's a file (not directory)
    if not path.is_file():
        return {
            "content": f"Not a file: {file_path}",
            "file_exists": False,
            "total_lines": 0,
            "lines_read": 0,
            "file_size_bytes": 0,
            "truncated": False,
        }

    try:
        # Read file with error handling for encoding issues
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        total_lines = len(lines)
        file_size = path.stat().st_size

        # Calculate line range
        start_idx = (offset - 1) if offset else 0
        start_idx = max(0, min(start_idx, total_lines))
        end_idx = min(start_idx + limit, total_lines)

        selected_lines = lines[start_idx:end_idx]
        truncated = end_idx < total_lines

        # Format with line numbers (like cat -n)
        # Using format: "    N->content" where N is right-aligned
        numbered_content = ""
        max_line_width = len(str(end_idx)) if end_idx > 0 else 1

        for i, line in enumerate(selected_lines, start=start_idx + 1):
            # Truncate very long lines (>2000 chars)
            if len(line) > 2000:
                line = line[:2000] + "... [truncated]\n"

            # Format: right-aligned line number + arrow + content
            numbered_content += f"{i:>{max_line_width + 1}}->{line}"

        return {
            "content": numbered_content,
            "file_exists": True,
            "total_lines": total_lines,
            "lines_read": len(selected_lines),
            "file_size_bytes": file_size,
            "truncated": truncated,
        }

    except PermissionError:
        return {
            "content": f"Permission denied: {file_path}",
            "file_exists": True,
            "total_lines": 0,
            "lines_read": 0,
            "file_size_bytes": 0,
            "truncated": False,
        }
    except Exception as e:
        return {
            "content": f"Error reading file: {str(e)}",
            "file_exists": True,
            "total_lines": 0,
            "lines_read": 0,
            "file_size_bytes": 0,
            "truncated": False,
        }
