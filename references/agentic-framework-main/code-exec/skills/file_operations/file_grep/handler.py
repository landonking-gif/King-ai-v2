"""
File Grep Skill Handler.

Module: code-exec/skills/prepackaged/file_operations/file_grep/handler.py

Search for text patterns in files using regex. Mirrors Claude Code's Grep tool.
"""

from typing import Any, Dict, List, Optional
from pathlib import Path
import re


def grep_files(
    pattern: str,
    path: Optional[str] = None,
    glob: Optional[str] = None,
    case_insensitive: bool = False,
    context_lines: int = 0,
    max_results: int = 100,
    output_mode: str = "files_with_matches",
) -> Dict[str, Any]:
    """
    Search for text patterns in files.

    Args:
        pattern: Regex pattern to search for
        path: File or directory to search in
        glob: Filter files by glob pattern
        case_insensitive: Case insensitive search
        context_lines: Number of context lines
        max_results: Maximum matches to return
        output_mode: Output mode (content, files_with_matches, count)

    Returns:
        Dictionary with search results
    """
    # Determine base path
    base_path = Path(path) if path else Path.cwd()

    if not base_path.exists():
        return {
            "matches": [],
            "files_with_matches": [],
            "total_matches": 0,
            "files_searched": 0,
            "truncated": False,
        }

    # Compile regex
    flags = re.IGNORECASE if case_insensitive else 0
    try:
        regex = re.compile(pattern, flags)
    except re.error as e:
        return {
            "matches": [],
            "files_with_matches": [],
            "total_matches": 0,
            "files_searched": 0,
            "truncated": False,
        }

    matches: List[Dict[str, Any]] = []
    files_with_matches: List[str] = []
    total_matches = 0
    files_searched = 0

    # Get files to search
    if base_path.is_file():
        files_to_search = [base_path]
    else:
        glob_pattern = glob if glob else "**/*"
        files_to_search = [f for f in base_path.glob(glob_pattern) if f.is_file()]

    for file_path in files_to_search:
        # Skip binary files and very large files
        if file_path.stat().st_size > 10 * 1024 * 1024:  # 10MB limit
            continue

        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        except (PermissionError, OSError):
            continue

        files_searched += 1
        file_has_match = False

        for i, line in enumerate(lines):
            if regex.search(line):
                file_has_match = True
                total_matches += 1

                if output_mode == "content" and len(matches) < max_results:
                    # Get context lines
                    ctx_before = []
                    ctx_after = []

                    if context_lines > 0:
                        start_ctx = max(0, i - context_lines)
                        ctx_before = [lines[j].rstrip() for j in range(start_ctx, i)]

                        end_ctx = min(len(lines), i + context_lines + 1)
                        ctx_after = [lines[j].rstrip() for j in range(i + 1, end_ctx)]

                    matches.append({
                        "file": str(file_path.absolute()),
                        "line_number": i + 1,
                        "content": line.rstrip(),
                        "context_before": ctx_before,
                        "context_after": ctx_after,
                    })

        if file_has_match and str(file_path.absolute()) not in files_with_matches:
            files_with_matches.append(str(file_path.absolute()))

    truncated = total_matches > max_results or len(files_with_matches) > max_results

    return {
        "matches": matches[:max_results] if output_mode == "content" else [],
        "files_with_matches": files_with_matches[:max_results],
        "total_matches": total_matches,
        "files_searched": files_searched,
        "truncated": truncated,
    }
