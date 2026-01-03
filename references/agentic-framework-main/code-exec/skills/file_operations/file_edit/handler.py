"""
File Edit Skill Handler.

Module: code-exec/skills/prepackaged/file_operations/file_edit/handler.py

Perform surgical text replacement in files. Mirrors Claude Code's Edit tool.
"""

from typing import Any, Dict
from pathlib import Path
import difflib


def edit_file(
    file_path: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
) -> Dict[str, Any]:
    """
    Edit a file by replacing text.

    Args:
        file_path: Path to file to edit
        old_string: Text to find and replace
        new_string: Replacement text
        replace_all: Replace all occurrences

    Returns:
        Dictionary with edit result
    """
    path = Path(file_path)

    # Check if file exists
    if not path.exists():
        return {
            "success": False,
            "file_path": file_path,
            "replacements_made": 0,
            "error": f"File not found: {file_path}",
        }

    if not path.is_file():
        return {
            "success": False,
            "file_path": file_path,
            "replacements_made": 0,
            "error": f"Not a file: {file_path}",
        }

    try:
        # Read current content
        with open(path, "r", encoding="utf-8") as f:
            original_content = f.read()

        # Check if old_string exists
        occurrences = original_content.count(old_string)

        if occurrences == 0:
            return {
                "success": False,
                "file_path": file_path,
                "replacements_made": 0,
                "error": f"String not found in file: '{old_string[:50]}...' (truncated)",
            }

        # Check uniqueness if not replace_all
        if not replace_all and occurrences > 1:
            return {
                "success": False,
                "file_path": file_path,
                "replacements_made": 0,
                "error": f"String found {occurrences} times. Use replace_all=true or provide more context for uniqueness.",
            }

        # Perform replacement
        if replace_all:
            new_content = original_content.replace(old_string, new_string)
            replacements_made = occurrences
        else:
            new_content = original_content.replace(old_string, new_string, 1)
            replacements_made = 1

        # Generate diff preview
        original_lines = original_content.splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)

        diff = difflib.unified_diff(
            original_lines,
            new_lines,
            fromfile=f"a/{path.name}",
            tofile=f"b/{path.name}",
            lineterm="",
        )
        diff_preview = "".join(list(diff)[:50])  # Limit diff preview

        # Write new content
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_content)

        return {
            "success": True,
            "file_path": str(path.absolute()),
            "replacements_made": replacements_made,
            "diff_preview": diff_preview,
        }

    except PermissionError:
        return {
            "success": False,
            "file_path": file_path,
            "replacements_made": 0,
            "error": f"Permission denied: {file_path}",
        }
    except Exception as e:
        return {
            "success": False,
            "file_path": file_path,
            "replacements_made": 0,
            "error": str(e),
        }
