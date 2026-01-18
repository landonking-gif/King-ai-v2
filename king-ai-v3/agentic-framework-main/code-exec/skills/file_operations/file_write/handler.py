"""
File Write Skill Handler.

Module: code-exec/skills/prepackaged/file_operations/file_write/handler.py

Write or create files. Mirrors Claude Code's Write tool behavior.
"""

from typing import Any, Dict, List
from pathlib import Path


def write_file(
    file_path: str,
    content: str,
    create_directories: bool = False,
) -> Dict[str, Any]:
    """
    Write content to a file.

    Args:
        file_path: Absolute path to the file
        content: Content to write
        create_directories: Create parent directories if needed

    Returns:
        Dictionary with write result
    """
    path = Path(file_path)
    created_dirs: List[str] = []

    try:
        # Create parent directories if requested
        if create_directories and not path.parent.exists():
            # Track which directories we create
            dirs_to_create = []
            current = path.parent

            while not current.exists():
                dirs_to_create.append(current)
                current = current.parent

            # Create directories
            path.parent.mkdir(parents=True, exist_ok=True)
            created_dirs = [str(d) for d in reversed(dirs_to_create)]

        elif not path.parent.exists():
            return {
                "success": False,
                "file_path": file_path,
                "bytes_written": 0,
                "created_directories": [],
                "error": f"Parent directory does not exist: {path.parent}",
            }

        # Write the file
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

        bytes_written = path.stat().st_size

        return {
            "success": True,
            "file_path": str(path.absolute()),
            "bytes_written": bytes_written,
            "created_directories": created_dirs,
        }

    except PermissionError:
        return {
            "success": False,
            "file_path": file_path,
            "bytes_written": 0,
            "created_directories": created_dirs,
            "error": f"Permission denied: {file_path}",
        }
    except Exception as e:
        return {
            "success": False,
            "file_path": file_path,
            "bytes_written": 0,
            "created_directories": created_dirs,
            "error": str(e),
        }
