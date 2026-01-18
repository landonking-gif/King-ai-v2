"""
Jupyter Notebook Edit Skill Handler.

Module: code-exec/skills/prepackaged/code_execution/notebook_edit/handler.py

Edit Jupyter notebook cells. Mirrors Claude Code's NotebookEdit tool.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional


def edit_notebook(
    notebook_path: str,
    new_source: str,
    cell_index: Optional[int] = None,
    cell_type: str = "code",
    edit_mode: str = "replace",
) -> Dict[str, Any]:
    """
    Edit a Jupyter notebook cell.

    Args:
        notebook_path: Path to the notebook file
        new_source: New source content for the cell
        cell_index: Index of cell to edit (0-based)
        cell_type: Type of cell (code or markdown)
        edit_mode: replace, insert, or delete

    Returns:
        Dictionary with edit results
    """
    path = Path(notebook_path)

    # Validate file exists
    if not path.exists():
        return {
            "success": False,
            "error": f"Notebook not found: {notebook_path}",
            "notebook_path": notebook_path,
        }

    if not path.suffix == ".ipynb":
        return {
            "success": False,
            "error": "File is not a Jupyter notebook (.ipynb)",
            "notebook_path": notebook_path,
        }

    try:
        # Read notebook
        with open(path, "r", encoding="utf-8") as f:
            notebook = json.load(f)

        cells = notebook.get("cells", [])

        # Create new cell structure
        new_cell = {
            "cell_type": cell_type,
            "metadata": {},
            "source": new_source.split("\n") if new_source else [],
        }

        if cell_type == "code":
            new_cell["execution_count"] = None
            new_cell["outputs"] = []

        # Handle edit modes
        if edit_mode == "insert":
            # Insert at position (or end if no index)
            insert_idx = cell_index if cell_index is not None else len(cells)
            insert_idx = min(insert_idx, len(cells))
            cells.insert(insert_idx, new_cell)
            affected_idx = insert_idx

        elif edit_mode == "delete":
            # Delete cell at index
            if cell_index is None or cell_index >= len(cells):
                return {
                    "success": False,
                    "error": f"Invalid cell index for delete: {cell_index}",
                    "notebook_path": notebook_path,
                }
            del cells[cell_index]
            affected_idx = cell_index

        else:  # replace
            if cell_index is None:
                cell_index = 0
            if cell_index >= len(cells):
                return {
                    "success": False,
                    "error": f"Cell index {cell_index} out of range (notebook has {len(cells)} cells)",
                    "notebook_path": notebook_path,
                }
            cells[cell_index] = new_cell
            affected_idx = cell_index

        notebook["cells"] = cells

        # Write notebook back
        with open(path, "w", encoding="utf-8") as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)

        return {
            "success": True,
            "notebook_path": str(path.absolute()),
            "cell_index": affected_idx,
            "total_cells": len(cells),
        }

    except json.JSONDecodeError as e:
        return {
            "success": False,
            "error": f"Invalid notebook JSON: {e}",
            "notebook_path": notebook_path,
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "notebook_path": notebook_path,
        }
