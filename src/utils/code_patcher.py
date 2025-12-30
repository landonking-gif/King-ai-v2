
import os
import shutil
from datetime import datetime
import difflib

class CodePatcher:
    """
    Safely applies code modifications to the codebase.
    Always creates backups and supports dry-runs.
    """
    
    @staticmethod
    def create_backup(file_path: str) -> str:
        """Creates a timestamped backup of a file."""
        if not os.path.exists(file_path):
            return ""
        
        backup_dir = os.path.join(os.path.dirname(file_path), ".backups")
        os.makedirs(backup_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(backup_dir, f"{os.path.basename(file_path)}.{timestamp}.bak")
        shutil.copy2(file_path, backup_path)
        return backup_path

    @staticmethod
    def apply_patch(file_path: str, new_content: str, is_dry_run: bool = False) -> dict:
        """
        Applies new content to a file.
        :return: Dict with success status and diff.
        """
        if not os.path.exists(file_path):
            return {"success": False, "error": "File not found"}

        with open(file_path, 'r', encoding='utf-8') as f:
            old_content = f.read()

        # Generate Diff
        diff = list(difflib.unified_diff(
            old_content.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}"
        ))

        if not diff:
            return {"success": True, "message": "No changes needed", "diff": ""}

        if is_dry_run:
            return {"success": True, "message": "Dry run successful", "diff": "".join(diff)}

        # Apply Change
        try:
            CodePatcher.create_backup(file_path)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            return {"success": True, "message": "Patch applied", "diff": "".join(diff)}
        except Exception as e:
            return {"success": False, "error": f"Write failed: {str(e)}"}

    @staticmethod
    def rollback(file_path: str, backup_path: str) -> bool:
        """Restores a file from a backup."""
        try:
            shutil.copy2(backup_path, file_path)
            return True
        except Exception:
            return False
