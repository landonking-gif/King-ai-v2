"""
Code Patcher - Safe code modification with diff generation and rollback.
Handles atomic file operations with backup capabilities.
"""

import os
import shutil
import difflib
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4

try:
    from src.utils.ast_parser import ASTParser
except ImportError:
    ASTParser = None

try:
    from src.utils.logging import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class PatchStatus(Enum):
    """Status of a code patch."""
    PENDING = "pending"
    VALIDATED = "validated"
    APPLIED = "applied"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class Patch:
    """Represents a single file patch."""
    file_path: str
    old_content: str
    new_content: str


@dataclass
class CodePatch:
    """Represents a single file patch with metadata."""
    file_path: str
    original_content: str
    new_content: str
    description: str = ""
    status: PatchStatus = field(default=PatchStatus.PENDING)
    applied_at: Optional[datetime] = None
    error: Optional[str] = None
    
    @property
    def diff(self) -> str:
        """Generate unified diff between original and new content."""
        original_lines = self.original_content.splitlines(keepends=True)
        new_lines = self.new_content.splitlines(keepends=True)
        diff_result = difflib.unified_diff(
            original_lines, 
            new_lines,
            fromfile=f"a/{self.file_path}",
            tofile=f"b/{self.file_path}"
        )
        return ''.join(diff_result)
    
    @property
    def stats(self) -> Dict[str, int]:
        """Return statistics about the patch."""
        original_lines = set(self.original_content.splitlines())
        new_lines = set(self.new_content.splitlines())
        
        additions = len(new_lines - original_lines)
        deletions = len(original_lines - new_lines)
        
        return {
            "additions": additions,
            "deletions": deletions,
            "files": 1
        }


@dataclass
class PatchSet:
    """Collection of patches to apply."""
    patches: List[CodePatch] = field(default_factory=list)
    id: str = field(default_factory=lambda: str(uuid4()))
    description: str = ""
    status: PatchStatus = field(default=PatchStatus.PENDING)
    created_at: datetime = field(default_factory=datetime.now)
    applied_at: Optional[datetime] = None
    backup_dir: Optional[str] = None
    
    def add_patch(self, patch_or_path, old_content: str = None, new_content: str = None):
        """Add a patch to the set.
        
        Can accept either:
            - A CodePatch object directly
            - file_path, old_content, new_content as separate args
        """
        if isinstance(patch_or_path, CodePatch):
            self.patches.append(patch_or_path)
        else:
            self.patches.append(CodePatch(patch_or_path, old_content, new_content))
    
    @property
    def total_stats(self) -> Dict[str, int]:
        """Aggregate statistics for all patches."""
        total = {"additions": 0, "deletions": 0, "files": len(self.patches)}
        for patch in self.patches:
            stats = patch.stats
            total["additions"] += stats["additions"]
            total["deletions"] += stats["deletions"]
        return total


class CodePatcher:
    """
    Handles safe code modifications with validation and rollback.
    """
    
    def __init__(self, project_root: str, backup_dir: str = None):
        """
        Initialize the patcher.
        
        Args:
            project_root: Root directory of the project
            backup_dir: Directory for backups (default: .king_ai_backups)
        """
        self.project_root = Path(project_root)
        self.backup_dir = Path(backup_dir) if backup_dir else self.project_root / ".king_ai_backups"
        self.parser = ASTParser()
        
        # Ensure backup directory exists
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
    def create_patch(
        self,
        file_path: str,
        new_content: str,
        description: str = ""
    ) -> CodePatch:
        """
        Create a patch for a file.
        
        Args:
            file_path: Relative path to the file
            new_content: New content for the file
            description: Description of the change
            
        Returns:
            CodePatch object
        """
        full_path = self.project_root / file_path
        
        # Read original content
        if full_path.exists():
            original_content = full_path.read_text(encoding='utf-8')
        else:
            original_content = ""
        
        return CodePatch(
            file_path=file_path,
            original_content=original_content,
            new_content=new_content,
            description=description
        )
    
    def create_patch_from_function(
        self,
        file_path: str,
        function_name: str,
        new_function_code: str
    ) -> CodePatch:
        """
        Create a patch that replaces a specific function.
        
        Args:
            file_path: Path to the file
            function_name: Name of the function to replace
            new_function_code: New function implementation
            
        Returns:
            CodePatch object
        """
        full_path = self.project_root / file_path
        original_content = full_path.read_text(encoding='utf-8')
        
        # Parse the file to find the function
        structure = self.parser.parse_source(original_content, file_path)
        func = structure.get_function(function_name)
        
        if not func:
            raise ValueError(f"Function '{function_name}' not found in {file_path}")
        
        # Replace the function
        lines = original_content.splitlines(keepends=True)
        
        # Get the function lines
        before = lines[:func.lineno - 1]
        after = lines[func.end_lineno:]
        
        # Ensure new function has proper newline
        if not new_function_code.endswith('\n'):
            new_function_code += '\n'
        
        new_content = ''.join(before) + new_function_code + ''.join(after)
        
        return CodePatch(
            file_path=file_path,
            original_content=original_content,
            new_content=new_content,
            description=f"Replace function: {function_name}"
        )
    
    def validate_patch(self, patch: CodePatch) -> Tuple[bool, List[str]]:
        """
        Validate a patch before applying.
        
        Args:
            patch: The patch to validate
            
        Returns:
            Tuple of (is_valid, list of errors)
        """
        errors = []
        
        # Check syntax
        if patch.file_path.endswith('.py'):
            is_valid, error = self.parser.validate_syntax(patch.new_content)
            if not is_valid:
                errors.append(f"Syntax error: {error}")
        
        # Check for dangerous patterns
        dangerous_patterns = [
            ('os.system(', "Direct system command execution"),
            ('subprocess.call(', "Subprocess call without validation"),
            ('eval(', "Eval usage is dangerous"),
            ('exec(', "Exec usage is dangerous"),
            ('__import__(', "Dynamic import"),
            ('open(', None),  # Check if writing to sensitive files
        ]
        
        for pattern, message in dangerous_patterns:
            if pattern in patch.new_content and pattern not in patch.original_content:
                if message:
                    errors.append(f"Dangerous pattern: {message}")
        
        # Check file path safety
        if '..' in patch.file_path:
            errors.append("Path traversal detected in file path")
        
        sensitive_files = ['settings.py', '.env', 'secrets', 'credentials']
        if any(s in patch.file_path.lower() for s in sensitive_files):
            errors.append("Modification of sensitive file requires extra review")
        
        patch.status = PatchStatus.VALIDATED if not errors else PatchStatus.PENDING
        return len(errors) == 0, errors
    
    def validate_patchset(self, patchset: PatchSet) -> Tuple[bool, Dict[str, List[str]]]:
        """
        Validate all patches in a set.
        
        Returns:
            Tuple of (all_valid, dict of file_path -> errors)
        """
        all_errors = {}
        all_valid = True
        
        for patch in patchset.patches:
            valid, errors = self.validate_patch(patch)
            if not valid:
                all_valid = False
                all_errors[patch.file_path] = errors
        
        patchset.status = PatchStatus.VALIDATED if all_valid else PatchStatus.PENDING
        return all_valid, all_errors
    
    def apply_patch(self, patch: CodePatch, create_backup: bool = True) -> bool:
        """
        Apply a single patch.
        
        Args:
            patch: The patch to apply
            create_backup: Whether to backup original file
            
        Returns:
            Success status
        """
        try:
            full_path = self.project_root / patch.file_path
            
            # Create backup
            if create_backup and full_path.exists():
                backup_path = self._create_backup(patch.file_path)
                logger.info(f"Created backup: {backup_path}")
            
            # Create parent directories if needed
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write new content
            full_path.write_text(patch.new_content, encoding='utf-8')
            
            patch.status = PatchStatus.APPLIED
            patch.applied_at = datetime.now()
            
            logger.info(f"Applied patch to {patch.file_path}", stats=patch.stats)
            return True
            
        except Exception as e:
            patch.status = PatchStatus.FAILED
            patch.error = str(e)
            logger.error(f"Failed to apply patch: {e}")
            return False
    
    def apply_patchset(self, patchset: PatchSet) -> bool:
        """
        Apply all patches in a set atomically.
        If any patch fails, rollback all applied patches.
        
        Args:
            patchset: The patchset to apply
            
        Returns:
            Success status
        """
        # Create a dedicated backup directory for this patchset
        backup_subdir = self.backup_dir / f"patchset_{patchset.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        backup_subdir.mkdir(parents=True, exist_ok=True)
        patchset.backup_dir = str(backup_subdir)
        
        applied_patches = []
        
        try:
            for patch in patchset.patches:
                # Backup to patchset-specific directory
                full_path = self.project_root / patch.file_path
                if full_path.exists():
                    backup_path = backup_subdir / patch.file_path
                    backup_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(full_path, backup_path)
                
                # Apply the patch
                success = self.apply_patch(patch, create_backup=False)
                
                if not success:
                    raise RuntimeError(f"Failed to apply patch to {patch.file_path}: {patch.error}")
                
                applied_patches.append(patch)
            
            patchset.status = PatchStatus.APPLIED
            patchset.applied_at = datetime.now()
            
            logger.info(f"Applied patchset {patchset.id}", stats=patchset.total_stats)
            return True
            
        except Exception as e:
            logger.error(f"Patchset failed, rolling back: {e}")
            
            # Rollback all applied patches
            for patch in reversed(applied_patches):
                self._rollback_patch(patch, backup_subdir)
            
            patchset.status = PatchStatus.FAILED
            return False
    
    def _create_backup(self, file_path: str) -> Path:
        """Create a backup of a file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{Path(file_path).name}.{timestamp}.bak"
        backup_path = self.backup_dir / backup_name
        
        full_path = self.project_root / file_path
        shutil.copy2(full_path, backup_path)
        
        return backup_path
    
    def _rollback_patch(self, patch: CodePatch, backup_dir: Path):
        """Rollback a single patch from backup."""
        try:
            backup_path = backup_dir / patch.file_path
            full_path = self.project_root / patch.file_path
            
            if backup_path.exists():
                shutil.copy2(backup_path, full_path)
                patch.status = PatchStatus.ROLLED_BACK
                logger.info(f"Rolled back: {patch.file_path}")
            else:
                # File was newly created, delete it
                if full_path.exists():
                    full_path.unlink()
                patch.status = PatchStatus.ROLLED_BACK
                
        except Exception as e:
            logger.error(f"Rollback failed for {patch.file_path}: {e}")
    
    def rollback_patchset(self, patchset: PatchSet) -> bool:
        """
        Rollback an applied patchset.
        
        Args:
            patchset: The patchset to rollback
            
        Returns:
            Success status
        """
        if not patchset.backup_dir:
            logger.error("No backup directory for patchset")
            return False
        
        backup_dir = Path(patchset.backup_dir)
        
        if not backup_dir.exists():
            logger.error(f"Backup directory not found: {backup_dir}")
            return False
        
        success = True
        for patch in patchset.patches:
            try:
                self._rollback_patch(patch, backup_dir)
            except Exception as e:
                logger.error(f"Failed to rollback {patch.file_path}: {e}")
                success = False
        
        if success:
            patchset.status = PatchStatus.ROLLED_BACK
        
        return success
    
    def generate_diff_report(self, patchset: PatchSet) -> str:
        """Generate a human-readable diff report."""
        report = []
        report.append(f"# Patch Report: {patchset.description}")
        report.append(f"ID: {patchset.id}")
        report.append(f"Created: {patchset.created_at}")
        report.append(f"Status: {patchset.status.value}")
        report.append("")
        
        stats = patchset.total_stats
        report.append(f"## Summary")
        report.append(f"- Files changed: {stats['files']}")
        report.append(f"- Additions: +{stats['additions']}")
        report.append(f"- Deletions: -{stats['deletions']}")
        report.append("")
        
        for patch in patchset.patches:
            report.append(f"## {patch.file_path}")
            report.append(f"Description: {patch.description}")
            report.append("```diff")
            report.append(patch.diff)
            report.append("```")
            report.append("")
        
        return '\n'.join(report)
