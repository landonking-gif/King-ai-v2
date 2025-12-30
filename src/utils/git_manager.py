"""
Git Manager - Version control integration for evolution engine.
Handles commits, branches, and rollbacks for code modifications.
"""

import os
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from src.utils.logging import logger


class GitError(Exception):
    """Custom exception for Git operations."""
    pass


@dataclass
class GitCommit:
    """Represents a Git commit."""
    hash: str
    short_hash: str
    author: str
    date: datetime
    message: str
    files_changed: List[str] = field(default_factory=list)
    
    @classmethod
    def from_log_line(cls, line: str) -> "GitCommit":
        """Parse a commit from git log output."""
        # Format: hash|short_hash|author|date|message
        parts = line.split('|')
        if len(parts) < 5:
            raise ValueError(f"Invalid log line: {line}")
        
        return cls(
            hash=parts[0],
            short_hash=parts[1],
            author=parts[2],
            date=datetime.fromisoformat(parts[3]),
            message=parts[4]
        )


@dataclass
class GitBranch:
    """Represents a Git branch."""
    name: str
    is_current: bool = False
    last_commit: Optional[str] = None
    tracking: Optional[str] = None


@dataclass
class GitStatus:
    """Current Git repository status."""
    branch: str
    is_clean: bool
    staged_files: List[str] = field(default_factory=list)
    modified_files: List[str] = field(default_factory=list)
    untracked_files: List[str] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)


class GitManager:
    """
    Manages Git operations for the evolution engine.
    All code modifications are version controlled.
    """
    
    # Branch naming conventions
    EVOLUTION_BRANCH_PREFIX = "evolution/"
    MAIN_BRANCH = "main"
    
    def __init__(self, repo_path: str):
        """
        Initialize Git manager.
        
        Args:
            repo_path: Path to the Git repository
        """
        self.repo_path = Path(repo_path)
        
        # Verify it's a Git repo
        if not (self.repo_path / ".git").exists():
            raise GitError(f"Not a Git repository: {repo_path}")
        
        # Configure Git for automation
        self._configure_git()
    
    def _configure_git(self):
        """Configure Git for automated operations."""
        # Set user for commits if not set
        try:
            self._run_git("config", "user.name")
        except GitError:
            self._run_git("config", "user.name", "King AI Evolution Engine")
            self._run_git("config", "user.email", "evolution@king-ai.local")
    
    def _run_git(self, *args, check: bool = True) -> str:
        """
        Run a Git command.
        
        Args:
            *args: Git command arguments
            check: Whether to raise on non-zero exit
            
        Returns:
            Command output
        """
        cmd = ["git"] + list(args)
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=check
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            raise GitError(f"Git command failed: {' '.join(cmd)}\n{e.stderr}")
    
    def get_status(self) -> GitStatus:
        """Get current repository status."""
        # Get current branch
        branch = self._run_git("branch", "--show-current")
        
        # Get status
        status_output = self._run_git("status", "--porcelain")
        
        staged = []
        modified = []
        untracked = []
        conflicts = []
        
        for line in status_output.split('\n'):
            if not line:
                continue
            
            status_code = line[:2]
            file_path = line[3:]
            
            if 'U' in status_code:
                conflicts.append(file_path)
            elif status_code[0] in 'MADRC':
                staged.append(file_path)
            elif status_code[1] in 'MD':
                modified.append(file_path)
            elif status_code == '??':
                untracked.append(file_path)
        
        return GitStatus(
            branch=branch,
            is_clean=len(staged) == 0 and len(modified) == 0,
            staged_files=staged,
            modified_files=modified,
            untracked_files=untracked,
            conflicts=conflicts
        )
    
    def create_evolution_branch(self, proposal_id: str) -> str:
        """
        Create a new branch for an evolution proposal.
        
        Args:
            proposal_id: ID of the evolution proposal
            
        Returns:
            Name of the created branch
        """
        branch_name = f"{self.EVOLUTION_BRANCH_PREFIX}{proposal_id}"
        
        # Create and checkout the branch
        self._run_git("checkout", "-b", branch_name)
        
        logger.info(f"Created evolution branch: {branch_name}")
        return branch_name
    
    def checkout_branch(self, branch_name: str):
        """Switch to a branch."""
        self._run_git("checkout", branch_name)
        logger.info(f"Switched to branch: {branch_name}")
    
    def get_current_branch(self) -> str:
        """Get the current branch name."""
        return self._run_git("branch", "--show-current")
    
    def list_branches(self, pattern: str = None) -> List[GitBranch]:
        """
        List branches matching a pattern.
        
        Args:
            pattern: Optional pattern to filter branches
            
        Returns:
            List of branch info
        """
        args = ["branch", "-v"]
        if pattern:
            args.extend(["--list", pattern])
        
        output = self._run_git(*args)
        branches = []
        
        for line in output.split('\n'):
            if not line.strip():
                continue
            
            is_current = line.startswith('*')
            parts = line.lstrip('* ').split()
            
            if len(parts) >= 2:
                branches.append(GitBranch(
                    name=parts[0],
                    is_current=is_current,
                    last_commit=parts[1] if len(parts) > 1 else None
                ))
        
        return branches
    
    def stage_files(self, files: List[str] = None):
        """
        Stage files for commit.
        
        Args:
            files: List of files to stage. If None, stages all changes.
        """
        if files:
            for file in files:
                self._run_git("add", file)
        else:
            self._run_git("add", "-A")
    
    def commit(
        self,
        message: str,
        author: str = None,
        files: List[str] = None
    ) -> str:
        """
        Create a commit.
        
        Args:
            message: Commit message
            author: Optional author override
            files: Optional list of files to commit
            
        Returns:
            Commit hash
        """
        # Stage files
        if files:
            self.stage_files(files)
        else:
            self.stage_files()
        
        # Check if there are staged changes
        status = self.get_status()
        if not status.staged_files:
            logger.warning("No changes to commit")
            return ""
        
        # Build commit command
        args = ["commit", "-m", message]
        if author:
            args.extend(["--author", author])
        
        self._run_git(*args)
        
        # Get the commit hash
        commit_hash = self._run_git("rev-parse", "HEAD")
        
        logger.info(f"Created commit: {commit_hash[:8]} - {message[:50]}")
        return commit_hash
    
    def get_commit_history(
        self,
        count: int = 20,
        branch: str = None,
        path: str = None
    ) -> List[GitCommit]:
        """
        Get commit history.
        
        Args:
            count: Number of commits to retrieve
            branch: Optional branch to get history from
            path: Optional path to filter commits
            
        Returns:
            List of commits
        """
        args = [
            "log",
            f"-{count}",
            "--format=%H|%h|%an|%aI|%s"
        ]
        
        if branch:
            args.append(branch)
        
        if path:
            args.extend(["--", path])
        
        output = self._run_git(*args)
        commits = []
        
        for line in output.split('\n'):
            if line.strip():
                try:
                    commits.append(GitCommit.from_log_line(line))
                except ValueError as e:
                    logger.warning(f"Failed to parse commit: {e}")
        
        return commits
    
    def get_diff(
        self,
        from_ref: str = None,
        to_ref: str = None,
        path: str = None
    ) -> str:
        """
        Get diff between references.
        
        Args:
            from_ref: Starting reference (default: HEAD~1)
            to_ref: Ending reference (default: HEAD)
            path: Optional path to filter
            
        Returns:
            Unified diff output
        """
        args = ["diff"]
        
        if from_ref:
            args.append(from_ref)
        if to_ref:
            args.append(to_ref)
        if path:
            args.extend(["--", path])
        
        return self._run_git(*args)
    
    def rollback_to_commit(self, commit_hash: str, hard: bool = False) -> bool:
        """
        Rollback to a specific commit.
        
        Args:
            commit_hash: Commit to rollback to
            hard: Whether to discard all changes (True) or keep as staged (False)
            
        Returns:
            Success status
        """
        try:
            if hard:
                self._run_git("reset", "--hard", commit_hash)
            else:
                self._run_git("reset", "--soft", commit_hash)
            
            logger.info(f"Rolled back to commit: {commit_hash[:8]}")
            return True
            
        except GitError as e:
            logger.error(f"Rollback failed: {e}")
            return False
    
    def revert_commit(self, commit_hash: str) -> Optional[str]:
        """
        Create a revert commit for a specific commit.
        
        Args:
            commit_hash: Commit to revert
            
        Returns:
            New commit hash if successful
        """
        try:
            self._run_git("revert", "--no-edit", commit_hash)
            new_hash = self._run_git("rev-parse", "HEAD")
            
            logger.info(f"Reverted commit {commit_hash[:8]}, new commit: {new_hash[:8]}")
            return new_hash
            
        except GitError as e:
            logger.error(f"Revert failed: {e}")
            return None
    
    def stash_changes(self, message: str = None) -> bool:
        """Stash current changes."""
        try:
            args = ["stash", "push"]
            if message:
                args.extend(["-m", message])
            
            self._run_git(*args)
            return True
        except GitError:
            return False
    
    def pop_stash(self) -> bool:
        """Pop the latest stash."""
        try:
            self._run_git("stash", "pop")
            return True
        except GitError:
            return False
    
    def merge_branch(
        self,
        branch: str,
        message: str = None,
        no_ff: bool = True
    ) -> Tuple[bool, List[str]]:
        """
        Merge a branch into current branch.
        
        Args:
            branch: Branch to merge
            message: Optional merge commit message
            no_ff: Whether to always create merge commit
            
        Returns:
            Tuple of (success, list of conflicts)
        """
        args = ["merge"]
        
        if no_ff:
            args.append("--no-ff")
        if message:
            args.extend(["-m", message])
        
        args.append(branch)
        
        try:
            self._run_git(*args)
            return True, []
            
        except GitError as e:
            # Check for merge conflicts
            status = self.get_status()
            if status.conflicts:
                logger.warning(f"Merge conflicts: {status.conflicts}")
                return False, status.conflicts
            raise
    
    def abort_merge(self):
        """Abort an in-progress merge."""
        self._run_git("merge", "--abort")
    
    def delete_branch(self, branch: str, force: bool = False):
        """Delete a branch."""
        args = ["branch"]
        args.append("-D" if force else "-d")
        args.append(branch)
        
        self._run_git(*args)
        logger.info(f"Deleted branch: {branch}")
    
    def tag(self, name: str, message: str = None, commit: str = None):
        """
        Create a tag.
        
        Args:
            name: Tag name
            message: Optional annotation message
            commit: Optional commit to tag (default: HEAD)
        """
        args = ["tag"]
        
        if message:
            args.extend(["-a", name, "-m", message])
        else:
            args.append(name)
        
        if commit:
            args.append(commit)
        
        self._run_git(*args)
        logger.info(f"Created tag: {name}")
    
    def get_file_at_commit(self, file_path: str, commit: str) -> str:
        """
        Get file contents at a specific commit.
        
        Args:
            file_path: Path to the file
            commit: Commit reference
            
        Returns:
            File contents
        """
        return self._run_git("show", f"{commit}:{file_path}")
    
    def blame(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Get blame information for a file.
        
        Returns:
            List of blame entries with commit, author, line info
        """
        output = self._run_git("blame", "--line-porcelain", file_path)
        
        entries = []
        current_entry = {}
        
        for line in output.split('\n'):
            if line.startswith('\t'):
                current_entry['content'] = line[1:]
                entries.append(current_entry)
                current_entry = {}
            elif line.startswith('author '):
                current_entry['author'] = line[7:]
            elif line.startswith('author-time '):
                current_entry['timestamp'] = int(line[12:])
            elif len(line) == 40:  # Commit hash
                current_entry['commit'] = line
        
        return entries
