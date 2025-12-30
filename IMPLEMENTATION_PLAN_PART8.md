# King AI v2 - Implementation Plan Part 8
## Evolution Engine - Git Integration & Rollback

**Target Timeline:** Week 6
**Objective:** Implement Git-based version control for all code modifications with safe rollback capabilities.

---

## Overview of All Parts

| Part | Title | Status |
|------|-------|--------|
| 1 | Infrastructure Layer & Core System Hardening | ✅ Complete |
| 2 | Master AI Brain - Core Enhancements | ✅ Complete |
| 3 | Master AI Brain - Context & Memory System | ✅ Complete |
| 4 | Master AI Brain - Planning & ReAct Implementation | ✅ Complete |
| 5 | Evolution Engine - Core Models & Proposal System | ✅ Complete |
| 6 | Evolution Engine - Code Analysis & AST Tools | ✅ Complete |
| 7 | Evolution Engine - Code Patching & Generation | ✅ Complete |
| **8** | **Evolution Engine - Git Integration & Rollback** | 🔄 Current |
| 9 | Evolution Engine - Sandbox Testing | ⏳ Pending |
| 10 | Sub-Agent: Research (Web/API) | ⏳ Pending |
| 11 | Sub-Agent: Code Generator | ⏳ Pending |
| 12 | Sub-Agent: Content (Blog/SEO) | ⏳ Pending |
| 13 | Sub-Agent: Commerce - Shopify | ⏳ Pending |
| 14 | Sub-Agent: Commerce - Suppliers | ⏳ Pending |
| 15 | Sub-Agent: Finance - Stripe | ⏳ Pending |
| 16 | Sub-Agent: Finance - Plaid/Banking | ⏳ Pending |
| 17 | Sub-Agent: Analytics | ⏳ Pending |
| 18 | Sub-Agent: Legal | ⏳ Pending |
| 19 | Business: Lifecycle Engine | ⏳ Pending |
| 20 | Business: Playbook System | ⏳ Pending |
| 21 | Business: Portfolio Management | ⏳ Pending |
| 22 | Dashboard: React Components | ⏳ Pending |
| 23 | Dashboard: Approval Workflows | ⏳ Pending |
| 24 | Dashboard: WebSocket & Monitoring | ⏳ Pending |

---

## Part 8 Scope

This part focuses on:
1. Git repository management utilities
2. Automatic commit generation for changes
3. Branch-based evolution workflow
4. Safe rollback to any previous state
5. Evolution history tracking via Git
6. Conflict detection and resolution

---

## Task 8.1: Create Git Manager

**File:** `src/utils/git_manager.py` (CREATE NEW FILE)

```python
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

from src.utils.structured_logging import get_logger

logger = get_logger("git_manager")


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
```

---

## Task 8.2: Create Evolution Version Controller

**File:** `src/master_ai/evolution_version_control.py` (CREATE NEW FILE)

```python
"""
Evolution Version Controller - Git-based versioning for evolution proposals.
Manages the lifecycle of code changes through Git workflows.
"""

from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from src.utils.git_manager import GitManager, GitError, GitCommit
from src.master_ai.evolution_models import EvolutionProposal, ProposalStatus
from src.utils.code_patcher import PatchSet, CodePatch
from src.utils.structured_logging import get_logger

logger = get_logger("evolution_version_control")


class MergeStrategy(str, Enum):
    """Strategy for merging evolution branches."""
    FAST_FORWARD = "fast_forward"
    MERGE_COMMIT = "merge_commit"
    SQUASH = "squash"


@dataclass
class EvolutionVersion:
    """Represents a versioned evolution state."""
    proposal_id: str
    branch_name: str
    base_commit: str
    current_commit: str
    commits: List[str] = field(default_factory=list)
    status: str = "active"
    created_at: datetime = field(default_factory=datetime.now)
    merged_at: Optional[datetime] = None


@dataclass
class RollbackResult:
    """Result of a rollback operation."""
    success: bool
    target_commit: str
    files_restored: List[str] = field(default_factory=list)
    error: Optional[str] = None


class EvolutionVersionController:
    """
    Controls version management for evolution proposals.
    Each proposal gets its own branch for isolation.
    """
    
    def __init__(self, git_manager: GitManager):
        """
        Initialize the version controller.
        
        Args:
            git_manager: Git manager instance
        """
        self.git = git_manager
        self._active_versions: Dict[str, EvolutionVersion] = {}
    
    def start_evolution(self, proposal: EvolutionProposal) -> EvolutionVersion:
        """
        Start version control for an evolution proposal.
        Creates a new branch from the current state.
        
        Args:
            proposal: The evolution proposal
            
        Returns:
            Evolution version info
        """
        # Ensure we're on main branch
        current_branch = self.git.get_current_branch()
        if current_branch != self.git.MAIN_BRANCH:
            self.git.checkout_branch(self.git.MAIN_BRANCH)
        
        # Get current commit as base
        base_commit = self.git._run_git("rev-parse", "HEAD")
        
        # Create evolution branch
        branch_name = self.git.create_evolution_branch(proposal.id)
        
        version = EvolutionVersion(
            proposal_id=proposal.id,
            branch_name=branch_name,
            base_commit=base_commit,
            current_commit=base_commit
        )
        
        self._active_versions[proposal.id] = version
        
        logger.info(
            f"Started evolution version control",
            proposal_id=proposal.id,
            branch=branch_name
        )
        
        return version
    
    def commit_changes(
        self,
        proposal_id: str,
        patchset: PatchSet,
        message: str = None
    ) -> Optional[str]:
        """
        Commit changes from a patchset.
        
        Args:
            proposal_id: ID of the proposal
            patchset: Applied patchset to commit
            message: Optional custom commit message
            
        Returns:
            Commit hash if successful
        """
        version = self._active_versions.get(proposal_id)
        if not version:
            logger.error(f"No active version for proposal: {proposal_id}")
            return None
        
        # Ensure we're on the right branch
        current = self.git.get_current_branch()
        if current != version.branch_name:
            self.git.checkout_branch(version.branch_name)
        
        # Generate commit message
        if not message:
            message = self._generate_commit_message(patchset)
        
        # Get files from patchset
        files = [patch.file_path for patch in patchset.patches]
        
        # Create commit
        commit_hash = self.git.commit(message, files=files)
        
        if commit_hash:
            version.commits.append(commit_hash)
            version.current_commit = commit_hash
            
            logger.info(
                f"Committed evolution changes",
                proposal_id=proposal_id,
                commit=commit_hash[:8],
                files=len(files)
            )
        
        return commit_hash
    
    def _generate_commit_message(self, patchset: PatchSet) -> str:
        """Generate a descriptive commit message."""
        stats = patchset.total_stats
        
        title = f"[Evolution] {patchset.description[:50]}"
        
        body = [
            "",
            f"Files changed: {stats['files']}",
            f"Additions: +{stats['additions']}",
            f"Deletions: -{stats['deletions']}",
            "",
            "Changes:",
        ]
        
        for patch in patchset.patches[:10]:
            body.append(f"  - {patch.file_path}: {patch.description[:40]}")
        
        if len(patchset.patches) > 10:
            body.append(f"  ... and {len(patchset.patches) - 10} more files")
        
        return title + '\n'.join(body)
    
    def finalize_evolution(
        self,
        proposal_id: str,
        strategy: MergeStrategy = MergeStrategy.SQUASH
    ) -> Tuple[bool, str]:
        """
        Finalize and merge an evolution branch.
        
        Args:
            proposal_id: ID of the proposal
            strategy: Merge strategy to use
            
        Returns:
            Tuple of (success, final commit hash or error message)
        """
        version = self._active_versions.get(proposal_id)
        if not version:
            return False, f"No active version for proposal: {proposal_id}"
        
        try:
            # Switch to main branch
            self.git.checkout_branch(self.git.MAIN_BRANCH)
            
            if strategy == MergeStrategy.FAST_FORWARD:
                self.git._run_git("merge", "--ff-only", version.branch_name)
            elif strategy == MergeStrategy.MERGE_COMMIT:
                message = f"Merge evolution: {proposal_id}"
                success, conflicts = self.git.merge_branch(
                    version.branch_name,
                    message=message,
                    no_ff=True
                )
                if not success:
                    self.git.abort_merge()
                    return False, f"Merge conflicts: {conflicts}"
            elif strategy == MergeStrategy.SQUASH:
                # Squash merge
                self.git._run_git("merge", "--squash", version.branch_name)
                message = f"[Evolution Complete] {proposal_id}\n\nSquashed {len(version.commits)} commits"
                self.git.commit(message)
            
            # Get final commit
            final_commit = self.git._run_git("rev-parse", "HEAD")
            
            # Tag the evolution
            tag_name = f"evolution-{proposal_id}"
            self.git.tag(
                tag_name,
                message=f"Evolution proposal {proposal_id} completed"
            )
            
            # Clean up branch
            self.git.delete_branch(version.branch_name, force=True)
            
            version.status = "merged"
            version.merged_at = datetime.now()
            
            logger.info(
                f"Finalized evolution",
                proposal_id=proposal_id,
                final_commit=final_commit[:8]
            )
            
            return True, final_commit
            
        except GitError as e:
            logger.error(f"Failed to finalize evolution: {e}")
            return False, str(e)
    
    def rollback_evolution(
        self,
        proposal_id: str,
        to_commit: str = None
    ) -> RollbackResult:
        """
        Rollback an evolution to a previous state.
        
        Args:
            proposal_id: ID of the proposal
            to_commit: Specific commit to rollback to (default: base commit)
            
        Returns:
            Rollback result
        """
        version = self._active_versions.get(proposal_id)
        if not version:
            return RollbackResult(
                success=False,
                target_commit="",
                error=f"No active version for proposal: {proposal_id}"
            )
        
        target = to_commit or version.base_commit
        
        try:
            # Ensure we're on the evolution branch
            current = self.git.get_current_branch()
            if current != version.branch_name:
                self.git.checkout_branch(version.branch_name)
            
            # Get files that will be affected
            diff_output = self.git._run_git(
                "diff", "--name-only", target, "HEAD"
            )
            files_affected = [f for f in diff_output.split('\n') if f]
            
            # Perform rollback
            self.git.rollback_to_commit(target, hard=True)
            
            # Update version info
            version.current_commit = target
            # Remove commits after the target
            if to_commit:
                try:
                    target_idx = version.commits.index(to_commit)
                    version.commits = version.commits[:target_idx + 1]
                except ValueError:
                    pass
            else:
                version.commits = []
            
            logger.info(
                f"Rolled back evolution",
                proposal_id=proposal_id,
                target=target[:8],
                files_restored=len(files_affected)
            )
            
            return RollbackResult(
                success=True,
                target_commit=target,
                files_restored=files_affected
            )
            
        except GitError as e:
            logger.error(f"Rollback failed: {e}")
            return RollbackResult(
                success=False,
                target_commit=target,
                error=str(e)
            )
    
    def abort_evolution(self, proposal_id: str) -> bool:
        """
        Abort an evolution and delete its branch.
        
        Args:
            proposal_id: ID of the proposal
            
        Returns:
            Success status
        """
        version = self._active_versions.get(proposal_id)
        if not version:
            return False
        
        try:
            # Switch to main branch
            current = self.git.get_current_branch()
            if current == version.branch_name:
                self.git.checkout_branch(self.git.MAIN_BRANCH)
            
            # Delete the branch
            self.git.delete_branch(version.branch_name, force=True)
            
            # Remove from tracking
            del self._active_versions[proposal_id]
            
            logger.info(f"Aborted evolution: {proposal_id}")
            return True
            
        except GitError as e:
            logger.error(f"Failed to abort evolution: {e}")
            return False
    
    def get_evolution_history(self, proposal_id: str) -> List[GitCommit]:
        """Get commit history for an evolution."""
        version = self._active_versions.get(proposal_id)
        if not version:
            return []
        
        return self.git.get_commit_history(
            count=50,
            branch=version.branch_name
        )
    
    def get_evolution_diff(self, proposal_id: str) -> str:
        """Get full diff for an evolution from base."""
        version = self._active_versions.get(proposal_id)
        if not version:
            return ""
        
        return self.git.get_diff(
            from_ref=version.base_commit,
            to_ref=version.current_commit
        )
    
    def list_active_evolutions(self) -> List[EvolutionVersion]:
        """List all active evolution versions."""
        return list(self._active_versions.values())
    
    def restore_file_from_commit(
        self,
        proposal_id: str,
        file_path: str,
        commit: str
    ) -> Optional[str]:
        """
        Restore a specific file from a commit.
        
        Returns:
            File contents if successful
        """
        try:
            content = self.git.get_file_at_commit(file_path, commit)
            return content
        except GitError as e:
            logger.error(f"Failed to restore file: {e}")
            return None
```

---

## Task 8.3: Create Rollback Service

**File:** `src/master_ai/rollback_service.py` (CREATE NEW FILE)

```python
"""
Rollback Service - High-level rollback operations for evolution engine.
Provides safe recovery from failed or unwanted changes.
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime

from src.utils.git_manager import GitManager, GitCommit
from src.master_ai.evolution_version_control import (
    EvolutionVersionController,
    RollbackResult
)
from src.master_ai.evolution_models import EvolutionProposal, ProposalStatus
from src.database.connection import get_db
from src.utils.structured_logging import get_logger

logger = get_logger("rollback_service")


@dataclass
class RollbackPoint:
    """A point in time that can be rolled back to."""
    commit_hash: str
    timestamp: datetime
    description: str
    proposal_id: Optional[str] = None
    is_safe: bool = True
    files_affected: List[str] = None


@dataclass
class RollbackPlan:
    """Plan for a rollback operation."""
    target: RollbackPoint
    current_state: str
    files_to_restore: List[str]
    estimated_impact: str
    requires_approval: bool
    warnings: List[str]


class RollbackService:
    """
    High-level service for rollback operations.
    Manages recovery from failed evolutions.
    """
    
    def __init__(
        self,
        git_manager: GitManager,
        version_controller: EvolutionVersionController
    ):
        """
        Initialize rollback service.
        
        Args:
            git_manager: Git manager instance
            version_controller: Evolution version controller
        """
        self.git = git_manager
        self.versions = version_controller
        
        # Keep track of safe rollback points
        self._rollback_points: List[RollbackPoint] = []
    
    def get_rollback_points(self, limit: int = 20) -> List[RollbackPoint]:
        """
        Get available rollback points.
        
        Args:
            limit: Maximum number of points to return
            
        Returns:
            List of rollback points, newest first
        """
        commits = self.git.get_commit_history(count=limit)
        
        points = []
        for commit in commits:
            # Determine if this is a safe rollback point
            is_safe = self._is_safe_rollback_point(commit)
            
            # Extract proposal ID if it's an evolution commit
            proposal_id = None
            if "[Evolution]" in commit.message:
                # Try to extract proposal ID from message
                parts = commit.message.split()
                for i, part in enumerate(parts):
                    if part == "[Evolution]" and i + 1 < len(parts):
                        proposal_id = parts[i + 1].strip(":")
                        break
            
            points.append(RollbackPoint(
                commit_hash=commit.hash,
                timestamp=commit.date,
                description=commit.message[:100],
                proposal_id=proposal_id,
                is_safe=is_safe
            ))
        
        return points
    
    def _is_safe_rollback_point(self, commit: GitCommit) -> bool:
        """
        Determine if a commit is a safe rollback point.
        
        Safe points:
        - Evolution completions (tagged)
        - Manual checkpoints
        - Pre-evolution states
        """
        # Check for evolution tags
        if "[Evolution Complete]" in commit.message:
            return True
        
        # Check for checkpoint markers
        if "[Checkpoint]" in commit.message:
            return True
        
        # Merges are generally safe
        if "Merge" in commit.message:
            return True
        
        return False
    
    def create_checkpoint(self, description: str = "Manual checkpoint") -> RollbackPoint:
        """
        Create a safe rollback checkpoint at current state.
        
        Args:
            description: Description for the checkpoint
            
        Returns:
            Created rollback point
        """
        # Create an empty commit as a checkpoint
        message = f"[Checkpoint] {description}"
        
        try:
            self.git._run_git("commit", "--allow-empty", "-m", message)
            commit_hash = self.git._run_git("rev-parse", "HEAD")
        except Exception:
            # If no changes, just get current commit
            commit_hash = self.git._run_git("rev-parse", "HEAD")
        
        point = RollbackPoint(
            commit_hash=commit_hash,
            timestamp=datetime.now(),
            description=message,
            is_safe=True
        )
        
        self._rollback_points.append(point)
        
        logger.info(f"Created checkpoint: {commit_hash[:8]}")
        return point
    
    def plan_rollback(
        self,
        target_commit: str
    ) -> RollbackPlan:
        """
        Create a plan for rolling back to a target commit.
        
        Args:
            target_commit: Commit hash to rollback to
            
        Returns:
            Detailed rollback plan
        """
        current_commit = self.git._run_git("rev-parse", "HEAD")
        
        # Get files that would be affected
        diff_output = self.git._run_git(
            "diff", "--name-only", target_commit, current_commit
        )
        files_to_restore = [f for f in diff_output.split('\n') if f]
        
        # Find the target rollback point
        target_point = None
        for point in self.get_rollback_points():
            if point.commit_hash == target_commit:
                target_point = point
                break
        
        if not target_point:
            target_point = RollbackPoint(
                commit_hash=target_commit,
                timestamp=datetime.now(),
                description="Unknown commit",
                is_safe=False
            )
        
        # Determine impact
        if len(files_to_restore) > 10:
            impact = "HIGH - Many files affected"
        elif len(files_to_restore) > 3:
            impact = "MEDIUM - Several files affected"
        else:
            impact = "LOW - Few files affected"
        
        # Generate warnings
        warnings = []
        
        if not target_point.is_safe:
            warnings.append("Target is not a verified safe rollback point")
        
        critical_files = ['settings.py', 'brain.py', 'models.py']
        affected_critical = [f for f in files_to_restore if any(c in f for c in critical_files)]
        if affected_critical:
            warnings.append(f"Critical files affected: {affected_critical}")
        
        # Check for active evolutions
        active = self.versions.list_active_evolutions()
        if active:
            warnings.append(f"{len(active)} active evolutions may be affected")
        
        return RollbackPlan(
            target=target_point,
            current_state=current_commit,
            files_to_restore=files_to_restore,
            estimated_impact=impact,
            requires_approval=len(warnings) > 0 or not target_point.is_safe,
            warnings=warnings
        )
    
    async def execute_rollback(
        self,
        target_commit: str,
        approved: bool = False
    ) -> RollbackResult:
        """
        Execute a rollback to a target commit.
        
        Args:
            target_commit: Commit to rollback to
            approved: Whether the rollback has been approved
            
        Returns:
            Rollback result
        """
        plan = self.plan_rollback(target_commit)
        
        # Check if approval needed
        if plan.requires_approval and not approved:
            return RollbackResult(
                success=False,
                target_commit=target_commit,
                error="Rollback requires approval"
            )
        
        # Create a checkpoint before rollback
        checkpoint = self.create_checkpoint(
            f"Pre-rollback to {target_commit[:8]}"
        )
        
        try:
            # Perform the rollback
            success = self.git.rollback_to_commit(target_commit, hard=True)
            
            if success:
                # Record in database
                await self._record_rollback(
                    target_commit=target_commit,
                    from_commit=plan.current_state,
                    files_affected=plan.files_to_restore
                )
                
                logger.info(
                    f"Rollback complete",
                    target=target_commit[:8],
                    files=len(plan.files_to_restore)
                )
                
                return RollbackResult(
                    success=True,
                    target_commit=target_commit,
                    files_restored=plan.files_to_restore
                )
            else:
                # Restore from checkpoint
                self.git.rollback_to_commit(checkpoint.commit_hash, hard=True)
                
                return RollbackResult(
                    success=False,
                    target_commit=target_commit,
                    error="Rollback failed, restored to checkpoint"
                )
                
        except Exception as e:
            logger.error(f"Rollback error: {e}")
            
            # Try to restore from checkpoint
            try:
                self.git.rollback_to_commit(checkpoint.commit_hash, hard=True)
            except:
                pass
            
            return RollbackResult(
                success=False,
                target_commit=target_commit,
                error=str(e)
            )
    
    async def _record_rollback(
        self,
        target_commit: str,
        from_commit: str,
        files_affected: List[str]
    ):
        """Record rollback in database for audit trail."""
        # This would insert into an audit/history table
        pass
    
    def undo_last_evolution(self) -> RollbackResult:
        """
        Quick undo of the last evolution.
        
        Returns:
            Rollback result
        """
        # Find the last evolution commit
        commits = self.git.get_commit_history(count=10)
        
        target = None
        for i, commit in enumerate(commits):
            if "[Evolution" in commit.message:
                # Found an evolution commit, rollback to before it
                if i + 1 < len(commits):
                    target = commits[i + 1].hash
                break
        
        if not target:
            return RollbackResult(
                success=False,
                target_commit="",
                error="No evolution commit found to undo"
            )
        
        return self.git.rollback_to_commit(target, hard=True)
```

---

## Testing Requirements

**File:** `tests/test_git_manager.py` (CREATE NEW FILE)

```python
"""Tests for Git manager and version control."""

import pytest
import tempfile
import os
from pathlib import Path

from src.utils.git_manager import GitManager, GitError, GitStatus


class TestGitManager:
    """Tests for GitManager."""
    
    @pytest.fixture
    def git_repo(self):
        """Create a temporary Git repository."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Initialize git repo
            os.system(f"cd {tmpdir} && git init")
            os.system(f"cd {tmpdir} && git config user.email 'test@test.com'")
            os.system(f"cd {tmpdir} && git config user.name 'Test'")
            
            # Create initial commit
            test_file = Path(tmpdir) / "test.py"
            test_file.write_text("# Test file\n")
            os.system(f"cd {tmpdir} && git add . && git commit -m 'Initial commit'")
            
            yield tmpdir
    
    @pytest.fixture
    def manager(self, git_repo):
        return GitManager(git_repo)
    
    def test_get_status(self, manager, git_repo):
        """Test getting repository status."""
        status = manager.get_status()
        
        assert isinstance(status, GitStatus)
        assert status.is_clean
    
    def test_create_branch(self, manager):
        """Test branch creation."""
        branch = manager.create_evolution_branch("test-123")
        
        assert branch == "evolution/test-123"
        assert manager.get_current_branch() == branch
    
    def test_commit(self, manager, git_repo):
        """Test creating commits."""
        # Make a change
        test_file = Path(git_repo) / "test.py"
        test_file.write_text("# Modified\n")
        
        commit_hash = manager.commit("Test commit")
        
        assert commit_hash
        assert len(commit_hash) == 40
    
    def test_rollback(self, manager, git_repo):
        """Test rollback functionality."""
        # Get initial commit
        initial = manager._run_git("rev-parse", "HEAD")
        
        # Make changes and commit
        test_file = Path(git_repo) / "test.py"
        test_file.write_text("# Modified\n")
        manager.commit("Modification")
        
        # Rollback
        success = manager.rollback_to_commit(initial, hard=True)
        
        assert success
        assert test_file.read_text() == "# Test file\n"
    
    def test_commit_history(self, manager, git_repo):
        """Test getting commit history."""
        # Create some commits
        for i in range(3):
            test_file = Path(git_repo) / f"file{i}.py"
            test_file.write_text(f"# File {i}\n")
            manager.commit(f"Commit {i}")
        
        history = manager.get_commit_history(count=5)
        
        assert len(history) >= 3
        assert all(h.hash for h in history)
```

---

## Acceptance Criteria

- [ ] `src/utils/git_manager.py` - Full Git operations wrapper
- [ ] `src/master_ai/evolution_version_control.py` - Evolution branch workflow
- [ ] `src/master_ai/rollback_service.py` - Safe rollback operations
- [ ] `tests/test_git_manager.py` - All tests passing
- [ ] Evolution branches created/merged correctly
- [ ] Rollback restores correct state
- [ ] Audit trail maintained

---

## File Summary

| Action | File Path |
|--------|-----------|
| CREATE | `src/utils/git_manager.py` |
| CREATE | `src/master_ai/evolution_version_control.py` |
| CREATE | `src/master_ai/rollback_service.py` |
| CREATE | `tests/test_git_manager.py` |

---

*End of Part 8*
