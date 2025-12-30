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
from src.utils.logging import logger


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
            f"Started evolution version control for proposal {proposal.id} on branch {branch_name}"
        )
        
        return version
    
    def commit_changes(
        self,
        proposal_id: str,
        files: List[str],
        message: str = None,
        description: str = None
    ) -> Optional[str]:
        """
        Commit changes for an evolution proposal.
        
        Args:
            proposal_id: ID of the proposal
            files: List of files to commit
            message: Optional custom commit message
            description: Description of changes for message generation
            
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
            message = self._generate_commit_message(proposal_id, description or "Evolution changes", len(files))
        
        # Create commit
        commit_hash = self.git.commit(message, files=files)
        
        if commit_hash:
            version.commits.append(commit_hash)
            version.current_commit = commit_hash
            
            logger.info(
                f"Committed evolution changes for proposal {proposal_id}, commit: {commit_hash[:8]}, files: {len(files)}"
            )
        
        return commit_hash
    
    def _generate_commit_message(self, proposal_id: str, description: str, file_count: int) -> str:
        """Generate a descriptive commit message."""
        title = f"[Evolution] {proposal_id}: {description[:50]}"
        
        body = [
            "",
            f"Files changed: {file_count}",
            "",
            "Automated evolution commit"
        ]
        
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
                f"Finalized evolution {proposal_id}, final commit: {final_commit[:8]}"
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
                f"Rolled back evolution {proposal_id} to {target[:8]}, restored {len(files_affected)} files"
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
