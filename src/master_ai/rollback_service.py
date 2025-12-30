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
from src.utils.logging import logger


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
                    f"Rollback complete to {target_commit[:8]}, restored {len(plan.files_to_restore)} files"
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
        # Placeholder for future database integration
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
        
        success = self.git.rollback_to_commit(target, hard=True)
        
        return RollbackResult(
            success=success,
            target_commit=target,
            error=None if success else "Rollback failed"
        )
