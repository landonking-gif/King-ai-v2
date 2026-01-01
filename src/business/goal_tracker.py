"""
Goal Tracking System.
OKR-style goal setting and progress tracking.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable
from enum import Enum
import uuid

from src.utils.structured_logging import get_logger

logger = get_logger("goal_tracking")


class GoalStatus(str, Enum):
    """Status of a goal."""
    DRAFT = "draft"
    ACTIVE = "active"
    ON_TRACK = "on_track"
    AT_RISK = "at_risk"
    BEHIND = "behind"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class GoalType(str, Enum):
    """Types of goals."""
    OBJECTIVE = "objective"  # High-level objective
    KEY_RESULT = "key_result"  # Measurable key result
    INITIATIVE = "initiative"  # Action item
    MILESTONE = "milestone"  # Checkpoint


class MetricType(str, Enum):
    """Types of metrics."""
    NUMBER = "number"  # Absolute number
    PERCENTAGE = "percentage"  # Percentage value
    CURRENCY = "currency"  # Money value
    BOOLEAN = "boolean"  # Yes/No achievement


class UpdateType(str, Enum):
    """Types of progress updates."""
    MANUAL = "manual"
    AUTOMATED = "automated"
    SYNC = "sync"


@dataclass
class ProgressUpdate:
    """A progress update for a goal."""
    id: str
    goal_id: str
    value: float
    previous_value: float
    update_type: UpdateType
    note: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    updated_by: str = "system"
    
    @property
    def change(self) -> float:
        return self.value - self.previous_value
    
    @property
    def change_percent(self) -> float:
        if self.previous_value == 0:
            return 100.0 if self.value > 0 else 0.0
        return (self.change / self.previous_value) * 100


@dataclass
class Goal:
    """A trackable goal."""
    id: str
    title: str
    description: str = ""
    type: GoalType = GoalType.KEY_RESULT
    status: GoalStatus = GoalStatus.DRAFT
    
    # Metrics
    metric_type: MetricType = MetricType.NUMBER
    target_value: float = 0.0
    current_value: float = 0.0
    start_value: float = 0.0
    unit: str = ""
    
    # Timeline
    start_date: datetime = field(default_factory=datetime.utcnow)
    end_date: Optional[datetime] = None
    
    # Hierarchy
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    
    # Ownership
    owner: str = ""
    team: str = ""
    tags: List[str] = field(default_factory=list)
    
    # Tracking
    updates: List[ProgressUpdate] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def progress(self) -> float:
        """Calculate progress percentage."""
        if self.target_value == self.start_value:
            return 100.0 if self.current_value >= self.target_value else 0.0
        
        progress = (self.current_value - self.start_value) / (self.target_value - self.start_value)
        return max(0.0, min(100.0, progress * 100))
    
    @property
    def days_remaining(self) -> Optional[int]:
        if not self.end_date:
            return None
        return max(0, (self.end_date - datetime.utcnow()).days)
    
    @property
    def days_elapsed(self) -> int:
        return (datetime.utcnow() - self.start_date).days
    
    @property
    def expected_progress(self) -> float:
        """Calculate expected progress based on timeline."""
        if not self.end_date:
            return 0.0
        
        total_days = (self.end_date - self.start_date).days
        if total_days <= 0:
            return 100.0
        
        elapsed = self.days_elapsed
        return min(100.0, (elapsed / total_days) * 100)
    
    @property
    def is_on_track(self) -> bool:
        """Check if goal is on track."""
        return self.progress >= self.expected_progress * 0.8
    
    def update_status(self) -> GoalStatus:
        """Update and return status based on progress."""
        if self.status in [GoalStatus.COMPLETED, GoalStatus.CANCELLED]:
            return self.status
        
        if self.progress >= 100:
            self.status = GoalStatus.COMPLETED
        elif self.is_on_track:
            self.status = GoalStatus.ON_TRACK
        elif self.progress >= self.expected_progress * 0.6:
            self.status = GoalStatus.AT_RISK
        else:
            self.status = GoalStatus.BEHIND
        
        return self.status
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "type": self.type.value,
            "status": self.status.value,
            "metric_type": self.metric_type.value,
            "target_value": self.target_value,
            "current_value": self.current_value,
            "start_value": self.start_value,
            "unit": self.unit,
            "progress": round(self.progress, 1),
            "expected_progress": round(self.expected_progress, 1),
            "is_on_track": self.is_on_track,
            "days_remaining": self.days_remaining,
            "owner": self.owner,
            "team": self.team,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat() if self.end_date else None,
        }


@dataclass
class OKR:
    """Objective and Key Results."""
    objective: Goal
    key_results: List[Goal] = field(default_factory=list)
    
    @property
    def overall_progress(self) -> float:
        """Calculate overall OKR progress."""
        if not self.key_results:
            return 0.0
        return sum(kr.progress for kr in self.key_results) / len(self.key_results)
    
    @property
    def status(self) -> GoalStatus:
        """Get overall OKR status."""
        if all(kr.status == GoalStatus.COMPLETED for kr in self.key_results):
            return GoalStatus.COMPLETED
        
        if any(kr.status == GoalStatus.BEHIND for kr in self.key_results):
            return GoalStatus.BEHIND
        
        if any(kr.status == GoalStatus.AT_RISK for kr in self.key_results):
            return GoalStatus.AT_RISK
        
        if all(kr.status == GoalStatus.ON_TRACK for kr in self.key_results):
            return GoalStatus.ON_TRACK
        
        return GoalStatus.ACTIVE
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "objective": self.objective.to_dict(),
            "key_results": [kr.to_dict() for kr in self.key_results],
            "overall_progress": round(self.overall_progress, 1),
            "status": self.status.value,
        }


@dataclass
class GoalSummary:
    """Summary of goals."""
    total_goals: int
    completed: int
    on_track: int
    at_risk: int
    behind: int
    avg_progress: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_goals": self.total_goals,
            "completed": self.completed,
            "on_track": self.on_track,
            "at_risk": self.at_risk,
            "behind": self.behind,
            "avg_progress": round(self.avg_progress, 1),
            "completion_rate": round(self.completed / self.total_goals * 100, 1) if self.total_goals > 0 else 0,
        }


class GoalTracker:
    """
    Goal Tracking System.
    
    Features:
    - OKR-style goal setting
    - Progress tracking
    - Status updates
    - Goal hierarchies
    - Automated progress sync
    """
    
    def __init__(self):
        self.goals: Dict[str, Goal] = {}
        self.metric_sources: Dict[str, Callable[[], float]] = {}
    
    def create_goal(
        self,
        title: str,
        target_value: float,
        metric_type: MetricType = MetricType.NUMBER,
        goal_type: GoalType = GoalType.KEY_RESULT,
        start_value: float = 0.0,
        unit: str = "",
        end_date: Optional[datetime] = None,
        parent_id: Optional[str] = None,
        owner: str = "",
        team: str = "",
        description: str = "",
        tags: List[str] = None,
    ) -> Goal:
        """
        Create a new goal.
        
        Args:
            title: Goal title
            target_value: Target value to achieve
            metric_type: Type of metric
            goal_type: Type of goal
            start_value: Starting value
            unit: Unit of measurement
            end_date: Target completion date
            parent_id: Parent goal ID
            owner: Goal owner
            team: Team responsible
            description: Goal description
            tags: Goal tags
            
        Returns:
            Created goal
        """
        goal_id = str(uuid.uuid4())[:8]
        
        goal = Goal(
            id=goal_id,
            title=title,
            description=description,
            type=goal_type,
            status=GoalStatus.ACTIVE,
            metric_type=metric_type,
            target_value=target_value,
            current_value=start_value,
            start_value=start_value,
            unit=unit,
            end_date=end_date,
            parent_id=parent_id,
            owner=owner,
            team=team,
            tags=tags or [],
        )
        
        self.goals[goal_id] = goal
        
        # Link to parent
        if parent_id and parent_id in self.goals:
            self.goals[parent_id].children_ids.append(goal_id)
        
        logger.info(f"Created goal: {title}", extra={"goal_id": goal_id})
        
        return goal
    
    def create_okr(
        self,
        objective_title: str,
        key_results: List[Dict[str, Any]],
        end_date: Optional[datetime] = None,
        owner: str = "",
        team: str = "",
    ) -> OKR:
        """
        Create an OKR (Objective and Key Results).
        
        Args:
            objective_title: Title of the objective
            key_results: List of key result definitions
            end_date: Target date for the OKR
            owner: OKR owner
            team: Team responsible
            
        Returns:
            Created OKR
        """
        # Create objective
        objective = self.create_goal(
            title=objective_title,
            target_value=100,
            metric_type=MetricType.PERCENTAGE,
            goal_type=GoalType.OBJECTIVE,
            end_date=end_date,
            owner=owner,
            team=team,
        )
        
        # Create key results
        krs = []
        for kr_def in key_results:
            kr = self.create_goal(
                title=kr_def["title"],
                target_value=kr_def.get("target", 100),
                metric_type=MetricType(kr_def.get("metric_type", "number")),
                goal_type=GoalType.KEY_RESULT,
                start_value=kr_def.get("start_value", 0),
                unit=kr_def.get("unit", ""),
                end_date=end_date,
                parent_id=objective.id,
                owner=kr_def.get("owner", owner),
                team=team,
            )
            krs.append(kr)
        
        return OKR(objective=objective, key_results=krs)
    
    def update_progress(
        self,
        goal_id: str,
        value: float,
        note: str = "",
        update_type: UpdateType = UpdateType.MANUAL,
        updated_by: str = "system",
    ) -> Optional[ProgressUpdate]:
        """
        Update goal progress.
        
        Args:
            goal_id: Goal ID
            value: New current value
            note: Update note
            update_type: Type of update
            updated_by: Who made the update
            
        Returns:
            Progress update record
        """
        goal = self.goals.get(goal_id)
        if not goal:
            return None
        
        update = ProgressUpdate(
            id=str(uuid.uuid4())[:8],
            goal_id=goal_id,
            value=value,
            previous_value=goal.current_value,
            update_type=update_type,
            note=note,
            updated_by=updated_by,
        )
        
        goal.current_value = value
        goal.updates.append(update)
        goal.updated_at = datetime.utcnow()
        goal.update_status()
        
        # Update parent objective progress
        if goal.parent_id and goal.parent_id in self.goals:
            self._update_parent_progress(goal.parent_id)
        
        logger.info(
            f"Updated goal progress",
            extra={
                "goal_id": goal_id,
                "value": value,
                "progress": goal.progress,
            },
        )
        
        return update
    
    def _update_parent_progress(self, parent_id: str) -> None:
        """Update parent goal progress based on children."""
        parent = self.goals.get(parent_id)
        if not parent or not parent.children_ids:
            return
        
        children = [self.goals[cid] for cid in parent.children_ids if cid in self.goals]
        if not children:
            return
        
        avg_progress = sum(c.progress for c in children) / len(children)
        parent.current_value = avg_progress
        parent.target_value = 100
        parent.update_status()
    
    def register_metric_source(
        self,
        goal_id: str,
        source: Callable[[], float],
    ) -> None:
        """
        Register an automated metric source for a goal.
        
        Args:
            goal_id: Goal ID
            source: Callable that returns the current metric value
        """
        self.metric_sources[goal_id] = source
        logger.info(f"Registered metric source for goal {goal_id}")
    
    async def sync_metrics(self) -> List[ProgressUpdate]:
        """
        Sync all goals with registered metric sources.
        
        Returns:
            List of progress updates made
        """
        updates = []
        
        for goal_id, source in self.metric_sources.items():
            try:
                value = source()
                update = self.update_progress(
                    goal_id=goal_id,
                    value=value,
                    update_type=UpdateType.SYNC,
                )
                if update:
                    updates.append(update)
            except Exception as e:
                logger.error(f"Failed to sync metric for goal {goal_id}: {e}")
        
        return updates
    
    def get_goal(self, goal_id: str) -> Optional[Goal]:
        """Get a goal by ID."""
        return self.goals.get(goal_id)
    
    def get_goals_by_team(self, team: str) -> List[Goal]:
        """Get all goals for a team."""
        return [g for g in self.goals.values() if g.team == team]
    
    def get_goals_by_owner(self, owner: str) -> List[Goal]:
        """Get all goals for an owner."""
        return [g for g in self.goals.values() if g.owner == owner]
    
    def get_goals_by_status(self, status: GoalStatus) -> List[Goal]:
        """Get all goals with a specific status."""
        return [g for g in self.goals.values() if g.status == status]
    
    def get_at_risk_goals(self) -> List[Goal]:
        """Get goals that are at risk or behind."""
        return [
            g for g in self.goals.values()
            if g.status in [GoalStatus.AT_RISK, GoalStatus.BEHIND]
        ]
    
    def get_summary(
        self,
        team: Optional[str] = None,
        owner: Optional[str] = None,
    ) -> GoalSummary:
        """
        Get summary of goals.
        
        Args:
            team: Filter by team
            owner: Filter by owner
            
        Returns:
            Goal summary
        """
        goals = list(self.goals.values())
        
        if team:
            goals = [g for g in goals if g.team == team]
        if owner:
            goals = [g for g in goals if g.owner == owner]
        
        # Update all statuses
        for goal in goals:
            goal.update_status()
        
        if not goals:
            return GoalSummary(
                total_goals=0,
                completed=0,
                on_track=0,
                at_risk=0,
                behind=0,
                avg_progress=0.0,
            )
        
        return GoalSummary(
            total_goals=len(goals),
            completed=sum(1 for g in goals if g.status == GoalStatus.COMPLETED),
            on_track=sum(1 for g in goals if g.status == GoalStatus.ON_TRACK),
            at_risk=sum(1 for g in goals if g.status == GoalStatus.AT_RISK),
            behind=sum(1 for g in goals if g.status == GoalStatus.BEHIND),
            avg_progress=sum(g.progress for g in goals) / len(goals),
        )
    
    def get_okr(self, objective_id: str) -> Optional[OKR]:
        """Get an OKR by objective ID."""
        objective = self.goals.get(objective_id)
        if not objective or objective.type != GoalType.OBJECTIVE:
            return None
        
        key_results = [
            self.goals[kid]
            for kid in objective.children_ids
            if kid in self.goals
        ]
        
        return OKR(objective=objective, key_results=key_results)
    
    def complete_goal(self, goal_id: str) -> bool:
        """Mark a goal as completed."""
        goal = self.goals.get(goal_id)
        if not goal:
            return False
        
        goal.status = GoalStatus.COMPLETED
        goal.current_value = goal.target_value
        goal.updated_at = datetime.utcnow()
        
        logger.info(f"Completed goal: {goal.title}")
        return True
    
    def cancel_goal(self, goal_id: str, reason: str = "") -> bool:
        """Cancel a goal."""
        goal = self.goals.get(goal_id)
        if not goal:
            return False
        
        goal.status = GoalStatus.CANCELLED
        goal.updated_at = datetime.utcnow()
        
        # Add cancellation note
        self.update_progress(
            goal_id=goal_id,
            value=goal.current_value,
            note=f"Cancelled: {reason}",
            update_type=UpdateType.MANUAL,
        )
        
        logger.info(f"Cancelled goal: {goal.title}", extra={"reason": reason})
        return True
    
    def get_progress_history(
        self,
        goal_id: str,
        days: int = 30,
    ) -> List[Dict[str, Any]]:
        """
        Get progress history for a goal.
        
        Args:
            goal_id: Goal ID
            days: Number of days of history
            
        Returns:
            List of progress updates
        """
        goal = self.goals.get(goal_id)
        if not goal:
            return []
        
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        return [
            {
                "timestamp": u.timestamp.isoformat(),
                "value": u.value,
                "change": u.change,
                "note": u.note,
            }
            for u in goal.updates
            if u.timestamp >= cutoff
        ]
    
    def generate_report(
        self,
        team: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate a goal tracking report.
        
        Args:
            team: Filter by team
            
        Returns:
            Comprehensive report
        """
        summary = self.get_summary(team=team)
        at_risk = self.get_at_risk_goals()
        
        if team:
            at_risk = [g for g in at_risk if g.team == team]
        
        # Get OKRs
        objectives = [
            g for g in self.goals.values()
            if g.type == GoalType.OBJECTIVE and (not team or g.team == team)
        ]
        
        okrs = [self.get_okr(o.id) for o in objectives]
        okrs = [o for o in okrs if o is not None]
        
        return {
            "summary": summary.to_dict(),
            "at_risk_goals": [g.to_dict() for g in at_risk[:5]],
            "okrs": [o.to_dict() for o in okrs[:5]],
            "generated_at": datetime.utcnow().isoformat(),
        }


# Global goal tracker instance
goal_tracker = GoalTracker()


def get_goal_tracker() -> GoalTracker:
    """Get the global goal tracker."""
    return goal_tracker
