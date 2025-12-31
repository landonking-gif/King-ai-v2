"""
Lifecycle Engine - Manages the progression of business units.
Implements the standard King AI transition logic and enhanced lifecycle management.
"""

import uuid
from datetime import datetime, timedelta
from typing import Any, Optional, Callable
from src.database.models import BusinessStatus
from src.business.lifecycle_models import (
    LifecycleStage, LifecycleState, LifecycleTransition,
    Milestone, MilestoneType, TransitionTrigger, StageRequirement,
    STAGE_CONFIG
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


class BasicLifecycleEngine:
    """
    Basic deterministic state machine for business progression.
    Maintains backward compatibility with original implementation.
    """
    
    # Define valid transitions
    TRANSITIONS = {
        BusinessStatus.DISCOVERY: BusinessStatus.VALIDATION,
        BusinessStatus.VALIDATION: BusinessStatus.SETUP,
        BusinessStatus.SETUP: BusinessStatus.OPERATION,
        BusinessStatus.OPERATION: BusinessStatus.OPTIMIZATION,
        BusinessStatus.OPTIMIZATION: BusinessStatus.REPLICATION
    }

    def get_next_status(self, current_status: BusinessStatus) -> BusinessStatus | None:
        """
        Returns the next logical stage for a business unit.
        Returns None for terminal states (REPLICATION, SUNSET).
        """
        if current_status in (BusinessStatus.SUNSET, BusinessStatus.REPLICATION):
            return None
        return self.TRANSITIONS.get(current_status)

    def is_failed(self, status: BusinessStatus) -> bool:
        """Checks if the business has been sunset."""
        return status == BusinessStatus.SUNSET


# Alias for backward compatibility
LifecycleEngine = BasicLifecycleEngine


class EnhancedLifecycleEngine:
    """Engine for managing business lifecycle transitions with milestones and health tracking."""

    def __init__(self):
        self._states: dict[str, LifecycleState] = {}
        self._hooks: dict[str, list[Callable]] = {
            "pre_transition": [],
            "post_transition": [],
            "milestone_achieved": [],
            "health_warning": [],
        }
        self._metric_providers: dict[str, Callable] = {}

    def register_hook(self, event: str, callback: Callable):
        """Register a callback for lifecycle events."""
        if event in self._hooks:
            self._hooks[event].append(callback)

    def register_metric_provider(self, metric_name: str, provider: Callable):
        """Register a function to provide metric values."""
        self._metric_providers[metric_name] = provider

    async def initialize_business(
        self,
        business_id: str,
        initial_stage: LifecycleStage = LifecycleStage.IDEATION,
    ) -> LifecycleState:
        """Initialize lifecycle tracking for a business."""
        milestones = self._create_stage_milestones(initial_stage)
        
        state = LifecycleState(
            business_id=business_id,
            current_stage=initial_stage,
            entered_at=datetime.utcnow(),
            milestones=milestones,
            next_stage=self._get_next_stage(initial_stage),
        )
        self._states[business_id] = state
        
        logger.info(f"Initialized lifecycle for {business_id} at {initial_stage.value}")
        return state

    async def get_state(self, business_id: str) -> Optional[LifecycleState]:
        """Get current lifecycle state."""
        return self._states.get(business_id)

    async def transition(
        self,
        business_id: str,
        to_stage: LifecycleStage,
        trigger: TransitionTrigger = TransitionTrigger.MANUAL,
        triggered_by: str = "system",
        notes: str = "",
        force: bool = False,
    ) -> tuple[bool, str]:
        """Attempt to transition to a new lifecycle stage."""
        state = self._states.get(business_id)
        if not state:
            return False, "Business not found"

        # Check if transition is valid
        config = STAGE_CONFIG.get(state.current_stage, {})
        valid_next = config.get("next_stages", [])
        
        if to_stage not in valid_next and not force:
            return False, f"Cannot transition from {state.current_stage.value} to {to_stage.value}"

        # Check requirements for new stage
        if not force:
            requirements_met, blockers = await self._check_requirements(business_id, to_stage)
            if not requirements_met:
                state.blockers = blockers
                return False, f"Requirements not met: {', '.join(blockers)}"

        # Execute pre-transition hooks
        for hook in self._hooks["pre_transition"]:
            try:
                await hook(business_id, state.current_stage, to_stage)
            except Exception as e:
                logger.error(f"Pre-transition hook error: {e}")

        # Create transition record
        transition = LifecycleTransition(
            id=str(uuid.uuid4()),
            from_stage=state.current_stage,
            to_stage=to_stage,
            trigger=trigger,
            triggered_by=triggered_by,
            notes=notes,
        )

        # Update state
        state.transitions.append(transition)
        state.current_stage = to_stage
        state.entered_at = datetime.utcnow()
        state.milestones = self._create_stage_milestones(to_stage)
        state.next_stage = self._get_next_stage(to_stage)
        state.blockers = []

        # Execute post-transition hooks
        for hook in self._hooks["post_transition"]:
            try:
                await hook(business_id, transition)
            except Exception as e:
                logger.error(f"Post-transition hook error: {e}")

        logger.info(f"Business {business_id} transitioned to {to_stage.value}")
        return True, f"Transitioned to {to_stage.value}"

    async def update_milestone(
        self,
        business_id: str,
        milestone_id: str,
        current_value: float,
    ) -> Optional[Milestone]:
        """Update milestone progress."""
        state = self._states.get(business_id)
        if not state:
            return None

        for milestone in state.milestones:
            if milestone.id == milestone_id:
                milestone.current_value = current_value
                
                if not milestone.achieved and current_value >= milestone.target_value:
                    milestone.achieved = True
                    milestone.achieved_at = datetime.utcnow()
                    
                    # Execute milestone hooks
                    for hook in self._hooks["milestone_achieved"]:
                        try:
                            await hook(business_id, milestone)
                        except Exception as e:
                            logger.error(f"Milestone hook error: {e}")
                    
                    # Check for automatic transition
                    await self._check_auto_transition(business_id)
                
                return milestone
        
        return None

    async def add_milestone(
        self,
        business_id: str,
        name: str,
        milestone_type: MilestoneType,
        target_value: float,
        target_date: Optional[str] = None,
    ) -> Optional[Milestone]:
        """Add a custom milestone."""
        state = self._states.get(business_id)
        if not state:
            return None

        milestone = Milestone(
            id=str(uuid.uuid4()),
            name=name,
            milestone_type=milestone_type,
            description=f"Custom milestone: {name}",
            target_value=target_value,
            target_date=datetime.fromisoformat(target_date).date() if target_date else None,
        )
        state.milestones.append(milestone)
        return milestone

    async def calculate_health(self, business_id: str) -> float:
        """Calculate business health score."""
        state = self._states.get(business_id)
        if not state:
            return 0.0

        score = 100.0
        
        # Factor 1: Milestone progress
        if state.milestones:
            achieved = sum(1 for m in state.milestones if m.achieved)
            milestone_score = (achieved / len(state.milestones)) * 30
            score = score - 30 + milestone_score

        # Factor 2: Time in stage
        config = STAGE_CONFIG.get(state.current_stage, {})
        typical_duration = config.get("typical_duration_days")
        if typical_duration:
            days_in_stage = (datetime.utcnow() - state.entered_at).days
            if days_in_stage > typical_duration * 1.5:
                score -= 20  # Overstaying in stage

        # Factor 3: Overdue milestones
        overdue = sum(1 for m in state.milestones if m.is_overdue)
        score -= overdue * 10

        # Factor 4: Blockers
        score -= len(state.blockers) * 5

        state.health_score = max(0, min(100, score))

        # Trigger warning if health is low
        if state.health_score < 50:
            for hook in self._hooks["health_warning"]:
                try:
                    await hook(business_id, state.health_score)
                except Exception as e:
                    logger.error(f"Health warning hook error: {e}")

        return state.health_score

    async def get_recommendations(self, business_id: str) -> list[dict]:
        """Get recommendations for business progress."""
        state = self._states.get(business_id)
        if not state:
            return []

        recommendations = []

        # Check incomplete milestones
        for milestone in state.milestones:
            if not milestone.achieved:
                progress = milestone.progress_percent
                if progress < 50:
                    recommendations.append({
                        "type": "milestone",
                        "priority": "high" if milestone.is_overdue else "medium",
                        "title": f"Complete: {milestone.name}",
                        "description": f"Progress: {progress:.0f}% - Target: {milestone.target_value}",
                        "milestone_id": milestone.id,
                    })

        # Check stage duration
        config = STAGE_CONFIG.get(state.current_stage, {})
        typical_duration = config.get("typical_duration_days")
        if typical_duration:
            days_in_stage = (datetime.utcnow() - state.entered_at).days
            if days_in_stage > typical_duration:
                recommendations.append({
                    "type": "stage",
                    "priority": "high",
                    "title": "Consider stage advancement",
                    "description": f"You've been in {state.current_stage.value} for {days_in_stage} days (typical: {typical_duration})",
                })

        # Check blockers
        for blocker in state.blockers:
            recommendations.append({
                "type": "blocker",
                "priority": "high",
                "title": "Address blocker",
                "description": blocker,
            })

        return recommendations

    async def _check_requirements(
        self, business_id: str, stage: LifecycleStage
    ) -> tuple[bool, list[str]]:
        """Check if requirements for a stage are met."""
        config = STAGE_CONFIG.get(stage, {})
        requirements = config.get("requirements", [])
        blockers = []

        for req in requirements:
            provider = self._metric_providers.get(req.metric_name)
            if provider:
                try:
                    current_value = await provider(business_id)
                    if not req.evaluate(current_value):
                        blockers.append(req.description)
                except Exception as e:
                    blockers.append(f"Could not evaluate {req.metric_name}: {e}")
            else:
                blockers.append(f"No provider for {req.metric_name}")

        return len(blockers) == 0, blockers

    async def _check_auto_transition(self, business_id: str):
        """Check if business should auto-transition based on milestones."""
        state = self._states.get(business_id)
        if not state:
            return

        # All milestones achieved?
        if state.milestones and all(m.achieved for m in state.milestones):
            next_stage = self._get_next_stage(state.current_stage)
            if next_stage:
                requirements_met, _ = await self._check_requirements(business_id, next_stage)
                if requirements_met:
                    await self.transition(
                        business_id,
                        next_stage,
                        trigger=TransitionTrigger.MILESTONE,
                        triggered_by="all_milestones_complete",
                    )

    def _create_stage_milestones(self, stage: LifecycleStage) -> list[Milestone]:
        """Create milestones for a stage."""
        config = STAGE_CONFIG.get(stage, {})
        milestone_defs = config.get("milestones", [])
        
        milestones = []
        for m in milestone_defs:
            milestones.append(Milestone(
                id=str(uuid.uuid4()),
                name=m["name"],
                milestone_type=m["type"],
                description=f"Stage milestone: {m['name']}",
                target_value=m["target"],
            ))
        return milestones

    def _get_next_stage(self, stage: LifecycleStage) -> Optional[LifecycleStage]:
        """Get the primary next stage."""
        config = STAGE_CONFIG.get(stage, {})
        next_stages = config.get("next_stages", [])
        return next_stages[0] if next_stages else None

    def get_stage_info(self, stage: LifecycleStage) -> dict:
        """Get information about a stage."""
        config = STAGE_CONFIG.get(stage, {})
        return {
            "stage": stage.value,
            "description": config.get("description", ""),
            "typical_duration_days": config.get("typical_duration_days"),
            "next_stages": [s.value for s in config.get("next_stages", [])],
            "requirement_count": len(config.get("requirements", [])),
            "milestone_count": len(config.get("milestones", [])),
        }
