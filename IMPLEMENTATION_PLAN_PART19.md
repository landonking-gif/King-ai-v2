# Implementation Plan Part 19: Business Lifecycle Engine

| Field | Value |
|-------|-------|
| Module | Business Lifecycle Management Engine |
| Priority | High |
| Estimated Effort | 5-6 hours |
| Dependencies | Part 3 (Database), Part 4 (Business Unit), Part 17 (Analytics) |

---

## 1. Scope

This module implements the business lifecycle management system:

- **Lifecycle States** - Define business stages from inception to maturity
- **State Machine** - Manage transitions between lifecycle phases
- **Milestone Tracking** - Track key business milestones and achievements
- **Automated Actions** - Trigger actions based on lifecycle events
- **Health Monitoring** - Continuous business health assessment

---

## 2. Tasks

### Task 19.1: Lifecycle Models

**File: `src/business/lifecycle_models.py`**

```python
"""
Business Lifecycle Models.
"""
from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
from typing import Any, Optional, Callable


class LifecycleStage(Enum):
    """Business lifecycle stages."""
    IDEATION = "ideation"
    VALIDATION = "validation"
    LAUNCH = "launch"
    GROWTH = "growth"
    SCALE = "scale"
    MATURITY = "maturity"
    DECLINE = "decline"
    EXIT = "exit"


class MilestoneType(Enum):
    """Types of business milestones."""
    REVENUE = "revenue"
    CUSTOMER = "customer"
    PRODUCT = "product"
    OPERATIONAL = "operational"
    FUNDING = "funding"
    TEAM = "team"


class TransitionTrigger(Enum):
    """What triggers a lifecycle transition."""
    MANUAL = "manual"
    MILESTONE = "milestone"
    METRIC = "metric"
    TIME = "time"
    AUTOMATED = "automated"


@dataclass
class Milestone:
    """Business milestone definition."""
    id: str
    name: str
    milestone_type: MilestoneType
    description: str
    target_value: float
    current_value: float = 0.0
    achieved: bool = False
    achieved_at: Optional[datetime] = None
    target_date: Optional[date] = None
    metadata: dict = field(default_factory=dict)

    @property
    def progress_percent(self) -> float:
        if self.target_value == 0:
            return 0.0
        return min(100, (self.current_value / self.target_value) * 100)

    @property
    def is_overdue(self) -> bool:
        if not self.target_date or self.achieved:
            return False
        return date.today() > self.target_date


@dataclass
class StageRequirement:
    """Requirements to enter a lifecycle stage."""
    metric_name: str
    operator: str  # gte, lte, eq, gt, lt
    value: float
    description: str = ""

    def evaluate(self, current_value: float) -> bool:
        ops = {
            "gte": lambda a, b: a >= b,
            "lte": lambda a, b: a <= b,
            "gt": lambda a, b: a > b,
            "lt": lambda a, b: a < b,
            "eq": lambda a, b: a == b,
        }
        return ops.get(self.operator, lambda a, b: False)(current_value, self.value)


@dataclass
class LifecycleTransition:
    """Record of a lifecycle transition."""
    id: str
    from_stage: LifecycleStage
    to_stage: LifecycleStage
    trigger: TransitionTrigger
    triggered_by: str  # milestone_id, metric_name, or user_id
    timestamp: datetime = field(default_factory=datetime.utcnow)
    notes: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class LifecycleState:
    """Current lifecycle state of a business."""
    business_id: str
    current_stage: LifecycleStage
    entered_at: datetime
    milestones: list[Milestone] = field(default_factory=list)
    transitions: list[LifecycleTransition] = field(default_factory=list)
    health_score: float = 100.0
    next_stage: Optional[LifecycleStage] = None
    blockers: list[str] = field(default_factory=list)


# Stage configuration with requirements and milestones
STAGE_CONFIG = {
    LifecycleStage.IDEATION: {
        "description": "Idea development and initial planning",
        "typical_duration_days": 30,
        "requirements": [],
        "milestones": [
            {"name": "Business Plan Complete", "type": MilestoneType.OPERATIONAL, "target": 1},
            {"name": "Market Research Done", "type": MilestoneType.OPERATIONAL, "target": 1},
        ],
        "next_stages": [LifecycleStage.VALIDATION],
    },
    LifecycleStage.VALIDATION: {
        "description": "Market validation and MVP development",
        "typical_duration_days": 60,
        "requirements": [
            StageRequirement("business_plan_complete", "eq", 1, "Business plan required"),
        ],
        "milestones": [
            {"name": "MVP Launched", "type": MilestoneType.PRODUCT, "target": 1},
            {"name": "First 10 Customers", "type": MilestoneType.CUSTOMER, "target": 10},
            {"name": "First Revenue", "type": MilestoneType.REVENUE, "target": 1},
        ],
        "next_stages": [LifecycleStage.LAUNCH, LifecycleStage.IDEATION],
    },
    LifecycleStage.LAUNCH: {
        "description": "Official launch and initial traction",
        "typical_duration_days": 90,
        "requirements": [
            StageRequirement("customers", "gte", 10, "Minimum 10 customers"),
            StageRequirement("has_revenue", "eq", 1, "Must have revenue"),
        ],
        "milestones": [
            {"name": "100 Customers", "type": MilestoneType.CUSTOMER, "target": 100},
            {"name": "$1K MRR", "type": MilestoneType.REVENUE, "target": 1000},
        ],
        "next_stages": [LifecycleStage.GROWTH],
    },
    LifecycleStage.GROWTH: {
        "description": "Rapid growth and market expansion",
        "typical_duration_days": 180,
        "requirements": [
            StageRequirement("mrr", "gte", 1000, "Minimum $1K MRR"),
            StageRequirement("customers", "gte", 100, "Minimum 100 customers"),
        ],
        "milestones": [
            {"name": "1000 Customers", "type": MilestoneType.CUSTOMER, "target": 1000},
            {"name": "$10K MRR", "type": MilestoneType.REVENUE, "target": 10000},
            {"name": "Team of 5", "type": MilestoneType.TEAM, "target": 5},
        ],
        "next_stages": [LifecycleStage.SCALE],
    },
    LifecycleStage.SCALE: {
        "description": "Scaling operations and market dominance",
        "typical_duration_days": 365,
        "requirements": [
            StageRequirement("mrr", "gte", 10000, "Minimum $10K MRR"),
            StageRequirement("growth_rate", "gte", 10, "Minimum 10% monthly growth"),
        ],
        "milestones": [
            {"name": "$100K MRR", "type": MilestoneType.REVENUE, "target": 100000},
            {"name": "10K Customers", "type": MilestoneType.CUSTOMER, "target": 10000},
        ],
        "next_stages": [LifecycleStage.MATURITY],
    },
    LifecycleStage.MATURITY: {
        "description": "Stable, profitable operations",
        "typical_duration_days": None,
        "requirements": [
            StageRequirement("mrr", "gte", 100000, "Minimum $100K MRR"),
            StageRequirement("profit_margin", "gte", 20, "Minimum 20% margin"),
        ],
        "milestones": [],
        "next_stages": [LifecycleStage.EXIT, LifecycleStage.DECLINE],
    },
    LifecycleStage.DECLINE: {
        "description": "Declining metrics, requires intervention",
        "typical_duration_days": None,
        "requirements": [],
        "milestones": [],
        "next_stages": [LifecycleStage.GROWTH, LifecycleStage.EXIT],
    },
    LifecycleStage.EXIT: {
        "description": "Business exit (sale, shutdown, or transition)",
        "typical_duration_days": None,
        "requirements": [],
        "milestones": [],
        "next_stages": [],
    },
}
```

---

### Task 19.2: Lifecycle Engine

**File: `src/business/lifecycle.py`**

```python
"""
Business Lifecycle Engine - State management and transitions.
"""
import uuid
from datetime import datetime, timedelta
from typing import Any, Optional, Callable
from src.business.lifecycle_models import (
    LifecycleStage, LifecycleState, LifecycleTransition,
    Milestone, MilestoneType, TransitionTrigger, StageRequirement,
    STAGE_CONFIG
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


class LifecycleEngine:
    """Engine for managing business lifecycle transitions."""

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
```

---

### Task 19.3: Lifecycle API Routes

**File: `src/api/routes/lifecycle.py`**

```python
"""
Lifecycle API Routes.
"""
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from src.business.lifecycle import LifecycleEngine
from src.business.lifecycle_models import LifecycleStage, MilestoneType, TransitionTrigger
from src.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/lifecycle", tags=["lifecycle"])

_engine: Optional[LifecycleEngine] = None


def get_engine() -> LifecycleEngine:
    global _engine
    if _engine is None:
        _engine = LifecycleEngine()
    return _engine


class InitRequest(BaseModel):
    business_id: str
    initial_stage: str = "ideation"


class TransitionRequest(BaseModel):
    business_id: str
    to_stage: str
    notes: str = ""
    force: bool = False


class MilestoneUpdateRequest(BaseModel):
    business_id: str
    milestone_id: str
    current_value: float


class AddMilestoneRequest(BaseModel):
    business_id: str
    name: str
    milestone_type: str
    target_value: float = Field(..., gt=0)
    target_date: Optional[str] = None


@router.post("/init")
async def initialize_business(req: InitRequest, engine: LifecycleEngine = Depends(get_engine)):
    """Initialize lifecycle tracking for a business."""
    try:
        stage = LifecycleStage(req.initial_stage)
    except ValueError:
        raise HTTPException(400, f"Invalid stage: {req.initial_stage}")
    
    state = await engine.initialize_business(req.business_id, stage)
    return {
        "business_id": state.business_id,
        "stage": state.current_stage.value,
        "milestones": len(state.milestones),
    }


@router.get("/state/{business_id}")
async def get_state(business_id: str, engine: LifecycleEngine = Depends(get_engine)):
    """Get current lifecycle state."""
    state = await engine.get_state(business_id)
    if not state:
        raise HTTPException(404, "Business not found")
    
    return {
        "business_id": state.business_id,
        "stage": state.current_stage.value,
        "entered_at": state.entered_at.isoformat(),
        "health_score": state.health_score,
        "next_stage": state.next_stage.value if state.next_stage else None,
        "blockers": state.blockers,
        "milestones": [
            {
                "id": m.id,
                "name": m.name,
                "type": m.milestone_type.value,
                "target": m.target_value,
                "current": m.current_value,
                "progress": m.progress_percent,
                "achieved": m.achieved,
                "overdue": m.is_overdue,
            }
            for m in state.milestones
        ],
        "transition_count": len(state.transitions),
    }


@router.post("/transition")
async def transition(req: TransitionRequest, engine: LifecycleEngine = Depends(get_engine)):
    """Transition to a new lifecycle stage."""
    try:
        to_stage = LifecycleStage(req.to_stage)
    except ValueError:
        raise HTTPException(400, f"Invalid stage: {req.to_stage}")
    
    success, message = await engine.transition(
        req.business_id,
        to_stage,
        TransitionTrigger.MANUAL,
        "api_user",
        req.notes,
        req.force,
    )
    
    if not success:
        raise HTTPException(400, message)
    
    return {"status": "ok", "message": message}


@router.post("/milestones/update")
async def update_milestone(req: MilestoneUpdateRequest, engine: LifecycleEngine = Depends(get_engine)):
    """Update milestone progress."""
    milestone = await engine.update_milestone(
        req.business_id, req.milestone_id, req.current_value
    )
    if not milestone:
        raise HTTPException(404, "Milestone not found")
    
    return {
        "milestone_id": milestone.id,
        "name": milestone.name,
        "progress": milestone.progress_percent,
        "achieved": milestone.achieved,
    }


@router.post("/milestones/add")
async def add_milestone(req: AddMilestoneRequest, engine: LifecycleEngine = Depends(get_engine)):
    """Add a custom milestone."""
    try:
        m_type = MilestoneType(req.milestone_type)
    except ValueError:
        raise HTTPException(400, f"Invalid type: {req.milestone_type}")
    
    milestone = await engine.add_milestone(
        req.business_id, req.name, m_type, req.target_value, req.target_date
    )
    if not milestone:
        raise HTTPException(404, "Business not found")
    
    return {"milestone_id": milestone.id, "name": milestone.name}


@router.get("/health/{business_id}")
async def get_health(business_id: str, engine: LifecycleEngine = Depends(get_engine)):
    """Calculate and return business health score."""
    score = await engine.calculate_health(business_id)
    return {"business_id": business_id, "health_score": score}


@router.get("/recommendations/{business_id}")
async def get_recommendations(business_id: str, engine: LifecycleEngine = Depends(get_engine)):
    """Get recommendations for business progress."""
    recommendations = await engine.get_recommendations(business_id)
    return {"recommendations": recommendations, "count": len(recommendations)}


@router.get("/stages")
async def list_stages(engine: LifecycleEngine = Depends(get_engine)):
    """List all lifecycle stages with info."""
    stages = [engine.get_stage_info(stage) for stage in LifecycleStage]
    return {"stages": stages}


@router.get("/stages/{stage}")
async def get_stage_info(stage: str, engine: LifecycleEngine = Depends(get_engine)):
    """Get information about a specific stage."""
    try:
        lifecycle_stage = LifecycleStage(stage)
    except ValueError:
        raise HTTPException(400, f"Invalid stage: {stage}")
    
    return engine.get_stage_info(lifecycle_stage)
```

---

### Task 19.4: Tests

**File: `tests/test_lifecycle.py`**

```python
"""Tests for Lifecycle Engine."""
import pytest
from datetime import datetime, timedelta
from src.business.lifecycle_models import (
    LifecycleStage, Milestone, MilestoneType, StageRequirement
)
from src.business.lifecycle import LifecycleEngine


@pytest.fixture
def engine():
    return LifecycleEngine()


class TestMilestone:
    def test_progress_percent(self):
        m = Milestone(
            id="m1", name="Test", milestone_type=MilestoneType.REVENUE,
            description="Test", target_value=100, current_value=50
        )
        assert m.progress_percent == 50.0

    def test_achieved_at_target(self):
        m = Milestone(
            id="m1", name="Test", milestone_type=MilestoneType.REVENUE,
            description="Test", target_value=100, current_value=100
        )
        assert m.progress_percent == 100.0


class TestStageRequirement:
    def test_gte_operator(self):
        req = StageRequirement("mrr", "gte", 1000)
        assert req.evaluate(1000) is True
        assert req.evaluate(1500) is True
        assert req.evaluate(500) is False

    def test_lt_operator(self):
        req = StageRequirement("churn", "lt", 5)
        assert req.evaluate(3) is True
        assert req.evaluate(5) is False


class TestLifecycleEngine:
    @pytest.mark.asyncio
    async def test_initialize_business(self, engine):
        state = await engine.initialize_business("biz_1")
        assert state.business_id == "biz_1"
        assert state.current_stage == LifecycleStage.IDEATION
        assert len(state.milestones) > 0

    @pytest.mark.asyncio
    async def test_get_state(self, engine):
        await engine.initialize_business("biz_1")
        state = await engine.get_state("biz_1")
        assert state is not None
        assert state.business_id == "biz_1"

    @pytest.mark.asyncio
    async def test_valid_transition(self, engine):
        await engine.initialize_business("biz_1")
        success, msg = await engine.transition(
            "biz_1", LifecycleStage.VALIDATION, force=True
        )
        assert success is True
        
        state = await engine.get_state("biz_1")
        assert state.current_stage == LifecycleStage.VALIDATION

    @pytest.mark.asyncio
    async def test_invalid_transition(self, engine):
        await engine.initialize_business("biz_1")
        success, msg = await engine.transition(
            "biz_1", LifecycleStage.SCALE  # Can't jump to SCALE from IDEATION
        )
        assert success is False

    @pytest.mark.asyncio
    async def test_update_milestone(self, engine):
        state = await engine.initialize_business("biz_1")
        milestone = state.milestones[0]
        
        updated = await engine.update_milestone("biz_1", milestone.id, milestone.target_value)
        assert updated.achieved is True

    @pytest.mark.asyncio
    async def test_add_custom_milestone(self, engine):
        await engine.initialize_business("biz_1")
        milestone = await engine.add_milestone(
            "biz_1", "Custom Goal", MilestoneType.REVENUE, 5000
        )
        assert milestone is not None
        assert milestone.name == "Custom Goal"

    @pytest.mark.asyncio
    async def test_calculate_health(self, engine):
        await engine.initialize_business("biz_1")
        score = await engine.calculate_health("biz_1")
        assert 0 <= score <= 100

    @pytest.mark.asyncio
    async def test_get_recommendations(self, engine):
        await engine.initialize_business("biz_1")
        recs = await engine.get_recommendations("biz_1")
        assert isinstance(recs, list)

    def test_get_stage_info(self, engine):
        info = engine.get_stage_info(LifecycleStage.GROWTH)
        assert info["stage"] == "growth"
        assert "description" in info
```

---

## 3. Acceptance Criteria

| Criteria | Validation |
|----------|------------|
| States initialized | Business starts in ideation stage |
| Transitions work | Valid transitions allowed, invalid blocked |
| Milestones tracked | Progress updates and achievements |
| Health calculated | Score reflects business status |
| Recommendations generated | Actionable next steps |
| Hooks execute | Pre/post transition callbacks work |

---

## 4. File Summary

| File | Purpose |
|------|---------|
| `src/business/lifecycle_models.py` | Data models and stage config |
| `src/business/lifecycle.py` | Lifecycle engine implementation |
| `src/api/routes/lifecycle.py` | REST API endpoints |
| `tests/test_lifecycle.py` | Unit tests |
