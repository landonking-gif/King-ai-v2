"""Tests for Lifecycle Engine."""
import pytest
from datetime import datetime, timedelta
from src.business.lifecycle_models import (
    LifecycleStage, Milestone, MilestoneType, StageRequirement
)
from src.business.lifecycle import EnhancedLifecycleEngine, BasicLifecycleEngine
from src.database.models import BusinessStatus


@pytest.fixture
def engine():
    return EnhancedLifecycleEngine()


@pytest.fixture
def basic_engine():
    return BasicLifecycleEngine()


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

    def test_progress_percent_zero_target(self):
        m = Milestone(
            id="m1", name="Test", milestone_type=MilestoneType.REVENUE,
            description="Test", target_value=0, current_value=10
        )
        assert m.progress_percent == 0.0

    def test_progress_percent_capped_at_100(self):
        m = Milestone(
            id="m1", name="Test", milestone_type=MilestoneType.REVENUE,
            description="Test", target_value=100, current_value=150
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

    def test_eq_operator(self):
        req = StageRequirement("status", "eq", 1)
        assert req.evaluate(1) is True
        assert req.evaluate(0) is False

    def test_gt_operator(self):
        req = StageRequirement("revenue", "gt", 100)
        assert req.evaluate(101) is True
        assert req.evaluate(100) is False

    def test_lte_operator(self):
        req = StageRequirement("cost", "lte", 50)
        assert req.evaluate(50) is True
        assert req.evaluate(49) is True
        assert req.evaluate(51) is False

    def test_invalid_operator(self):
        req = StageRequirement("test", "invalid", 10)
        assert req.evaluate(10) is False


class TestBasicLifecycleEngine:
    def test_lifecycle_transitions(self, basic_engine):
        # Valid transition
        next_step = basic_engine.get_next_status(BusinessStatus.DISCOVERY)
        assert next_step == BusinessStatus.VALIDATION
        
        # Late stage transition
        next_step = basic_engine.get_next_status(BusinessStatus.OPTIMIZATION)
        assert next_step == BusinessStatus.REPLICATION
        
        # End state should return None
        next_step = basic_engine.get_next_status(BusinessStatus.REPLICATION)
        assert next_step is None

    def test_is_failed(self, basic_engine):
        assert basic_engine.is_failed(BusinessStatus.SUNSET) is True
        assert basic_engine.is_failed(BusinessStatus.OPERATION) is False


class TestEnhancedLifecycleEngine:
    @pytest.mark.asyncio
    async def test_initialize_business(self, engine):
        state = await engine.initialize_business("biz_1")
        assert state.business_id == "biz_1"
        assert state.current_stage == LifecycleStage.IDEATION
        assert len(state.milestones) > 0

    @pytest.mark.asyncio
    async def test_initialize_business_custom_stage(self, engine):
        state = await engine.initialize_business("biz_2", LifecycleStage.VALIDATION)
        assert state.business_id == "biz_2"
        assert state.current_stage == LifecycleStage.VALIDATION

    @pytest.mark.asyncio
    async def test_get_state(self, engine):
        await engine.initialize_business("biz_1")
        state = await engine.get_state("biz_1")
        assert state is not None
        assert state.business_id == "biz_1"

    @pytest.mark.asyncio
    async def test_get_state_not_found(self, engine):
        state = await engine.get_state("nonexistent")
        assert state is None

    @pytest.mark.asyncio
    async def test_valid_transition_with_force(self, engine):
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
        assert "Cannot transition" in msg

    @pytest.mark.asyncio
    async def test_transition_not_found_business(self, engine):
        success, msg = await engine.transition(
            "nonexistent", LifecycleStage.VALIDATION
        )
        assert success is False
        assert "not found" in msg

    @pytest.mark.asyncio
    async def test_update_milestone(self, engine):
        state = await engine.initialize_business("biz_1")
        milestone = state.milestones[0]
        
        updated = await engine.update_milestone("biz_1", milestone.id, milestone.target_value)
        assert updated.achieved is True
        assert updated.achieved_at is not None

    @pytest.mark.asyncio
    async def test_update_milestone_partial_progress(self, engine):
        state = await engine.initialize_business("biz_1")
        milestone = state.milestones[0]
        
        updated = await engine.update_milestone("biz_1", milestone.id, milestone.target_value / 2)
        assert updated.achieved is False
        assert updated.progress_percent == 50.0

    @pytest.mark.asyncio
    async def test_update_milestone_not_found(self, engine):
        await engine.initialize_business("biz_1")
        updated = await engine.update_milestone("biz_1", "invalid_id", 100)
        assert updated is None

    @pytest.mark.asyncio
    async def test_add_custom_milestone(self, engine):
        await engine.initialize_business("biz_1")
        milestone = await engine.add_milestone(
            "biz_1", "Custom Goal", MilestoneType.REVENUE, 5000
        )
        assert milestone is not None
        assert milestone.name == "Custom Goal"
        assert milestone.target_value == 5000

    @pytest.mark.asyncio
    async def test_add_milestone_business_not_found(self, engine):
        milestone = await engine.add_milestone(
            "nonexistent", "Custom Goal", MilestoneType.REVENUE, 5000
        )
        assert milestone is None

    @pytest.mark.asyncio
    async def test_calculate_health(self, engine):
        await engine.initialize_business("biz_1")
        score = await engine.calculate_health("biz_1")
        assert 0 <= score <= 100

    @pytest.mark.asyncio
    async def test_calculate_health_not_found(self, engine):
        score = await engine.calculate_health("nonexistent")
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_get_recommendations(self, engine):
        await engine.initialize_business("biz_1")
        recs = await engine.get_recommendations("biz_1")
        assert isinstance(recs, list)

    @pytest.mark.asyncio
    async def test_get_recommendations_not_found(self, engine):
        recs = await engine.get_recommendations("nonexistent")
        assert recs == []

    def test_get_stage_info(self, engine):
        info = engine.get_stage_info(LifecycleStage.GROWTH)
        assert info["stage"] == "growth"
        assert "description" in info
        assert "typical_duration_days" in info
        assert "next_stages" in info

    def test_get_stage_info_all_stages(self, engine):
        for stage in LifecycleStage:
            info = engine.get_stage_info(stage)
            assert info["stage"] == stage.value

    @pytest.mark.asyncio
    async def test_register_hook(self, engine):
        hook_called = []
        
        async def test_hook(business_id, transition):
            hook_called.append(business_id)
        
        engine.register_hook("post_transition", test_hook)
        await engine.initialize_business("biz_1")
        await engine.transition("biz_1", LifecycleStage.VALIDATION, force=True)
        
        assert "biz_1" in hook_called

    @pytest.mark.asyncio
    async def test_register_metric_provider(self, engine):
        async def customer_count(business_id):
            return 15
        
        engine.register_metric_provider("customers", customer_count)
        await engine.initialize_business("biz_1")
        
        # Try to transition to LAUNCH which requires customers >= 10
        await engine.transition("biz_1", LifecycleStage.VALIDATION, force=True)
        success, msg = await engine.transition("biz_1", LifecycleStage.LAUNCH)
        # Should fail because has_revenue requirement is not met
        assert success is False

    @pytest.mark.asyncio
    async def test_transition_history(self, engine):
        await engine.initialize_business("biz_1")
        await engine.transition("biz_1", LifecycleStage.VALIDATION, force=True)
        await engine.transition("biz_1", LifecycleStage.LAUNCH, force=True)
        
        state = await engine.get_state("biz_1")
        assert len(state.transitions) == 2
        assert state.transitions[0].from_stage == LifecycleStage.IDEATION
        assert state.transitions[0].to_stage == LifecycleStage.VALIDATION
        assert state.transitions[1].from_stage == LifecycleStage.VALIDATION
        assert state.transitions[1].to_stage == LifecycleStage.LAUNCH
