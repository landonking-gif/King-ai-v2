"""
Tests for planning and execution system.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime

from src.master_ai.planning_models import (
    ExecutionPlan, PlanTask, TaskStatus, RiskLevel, PlanningContext
)
from src.master_ai.react_planner import ReActPlanner
from src.master_ai.plan_executor import PlanExecutor
from src.master_ai.planner import Planner


class TestPlanTask:
    """Tests for PlanTask model."""
    
    def test_can_execute_no_dependencies(self):
        """Task with no dependencies can execute."""
        task = PlanTask(name="Test", description="Test", agent="research")
        assert task.can_execute(set())
    
    def test_can_execute_with_satisfied_dependencies(self):
        """Task executes when dependencies are met."""
        task = PlanTask(
            name="Test",
            description="Test",
            agent="research",
            depends_on=["task-1", "task-2"]
        )
        assert task.can_execute({"task-1", "task-2", "task-3"})
    
    def test_cannot_execute_with_unsatisfied_dependencies(self):
        """Task blocks when dependencies aren't met."""
        task = PlanTask(
            name="Test",
            description="Test",
            agent="research",
            depends_on=["task-1", "task-2"]
        )
        assert not task.can_execute({"task-1"})
    
    def test_duration_calculation(self):
        """Test duration calculation."""
        task = PlanTask(
            name="Test",
            description="Test",
            agent="research",
            started_at=datetime(2024, 1, 1, 12, 0, 0),
            completed_at=datetime(2024, 1, 1, 12, 0, 5)
        )
        assert task.duration_ms == 5000.0


class TestExecutionPlan:
    """Tests for ExecutionPlan model."""
    
    def test_get_ready_tasks(self):
        """Get tasks ready for execution."""
        plan = ExecutionPlan(goal="Test")
        
        task1 = PlanTask(id="1", name="First", description="", agent="research")
        task2 = PlanTask(id="2", name="Second", description="", agent="commerce", depends_on=["1"])
        
        plan.tasks = [task1, task2]
        
        ready = plan.get_ready_tasks()
        assert len(ready) == 1
        assert ready[0].id == "1"
    
    def test_get_next_task_priority(self):
        """Next task respects priority."""
        plan = ExecutionPlan(goal="Test")
        
        task1 = PlanTask(id="1", name="Low", description="", agent="research", priority=10)
        task2 = PlanTask(id="2", name="High", description="", agent="research", priority=1)
        
        plan.tasks = [task1, task2]
        
        next_task = plan.get_next_task()
        assert next_task.id == "2"
    
    def test_update_metrics(self):
        """Metrics update correctly."""
        plan = ExecutionPlan(goal="Test")
        
        task1 = PlanTask(id="1", name="Done", description="", agent="research", status=TaskStatus.COMPLETED)
        task2 = PlanTask(id="2", name="Failed", description="", agent="commerce", status=TaskStatus.FAILED)
        task3 = PlanTask(id="3", name="Pending", description="", agent="content")
        
        plan.tasks = [task1, task2, task3]
        plan.update_metrics()
        
        assert plan.total_tasks == 3
        assert plan.completed_tasks == 1
        assert plan.failed_tasks == 1
    
    def test_get_task_by_id(self):
        """Test getting task by ID."""
        plan = ExecutionPlan(goal="Test")
        task1 = PlanTask(id="test-1", name="Test", description="", agent="research")
        plan.tasks = [task1]
        
        found = plan.get_task("test-1")
        assert found is not None
        assert found.id == "test-1"
        
        not_found = plan.get_task("nonexistent")
        assert not_found is None


class TestReActPlanner:
    """Tests for ReAct planner."""
    
    @pytest.fixture
    def planner(self):
        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(return_value='''
        {
            "tasks": [
                {"name": "Research", "description": "Do research", "agent": "research", "priority": 1, "duration_minutes": 10, "risk_level": "low"},
                {"name": "Setup", "description": "Setup store", "agent": "commerce", "priority": 2, "duration_minutes": 15, "risk_level": "medium"}
            ]
        }
        ''')
        return ReActPlanner(mock_llm)
    
    @pytest.mark.asyncio
    async def test_create_plan(self, planner):
        """Test plan creation."""
        plan = await planner.create_plan(
            goal="Start a business",
            context="Empire context"
        )
        
        assert plan.status == "ready"
        assert len(plan.tasks) == 2
        assert plan.overall_risk in [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]
    
    @pytest.mark.asyncio
    async def test_dependency_building(self, planner):
        """Test that dependencies are built."""
        plan = await planner.create_plan(
            goal="Start a business",
            context="Empire context"
        )
        
        # Commerce should depend on research
        commerce_task = next((t for t in plan.tasks if t.agent == "commerce"), None)
        research_task = next((t for t in plan.tasks if t.agent == "research"), None)
        
        if commerce_task and research_task:
            assert research_task.id in commerce_task.depends_on
    
    def test_parse_json_response(self, planner):
        """Test JSON parsing from LLM responses."""
        # Test with markdown code block
        response1 = '```json\n{"tasks": []}\n```'
        result1 = planner._parse_json_response(response1)
        assert result1 == {"tasks": []}
        
        # Test with plain code block
        response2 = '```\n{"tasks": []}\n```'
        result2 = planner._parse_json_response(response2)
        assert result2 == {"tasks": []}
        
        # Test with plain JSON
        response3 = '{"tasks": []}'
        result3 = planner._parse_json_response(response3)
        assert result3 == {"tasks": []}
    
    def test_build_dependencies(self, planner):
        """Test dependency building logic."""
        tasks = [
            PlanTask(id="1", name="Research", description="", agent="research"),
            PlanTask(id="2", name="Commerce", description="", agent="commerce"),
            PlanTask(id="3", name="Content", description="", agent="content"),
        ]
        
        result = planner._build_dependencies(tasks)
        
        # Commerce should depend on research
        commerce = next(t for t in result if t.agent == "commerce")
        research = next(t for t in result if t.agent == "research")
        assert research.id in commerce.depends_on
    
    def test_optimize_order(self, planner):
        """Test topological sort of tasks."""
        task1 = PlanTask(id="1", name="First", description="", agent="research", priority=1)
        task2 = PlanTask(id="2", name="Second", description="", agent="commerce", priority=2, depends_on=["1"])
        task3 = PlanTask(id="3", name="Third", description="", agent="content", priority=3, depends_on=["2"])
        
        # Build blocks
        task1.blocks = ["2"]
        task2.blocks = ["3"]
        
        tasks = [task3, task1, task2]  # Intentionally out of order
        result = planner._optimize_order(tasks)
        
        # Should be ordered by dependencies
        assert result[0].id == "1"
        assert result[1].id == "2"
        assert result[2].id == "3"


class TestPlanExecutor:
    """Tests for plan executor."""
    
    @pytest.fixture
    def executor(self):
        mock_planner = MagicMock()
        mock_router = MagicMock()
        mock_router.execute = AsyncMock(return_value={"success": True, "output": "Done"})
        
        return PlanExecutor(
            planner=mock_planner,
            agent_router=mock_router
        )
    
    @pytest.mark.asyncio
    async def test_execute_simple_plan(self, executor):
        """Test executing a simple plan."""
        plan = ExecutionPlan(goal="Test")
        task = PlanTask(
            id="1",
            name="Simple task",
            description="Test",
            agent="research"
        )
        plan.tasks = [task]
        plan.update_metrics()
        
        with patch('src.master_ai.plan_executor.get_db') as mock_db:
            mock_session = AsyncMock()
            mock_session.get = AsyncMock(return_value=None)
            mock_session.add = MagicMock()
            mock_session.commit = AsyncMock()
            mock_db.return_value.__aenter__.return_value = mock_session
            
            result = await executor.execute_plan(plan)
        
        assert result.status == "completed"
        assert result.completed_tasks == 1
    
    @pytest.mark.asyncio
    async def test_approval_pauses_plan(self, executor):
        """Test that approval requirement pauses plan."""
        plan = ExecutionPlan(goal="Test")
        task = PlanTask(
            id="1",
            name="Needs approval",
            description="Test",
            agent="finance",
            requires_approval=True
        )
        plan.tasks = [task]
        plan.update_metrics()
        
        with patch('src.master_ai.plan_executor.get_db') as mock_db:
            mock_session = AsyncMock()
            mock_session.add = MagicMock()
            mock_session.commit = AsyncMock()
            mock_db.return_value.__aenter__.return_value = mock_session
            
            result = await executor.execute_plan(plan, auto_continue=False)
        
        assert result.status == "paused"
        assert task.status == TaskStatus.WAITING_APPROVAL
    
    @pytest.mark.asyncio
    async def test_task_dependencies_respected(self, executor):
        """Test that task dependencies are respected."""
        plan = ExecutionPlan(goal="Test")
        task1 = PlanTask(id="1", name="First", description="", agent="research")
        task2 = PlanTask(id="2", name="Second", description="", agent="commerce", depends_on=["1"])
        
        plan.tasks = [task1, task2]
        plan.update_metrics()
        
        # Next task should be task1 since task2 depends on it
        next_task = plan.get_next_task()
        assert next_task.id == "1"
    
    def test_get_plan_status(self, executor):
        """Test getting plan status."""
        plan = ExecutionPlan(goal="Test Goal", status="executing")
        task1 = PlanTask(id="1", name="Task 1", description="", agent="research", status=TaskStatus.COMPLETED)
        task2 = PlanTask(id="2", name="Task 2", description="", agent="commerce", status=TaskStatus.WAITING_APPROVAL, requires_approval=True, approval_reason="High cost")
        
        plan.tasks = [task1, task2]
        plan.update_metrics()
        executor._active_plans[plan.id] = plan
        
        status = executor.get_plan_status(plan.id)
        
        assert status is not None
        assert status["goal"] == "Test Goal"
        assert status["status"] == "executing"
        assert status["progress"]["total"] == 2
        assert status["progress"]["completed"] == 1
        assert len(status["pending_approvals"]) == 1


class TestPlannerBackwardCompatibility:
    """Tests for Planner backward compatibility."""
    
    @pytest.fixture
    def planner(self):
        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(return_value='''
        {
            "tasks": [
                {"name": "Task 1", "description": "Test", "agent": "research", "priority": 1, "duration_minutes": 5, "risk_level": "low"}
            ]
        }
        ''')
        return Planner(mock_llm)
    
    @pytest.mark.asyncio
    async def test_legacy_format_conversion(self, planner):
        """Test that plan is converted to legacy format."""
        result = await planner.create_plan(
            goal="Test goal",
            context="Test context"
        )
        
        # Check legacy format fields
        assert "goal" in result
        assert "steps" in result
        assert "status" in result
        assert isinstance(result["steps"], list)
        
        if len(result["steps"]) > 0:
            step = result["steps"][0]
            assert "name" in step
            assert "description" in step
            assert "agent" in step
            assert "type" in step  # Backward compatibility alias
    
    @pytest.mark.asyncio
    async def test_fallback_on_error(self, planner):
        """Test fallback plan creation on error."""
        planner.react_planner.create_plan = AsyncMock(side_effect=Exception("Test error"))
        
        result = await planner.create_plan(
            goal="Test goal",
            context="Test context"
        )
        
        assert result["status"] == "fallback"
        assert len(result["steps"]) == 1
        assert result["steps"][0]["requires_approval"] is True
