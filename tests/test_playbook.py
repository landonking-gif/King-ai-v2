"""Tests for Playbook System."""
import pytest
from src.business.playbook_models import (
    PlaybookDefinition, PlaybookType, TaskDefinition, TaskStatus
)
from src.business.playbook_loader import PlaybookLoader
from src.business.playbook_executor import PlaybookExecutor


@pytest.fixture
def loader():
    return PlaybookLoader()


@pytest.fixture
def executor():
    return PlaybookExecutor()


@pytest.fixture
def sample_playbook():
    return PlaybookDefinition(
        id="test_pb",
        name="Test Playbook",
        playbook_type=PlaybookType.CUSTOM,
        description="Test",
        tasks=[
            TaskDefinition(
                id="task_1",
                name="First Task",
                description="",
                agent="test_agent",
                action="test_action",
            ),
            TaskDefinition(
                id="task_2",
                name="Second Task",
                description="",
                agent="test_agent",
                action="test_action",
                dependencies=["task_1"],
            ),
        ],
    )


class TestPlaybookLoader:
    def test_load_from_template(self, loader):
        pb = loader.load_from_template(PlaybookType.DROPSHIPPING)
        assert pb is not None
        assert pb.playbook_type == PlaybookType.DROPSHIPPING
        assert len(pb.tasks) > 0

    def test_load_saas_template(self, loader):
        pb = loader.load_from_template(PlaybookType.SAAS)
        assert pb is not None
        assert "SaaS" in pb.name

    def test_validate_playbook(self, loader, sample_playbook):
        valid, errors = loader.validate_playbook(sample_playbook)
        assert valid is True
        assert len(errors) == 0

    def test_validate_invalid_dependency(self, loader):
        pb = PlaybookDefinition(
            id="bad_pb",
            name="Bad",
            playbook_type=PlaybookType.CUSTOM,
            description="",
            tasks=[
                TaskDefinition(
                    id="task_1",
                    name="Task",
                    description="",
                    agent="test",
                    action="test",
                    dependencies=["nonexistent"],
                ),
            ],
        )
        valid, errors = loader.validate_playbook(pb)
        assert valid is False
        assert len(errors) > 0


class MockAgent:
    async def execute(self, action, context):
        return {"success": True, "action": action}


class TestPlaybookExecutor:
    @pytest.mark.asyncio
    async def test_execute_playbook(self, executor, sample_playbook):
        executor.register_agent("test_agent", MockAgent())
        
        run = await executor.execute(sample_playbook, "biz_1")
        
        assert run.status == TaskStatus.COMPLETED
        assert run.progress_percent == 100.0

    @pytest.mark.asyncio
    async def test_task_dependencies(self, executor, sample_playbook):
        executor.register_agent("test_agent", MockAgent())
        
        run = await executor.execute(sample_playbook, "biz_1")
        
        task_1 = run.task_executions["task_1"]
        task_2 = run.task_executions["task_2"]
        
        assert task_1.status == TaskStatus.COMPLETED
        assert task_2.status == TaskStatus.COMPLETED
        # Task 2 should complete after Task 1
        assert task_2.started_at >= task_1.completed_at

    @pytest.mark.asyncio
    async def test_get_run(self, executor, sample_playbook):
        executor.register_agent("test_agent", MockAgent())
        
        run = await executor.execute(sample_playbook, "biz_1")
        retrieved = await executor.get_run(run.id)
        
        assert retrieved is not None
        assert retrieved.id == run.id

    @pytest.mark.asyncio
    async def test_cancel_run(self, executor):
        # Can't really test cancellation mid-run without async tricks
        result = await executor.cancel_run("nonexistent")
        assert result is False
