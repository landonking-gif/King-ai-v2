"""Tests for the Scheduler Service."""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

from src.services.scheduler import (
    Scheduler, ScheduledTask, TaskFrequency, TaskStatus
)


class TestTaskFrequency:
    """Test TaskFrequency enum."""
    
    def test_frequency_values(self):
        """Test that all frequency values are defined."""
        assert TaskFrequency.EVERY_MINUTE.value == "every_minute"
        assert TaskFrequency.HOURLY.value == "hourly"
        assert TaskFrequency.EVERY_6_HOURS.value == "every_6_hours"
        assert TaskFrequency.DAILY.value == "daily"
        assert TaskFrequency.WEEKLY.value == "weekly"


class TestScheduledTask:
    """Test ScheduledTask dataclass."""
    
    def test_create_task(self):
        """Test creating a scheduled task."""
        callback = AsyncMock()
        task = ScheduledTask(
            id="test-id",
            name="test_task",
            frequency=TaskFrequency.HOURLY,
            callback=callback
        )
        
        assert task.id == "test-id"
        assert task.name == "test_task"
        assert task.frequency == TaskFrequency.HOURLY
        assert task.enabled is True
        assert task.error_count == 0
        assert task.run_count == 0


class TestScheduler:
    """Test Scheduler class."""
    
    @pytest.fixture
    def scheduler(self):
        """Create a fresh scheduler for each test."""
        return Scheduler()
    
    def test_register_task(self, scheduler):
        """Test registering a task."""
        callback = AsyncMock()
        task_id = scheduler.register_task(
            name="test_task",
            callback=callback,
            frequency=TaskFrequency.HOURLY
        )
        
        assert task_id is not None
        assert len(scheduler._tasks) == 1
        
        task = scheduler.get_task(task_id)
        assert task is not None
        assert task.name == "test_task"
        assert task.frequency == TaskFrequency.HOURLY
    
    def test_register_task_run_immediately(self, scheduler):
        """Test registering a task that runs immediately."""
        callback = AsyncMock()
        task_id = scheduler.register_task(
            name="immediate_task",
            callback=callback,
            frequency=TaskFrequency.DAILY,
            run_immediately=True
        )
        
        task = scheduler.get_task(task_id)
        assert task.next_run is not None
        # Should be scheduled for now (or very close to now)
        assert task.next_run <= datetime.utcnow() + timedelta(seconds=5)
    
    def test_unregister_task(self, scheduler):
        """Test unregistering a task."""
        callback = AsyncMock()
        task_id = scheduler.register_task(
            name="temp_task",
            callback=callback,
            frequency=TaskFrequency.HOURLY
        )
        
        assert len(scheduler._tasks) == 1
        result = scheduler.unregister_task(task_id)
        assert result is True
        assert len(scheduler._tasks) == 0
    
    def test_unregister_nonexistent_task(self, scheduler):
        """Test unregistering a non-existent task."""
        result = scheduler.unregister_task("nonexistent-id")
        assert result is False
    
    def test_enable_task(self, scheduler):
        """Test enabling a task."""
        callback = AsyncMock()
        task_id = scheduler.register_task(
            name="disabled_task",
            callback=callback,
            frequency=TaskFrequency.HOURLY,
            enabled=False
        )
        
        task = scheduler.get_task(task_id)
        assert task.enabled is False
        
        scheduler.enable_task(task_id)
        assert task.enabled is True
    
    def test_disable_task(self, scheduler):
        """Test disabling a task."""
        callback = AsyncMock()
        task_id = scheduler.register_task(
            name="active_task",
            callback=callback,
            frequency=TaskFrequency.HOURLY,
            enabled=True
        )
        
        task = scheduler.get_task(task_id)
        assert task.enabled is True
        
        scheduler.disable_task(task_id)
        assert task.enabled is False
        assert task.status == TaskStatus.DISABLED
    
    def test_get_task_by_name(self, scheduler):
        """Test getting a task by name."""
        callback = AsyncMock()
        scheduler.register_task(
            name="named_task",
            callback=callback,
            frequency=TaskFrequency.HOURLY
        )
        
        task = scheduler.get_task_by_name("named_task")
        assert task is not None
        assert task.name == "named_task"
        
        missing = scheduler.get_task_by_name("nonexistent")
        assert missing is None
    
    def test_get_status(self, scheduler):
        """Test getting scheduler status."""
        callback = AsyncMock()
        scheduler.register_task(
            name="task1",
            callback=callback,
            frequency=TaskFrequency.HOURLY
        )
        scheduler.register_task(
            name="task2",
            callback=callback,
            frequency=TaskFrequency.DAILY,
            enabled=False
        )
        
        status = scheduler.get_status()
        assert status["running"] is False
        assert status["total_tasks"] == 2
        assert status["active_tasks"] == 1
        assert len(status["tasks"]) == 2
    
    def test_list_tasks(self, scheduler):
        """Test listing all tasks."""
        callback = AsyncMock()
        scheduler.register_task(
            name="task1",
            callback=callback,
            frequency=TaskFrequency.HOURLY
        )
        
        tasks = scheduler.list_tasks()
        assert len(tasks) == 1
        assert tasks[0]["name"] == "task1"
        assert "id" in tasks[0]
        assert "frequency" in tasks[0]
        assert "enabled" in tasks[0]
    
    @pytest.mark.asyncio
    async def test_execute_task_success(self, scheduler):
        """Test successful task execution."""
        callback = AsyncMock()
        task_id = scheduler.register_task(
            name="success_task",
            callback=callback,
            frequency=TaskFrequency.HOURLY
        )
        
        task = scheduler.get_task(task_id)
        await scheduler._execute_task(task)
        
        callback.assert_called_once()
        assert task.run_count == 1
        assert task.error_count == 0
        assert task.status == TaskStatus.COMPLETED
        assert task.last_run is not None
    
    @pytest.mark.asyncio
    async def test_execute_task_failure(self, scheduler):
        """Test task execution with error."""
        callback = AsyncMock(side_effect=Exception("Test error"))
        task_id = scheduler.register_task(
            name="failing_task",
            callback=callback,
            frequency=TaskFrequency.HOURLY
        )
        
        task = scheduler.get_task(task_id)
        await scheduler._execute_task(task)
        
        assert task.error_count == 1
        # After a failure with remaining retries, status goes to PENDING for retry
        assert task.status == TaskStatus.PENDING
        assert task.last_error == "Test error"
    
    @pytest.mark.asyncio
    async def test_task_disabled_after_max_errors(self, scheduler):
        """Test that task is disabled after max errors."""
        callback = AsyncMock(side_effect=Exception("Persistent error"))
        task_id = scheduler.register_task(
            name="unstable_task",
            callback=callback,
            frequency=TaskFrequency.HOURLY
        )
        
        task = scheduler.get_task(task_id)
        task.max_errors = 3
        
        # Execute until max errors
        for _ in range(3):
            await scheduler._execute_task(task)
        
        assert task.error_count == 3
        assert task.enabled is False
        assert task.status == TaskStatus.DISABLED
    
    @pytest.mark.asyncio
    async def test_run_task_now(self, scheduler):
        """Test manually running a task."""
        callback = AsyncMock()
        task_id = scheduler.register_task(
            name="manual_task",
            callback=callback,
            frequency=TaskFrequency.DAILY
        )
        
        result = await scheduler.run_task_now(task_id)
        assert result is True
        callback.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_run_task_now_nonexistent(self, scheduler):
        """Test running non-existent task."""
        result = await scheduler.run_task_now("nonexistent-id")
        assert result is False
    
    @pytest.mark.asyncio
    async def test_start_stop(self, scheduler):
        """Test starting and stopping scheduler."""
        assert scheduler._running is False
        
        await scheduler.start()
        assert scheduler._running is True
        
        await scheduler.stop()
        assert scheduler._running is False
    
    @pytest.mark.asyncio
    async def test_start_idempotent(self, scheduler):
        """Test that starting twice doesn't create multiple loops."""
        await scheduler.start()
        task1 = scheduler._task
        
        await scheduler.start()  # Should be no-op
        task2 = scheduler._task
        
        assert task1 is task2
        
        await scheduler.stop()
    
    def test_task_to_dict(self, scheduler):
        """Test task serialization."""
        callback = AsyncMock()
        task_id = scheduler.register_task(
            name="serialize_task",
            callback=callback,
            frequency=TaskFrequency.HOURLY,
            metadata={"key": "value"}
        )
        
        task = scheduler.get_task(task_id)
        task.run_count = 5
        task.total_duration_ms = 1000
        
        data = scheduler._task_to_dict(task)
        
        assert data["id"] == task_id
        assert data["name"] == "serialize_task"
        assert data["frequency"] == "hourly"
        assert data["run_count"] == 5
        assert data["avg_duration_ms"] == 200.0
        assert data["metadata"] == {"key": "value"}


class TestSchedulerIntegration:
    """Integration tests for scheduler."""
    
    @pytest.mark.asyncio
    async def test_scheduler_executes_due_tasks(self):
        """Test that scheduler executes tasks when they're due."""
        scheduler = Scheduler()
        scheduler._check_interval = 0.1  # Check quickly for test
        
        executed = []
        
        async def task_callback():
            executed.append(datetime.utcnow())
        
        # Register task that should run immediately
        scheduler.register_task(
            name="immediate",
            callback=task_callback,
            frequency=TaskFrequency.EVERY_MINUTE,
            run_immediately=True
        )
        
        await scheduler.start()
        await asyncio.sleep(0.3)  # Wait for execution
        await scheduler.stop()
        
        assert len(executed) >= 1
