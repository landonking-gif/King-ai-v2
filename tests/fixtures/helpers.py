"""
Test Helpers and Utilities.
Common functions and decorators for testing.
"""

import asyncio
import functools
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, TypeVar
import uuid
import pytest

from tests.fixtures.factories import (
    BusinessFactory,
    ApprovalRequestFactory,
    UserFactory,
    PlanFactory,
    TaskFactory,
)


T = TypeVar('T')


def async_test(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator to run async test functions.
    
    Usage:
        @async_test
        async def test_something():
            result = await async_function()
            assert result == expected
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return asyncio.get_event_loop().run_until_complete(func(*args, **kwargs))
    return wrapper


def parametrize_async(*parametrize_args, **parametrize_kwargs):
    """
    Combine pytest.mark.parametrize with async_test.
    
    Usage:
        @parametrize_async("input,expected", [(1, 2), (2, 4)])
        async def test_double(input, expected):
            assert await double(input) == expected
    """
    def decorator(func):
        @pytest.mark.parametrize(*parametrize_args, **parametrize_kwargs)
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return asyncio.get_event_loop().run_until_complete(func(*args, **kwargs))
        return wrapper
    return decorator


@asynccontextmanager
async def create_test_database():
    """
    Create a temporary test database.
    
    Usage:
        async with create_test_database() as db:
            await db.execute("SELECT 1")
    """
    from tests.fixtures.mocks import MockDatabaseSession
    
    db = MockDatabaseSession()
    try:
        yield db
    finally:
        await db.close()


async def cleanup_test_database(db) -> None:
    """Clean up test database."""
    if hasattr(db, 'rollback'):
        await db.rollback()
    if hasattr(db, 'close'):
        await db.close()


def create_test_business(**kwargs) -> Dict[str, Any]:
    """
    Create a test business with sensible defaults.
    
    Usage:
        business = create_test_business(name="My Business")
    """
    return BusinessFactory.create(**kwargs)


def create_test_approval(**kwargs) -> Dict[str, Any]:
    """
    Create a test approval request with sensible defaults.
    
    Usage:
        approval = create_test_approval(risk_level="high")
    """
    return ApprovalRequestFactory.create(**kwargs)


def create_test_user(**kwargs) -> Dict[str, Any]:
    """
    Create a test user with sensible defaults.
    
    Usage:
        user = create_test_user(role="admin")
    """
    return UserFactory.create(**kwargs)


class TestContext:
    """
    Context manager for test setup and teardown.
    
    Usage:
        async with TestContext() as ctx:
            ctx.add_business()
            ctx.add_approval()
            # Test code here
    """
    
    def __init__(self):
        self.businesses: List[Dict[str, Any]] = []
        self.users: List[Dict[str, Any]] = []
        self.approvals: List[Dict[str, Any]] = []
        self.plans: List[Dict[str, Any]] = []
        self.tasks: List[Dict[str, Any]] = []
        self._cleanup_callbacks: List[Callable] = []
    
    def add_business(self, **kwargs) -> Dict[str, Any]:
        """Add a test business."""
        business = BusinessFactory.create(**kwargs)
        self.businesses.append(business)
        return business
    
    def add_user(self, **kwargs) -> Dict[str, Any]:
        """Add a test user."""
        user = UserFactory.create(**kwargs)
        self.users.append(user)
        return user
    
    def add_approval(self, **kwargs) -> Dict[str, Any]:
        """Add a test approval."""
        approval = ApprovalRequestFactory.create(**kwargs)
        self.approvals.append(approval)
        return approval
    
    def add_plan(self, **kwargs) -> Dict[str, Any]:
        """Add a test plan."""
        plan = PlanFactory.create(**kwargs)
        self.plans.append(plan)
        return plan
    
    def add_task(self, **kwargs) -> Dict[str, Any]:
        """Add a test task."""
        task = TaskFactory.create(**kwargs)
        self.tasks.append(task)
        return task
    
    def on_cleanup(self, callback: Callable) -> None:
        """Register a cleanup callback."""
        self._cleanup_callbacks.append(callback)
    
    async def __aenter__(self) -> "TestContext":
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        for callback in reversed(self._cleanup_callbacks):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback()
                else:
                    callback()
            except Exception:
                pass


class MockTime:
    """
    Mock time for testing time-dependent code.
    
    Usage:
        with MockTime() as time:
            time.advance(hours=1)
            # Code that uses datetime.utcnow()
    """
    
    def __init__(self, start_time: Optional[datetime] = None):
        self.current_time = start_time or datetime.utcnow()
        self._original_utcnow = None
    
    def advance(
        self,
        seconds: int = 0,
        minutes: int = 0,
        hours: int = 0,
        days: int = 0,
    ) -> datetime:
        """Advance the mock time."""
        self.current_time += timedelta(
            seconds=seconds,
            minutes=minutes,
            hours=hours,
            days=days,
        )
        return self.current_time
    
    def set(self, time: datetime) -> None:
        """Set the mock time."""
        self.current_time = time
    
    def __enter__(self) -> "MockTime":
        import datetime as dt
        self._original_utcnow = dt.datetime.utcnow
        
        def mock_utcnow():
            return self.current_time
        
        dt.datetime.utcnow = staticmethod(mock_utcnow)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        import datetime as dt
        if self._original_utcnow:
            dt.datetime.utcnow = self._original_utcnow


def assert_dict_contains(actual: Dict, expected: Dict) -> None:
    """
    Assert that actual dict contains all key-value pairs from expected.
    
    Usage:
        assert_dict_contains(
            {"a": 1, "b": 2, "c": 3},
            {"a": 1, "b": 2}
        )
    """
    for key, value in expected.items():
        assert key in actual, f"Missing key: {key}"
        assert actual[key] == value, f"Key {key}: expected {value}, got {actual[key]}"


def assert_raises_async(exception_type: type):
    """
    Assert that an async function raises an exception.
    
    Usage:
        async with assert_raises_async(ValueError):
            await some_function()
    """
    class AsyncExceptionContext:
        def __init__(self, expected):
            self.expected = expected
            self.exception = None
        
        async def __aenter__(self):
            return self
        
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            if exc_type is None:
                raise AssertionError(f"Expected {self.expected.__name__} to be raised")
            if not issubclass(exc_type, self.expected):
                raise AssertionError(
                    f"Expected {self.expected.__name__}, got {exc_type.__name__}"
                )
            self.exception = exc_val
            return True
    
    return AsyncExceptionContext(exception_type)


def generate_test_id() -> str:
    """Generate a unique test ID."""
    return f"test_{uuid.uuid4().hex[:8]}"


def retry_test(max_attempts: int = 3, delay: float = 0.1):
    """
    Decorator to retry flaky tests.
    
    Usage:
        @retry_test(max_attempts=3)
        def test_flaky_operation():
            ...
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except AssertionError as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        import time
                        time.sleep(delay)
            raise last_exception
        return wrapper
    return decorator


class CapturedLogs:
    """
    Capture log messages for testing.
    
    Usage:
        with CapturedLogs() as logs:
            logger.info("Test message")
        assert "Test message" in logs.messages
    """
    
    def __init__(self, logger_name: Optional[str] = None):
        self.logger_name = logger_name
        self.messages: List[str] = []
        self.records: List[Any] = []
        self._handler = None
    
    def __enter__(self) -> "CapturedLogs":
        import logging
        
        class CaptureHandler(logging.Handler):
            def __init__(self, captured: "CapturedLogs"):
                super().__init__()
                self.captured = captured
            
            def emit(self, record):
                self.captured.messages.append(record.getMessage())
                self.captured.records.append(record)
        
        self._handler = CaptureHandler(self)
        
        if self.logger_name:
            logger = logging.getLogger(self.logger_name)
        else:
            logger = logging.getLogger()
        
        logger.addHandler(self._handler)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        import logging
        
        if self.logger_name:
            logger = logging.getLogger(self.logger_name)
        else:
            logger = logging.getLogger()
        
        logger.removeHandler(self._handler)
    
    def contains(self, text: str) -> bool:
        """Check if any message contains the text."""
        return any(text in msg for msg in self.messages)
    
    def count(self, text: str) -> int:
        """Count messages containing the text."""
        return sum(1 for msg in self.messages if text in msg)


# Pytest fixtures as functions
@pytest.fixture
def test_business():
    """Pytest fixture for a test business."""
    return create_test_business()


@pytest.fixture
def test_user():
    """Pytest fixture for a test user."""
    return create_test_user()


@pytest.fixture
def test_approval():
    """Pytest fixture for a test approval."""
    return create_test_approval()


@pytest.fixture
async def test_context():
    """Pytest fixture for test context."""
    async with TestContext() as ctx:
        yield ctx
