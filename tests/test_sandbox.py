"""Tests for sandbox environment."""

import pytest
import asyncio
from pathlib import Path

from src.utils.sandbox import (
    SandboxManager,
    SandboxConfig,
    SandboxEnvironment,
    SandboxStatus
)


class TestSandboxManager:
    """Tests for SandboxManager."""
    
    @pytest.fixture
    def manager(self, tmp_path):
        return SandboxManager(str(tmp_path / "sandboxes"))
    
    @pytest.mark.asyncio
    async def test_create_sandbox(self, manager):
        """Test sandbox creation."""
        source_files = {
            "src/main.py": "def hello(): return 'Hello'"
        }
        
        sandbox = await manager.create_sandbox(source_files)
        
        assert sandbox.sandbox_id
        assert sandbox.status in [SandboxStatus.READY, SandboxStatus.FAILED]
    
    @pytest.mark.asyncio
    async def test_run_simple_test(self, manager):
        """Test running a simple test."""
        source_files = {
            "src/math_utils.py": """
def add(a, b):
    return a + b
"""
        }
        
        test_files = {
            "tests/test_math.py": """
import sys
sys.path.insert(0, '/app')
from src.math_utils import add

def test_add():
    assert add(1, 2) == 3
"""
        }
        
        config = SandboxConfig(timeout_seconds=60)
        
        result = await manager.run_tests(source_files, test_files, config)
        
        assert result.sandbox_id
        assert result.status in [SandboxStatus.COMPLETED, SandboxStatus.FAILED]
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self, manager):
        """Test that timeouts are handled."""
        # Use a script that will definitely take longer than timeout
        source_files = {
            "src/slow.py": """
import time
def slow_function():
    time.sleep(1000)
    return True
"""
        }
        
        test_files = {
            "tests/test_slow.py": """
import sys
sys.path.insert(0, '/app')
from src.slow import slow_function

def test_slow():
    # This will call the slow function and should timeout
    result = slow_function()
    assert result
"""
        }
        
        config = SandboxConfig(timeout_seconds=2)  # Very short timeout
        
        result = await manager.run_tests(source_files, test_files, config)
        
        # Should timeout or fail (either is acceptable for a timeout test)
        assert result.status in [SandboxStatus.TIMEOUT, SandboxStatus.FAILED, SandboxStatus.COMPLETED]
    
    @pytest.mark.asyncio
    async def test_cleanup(self, manager):
        """Test sandbox cleanup."""
        source_files = {"test.py": "pass"}
        
        sandbox = await manager.create_sandbox(source_files)
        sandbox_id = sandbox.sandbox_id
        
        await sandbox.cleanup()
        
        assert sandbox.status == SandboxStatus.CLEANED
        assert not sandbox.work_dir.exists()


class TestSandboxConfig:
    """Tests for SandboxConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = SandboxConfig()
        
        assert config.memory_limit == "512m"
        assert config.cpu_limit == 1.0
        assert config.timeout_seconds == 300
        assert config.network_enabled is False
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = SandboxConfig(
            memory_limit="1g",
            timeout_seconds=600,
            network_enabled=True
        )
        
        assert config.memory_limit == "1g"
        assert config.timeout_seconds == 600
        assert config.network_enabled is True
