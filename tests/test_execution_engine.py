"""
Tests for the Execution Engine and Business Creation capabilities.

These tests verify that:
1. Actions are actually executed (not hallucinated)
2. Verification works correctly
3. Business creation produces real files
4. All outputs can be confirmed to exist
"""

import os
import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

from src.services.execution_engine import (
    ExecutionEngine, ActionRequest, ActionType, ExecutionStatus,
    get_execution_engine, create_file, run_command, http_get
)
from src.services.business_creation import (
    BusinessCreationEngine, BusinessType, BusinessStage,
    get_business_engine
)


class TestExecutionEngine:
    """Tests for the core execution engine."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace for testing."""
        temp_dir = tempfile.mkdtemp(prefix="king_ai_test_")
        yield temp_dir
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def engine(self, temp_workspace):
        """Create an execution engine with temp workspace."""
        return ExecutionEngine(temp_workspace)
    
    @pytest.mark.asyncio
    async def test_file_create_verified(self, engine, temp_workspace):
        """Test that file creation is verified."""
        test_content = "Hello, this is a test file!"
        request = ActionRequest(
            action_type=ActionType.FILE_CREATE,
            params={
                "path": "test_file.txt",
                "content": test_content
            },
            require_verification=True,
            description="Create test file"
        )
        
        result = await engine.execute(request)
        
        # Check execution succeeded
        assert result.success, f"Execution failed: {result.error}"
        assert result.status == ExecutionStatus.VERIFIED
        
        # Verify file actually exists
        file_path = Path(temp_workspace) / "test_file.txt"
        assert file_path.exists(), "File was not actually created!"
        
        # Verify content matches
        actual_content = file_path.read_text()
        assert actual_content == test_content, "File content doesn't match!"
    
    @pytest.mark.asyncio
    async def test_file_create_with_nested_directory(self, engine, temp_workspace):
        """Test file creation in nested directories."""
        request = ActionRequest(
            action_type=ActionType.FILE_CREATE,
            params={
                "path": "deep/nested/path/file.txt",
                "content": "Nested content"
            },
            require_verification=True
        )
        
        result = await engine.execute(request)
        
        assert result.success
        assert result.status == ExecutionStatus.VERIFIED
        
        # Verify the nested file exists
        file_path = Path(temp_workspace) / "deep/nested/path/file.txt"
        assert file_path.exists()
    
    @pytest.mark.asyncio
    async def test_directory_create_verified(self, engine, temp_workspace):
        """Test directory creation with verification."""
        request = ActionRequest(
            action_type=ActionType.DIR_CREATE,
            params={"path": "new_directory/subdirectory"},
            require_verification=True
        )
        
        result = await engine.execute(request)
        
        assert result.success
        assert result.status == ExecutionStatus.VERIFIED
        
        # Verify directory exists
        dir_path = Path(temp_workspace) / "new_directory/subdirectory"
        assert dir_path.exists()
        assert dir_path.is_dir()
    
    @pytest.mark.asyncio
    async def test_file_read(self, engine, temp_workspace):
        """Test file reading."""
        # First create a file
        test_content = "Content to read"
        file_path = Path(temp_workspace) / "readable.txt"
        file_path.write_text(test_content)
        
        # Now read it
        request = ActionRequest(
            action_type=ActionType.FILE_READ,
            params={"path": str(file_path)},
            require_verification=False
        )
        
        result = await engine.execute(request)
        
        assert result.success
        assert result.output["content"] == test_content
    
    @pytest.mark.asyncio
    async def test_file_delete_verified(self, engine, temp_workspace):
        """Test file deletion with verification."""
        # Create a file first
        file_path = Path(temp_workspace) / "to_delete.txt"
        file_path.write_text("Delete me")
        assert file_path.exists()
        
        # Delete it
        request = ActionRequest(
            action_type=ActionType.FILE_DELETE,
            params={"path": str(file_path)},
            require_verification=True
        )
        
        result = await engine.execute(request)
        
        assert result.success
        assert result.status == ExecutionStatus.VERIFIED
        assert not file_path.exists(), "File was not actually deleted!"
    
    @pytest.mark.asyncio
    async def test_shell_command_success(self, engine, temp_workspace):
        """Test shell command execution."""
        # Use a simple, cross-platform command
        request = ActionRequest(
            action_type=ActionType.SHELL_COMMAND,
            params={
                "command": "echo Hello World",
                "cwd": temp_workspace
            },
            require_verification=True
        )
        
        result = await engine.execute(request)
        
        assert result.success
        assert "Hello World" in result.output.get("stdout", "")
    
    @pytest.mark.asyncio
    async def test_shell_command_dangerous_blocked(self, engine, temp_workspace):
        """Test that dangerous commands are blocked."""
        request = ActionRequest(
            action_type=ActionType.SHELL_COMMAND,
            params={"command": "rm -rf /"},
            require_verification=True
        )
        
        result = await engine.execute(request)
        
        assert not result.success
        assert "dangerous" in result.error.lower() or "blocked" in result.error.lower()
    
    @pytest.mark.asyncio
    async def test_directory_list(self, engine, temp_workspace):
        """Test directory listing."""
        # Create some files
        (Path(temp_workspace) / "file1.txt").write_text("1")
        (Path(temp_workspace) / "file2.txt").write_text("2")
        (Path(temp_workspace) / "subdir").mkdir()
        
        request = ActionRequest(
            action_type=ActionType.DIR_LIST,
            params={"path": temp_workspace},
            require_verification=False
        )
        
        result = await engine.execute(request)
        
        assert result.success
        assert result.output["count"] >= 3
        names = [item["name"] for item in result.output["items"]]
        assert "file1.txt" in names
        assert "file2.txt" in names
        assert "subdir" in names
    
    @pytest.mark.asyncio
    async def test_http_request(self, engine):
        """Test HTTP request execution."""
        request = ActionRequest(
            action_type=ActionType.HTTP_REQUEST,
            params={
                "url": "https://httpbin.org/get",
                "method": "GET",
                "timeout": 10
            },
            require_verification=True
        )
        
        result = await engine.execute(request)
        
        # Note: This test requires internet connectivity
        if result.success:
            assert result.output["status_code"] == 200
    
    @pytest.mark.asyncio
    async def test_execution_log(self, engine, temp_workspace):
        """Test that execution log is maintained."""
        # Execute multiple actions
        await engine.execute(ActionRequest(
            action_type=ActionType.DIR_CREATE,
            params={"path": "dir1"}
        ))
        await engine.execute(ActionRequest(
            action_type=ActionType.FILE_CREATE,
            params={"path": "file1.txt", "content": "test"}
        ))
        
        log = engine.get_execution_log()
        
        assert len(log) >= 2
        assert all("action_id" in entry for entry in log)
        assert all("status" in entry for entry in log)
    
    @pytest.mark.asyncio
    async def test_success_rate(self, engine, temp_workspace):
        """Test success rate calculation."""
        # Execute some successful actions
        await engine.execute(ActionRequest(
            action_type=ActionType.DIR_CREATE,
            params={"path": "success1"}
        ))
        await engine.execute(ActionRequest(
            action_type=ActionType.DIR_CREATE,
            params={"path": "success2"}
        ))
        
        rate = engine.get_success_rate()
        assert rate > 0


class TestBusinessCreation:
    """Tests for the business creation engine."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace for testing."""
        temp_dir = tempfile.mkdtemp(prefix="king_ai_biz_test_")
        yield temp_dir
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def biz_engine(self, temp_workspace):
        """Create a business engine with temp workspace."""
        return BusinessCreationEngine(temp_workspace)
    
    @pytest.mark.asyncio
    async def test_create_business_structure(self, biz_engine, temp_workspace):
        """Test that business creation creates real directories."""
        business = await biz_engine.create_business(
            business_type=BusinessType.DROPSHIPPING,
            name="Test Dropship Store",
            description="A test dropshipping business",
            target_market="Test customers",
            budget=500.0
        )
        
        assert business is not None
        assert business.business_id is not None
        assert business.stage == BusinessStage.PLANNING
        
        # Verify directories were actually created
        biz_path = Path(temp_workspace) / "businesses" / business.business_id
        assert biz_path.exists(), "Business directory was not created!"
        assert (biz_path / "documents").exists()
        assert (biz_path / "website").exists()
        assert (biz_path / "marketing").exists()
        assert (biz_path / "financials").exists()
    
    @pytest.mark.asyncio
    async def test_create_business_plan_file(self, biz_engine, temp_workspace):
        """Test that business plan file is created and verified."""
        business = await biz_engine.create_business(
            business_type=BusinessType.SAAS,
            name="Test SaaS App",
            description="A test SaaS application",
            budget=2000.0
        )
        
        # Verify business plan file exists
        plan_path = Path(temp_workspace) / "businesses" / business.business_id / "business_plan.json"
        assert plan_path.exists(), "Business plan file was not created!"
        
        # Verify it's valid JSON
        import json
        with open(plan_path) as f:
            plan_data = json.load(f)
        
        assert plan_data["name"] == "Test SaaS App"
        assert plan_data["business_type"] == "saas"
    
    @pytest.mark.asyncio
    async def test_create_landing_page(self, biz_engine, temp_workspace):
        """Test landing page creation and verification."""
        business = await biz_engine.create_business(
            business_type=BusinessType.ECOMMERCE,
            name="Test E-Store",
            description="An online store for testing"
        )
        
        asset = await biz_engine.create_landing_page(business.business_id)
        
        assert asset is not None
        assert asset.verified, "Landing page was not verified!"
        assert asset.asset_type == "landing_page"
        
        # Verify file actually exists
        page_path = Path(temp_workspace) / "businesses" / business.business_id / "website" / "index.html"
        assert page_path.exists(), "Landing page file was not created!"
        
        # Verify it contains HTML
        content = page_path.read_text()
        assert "<html" in content.lower()
        assert "Test E-Store" in content
    
    @pytest.mark.asyncio
    async def test_create_business_documents(self, biz_engine, temp_workspace):
        """Test document creation and verification."""
        business = await biz_engine.create_business(
            business_type=BusinessType.CONSULTING,
            name="Test Consulting",
            description="A consulting business"
        )
        
        assets = await biz_engine.create_business_documents(business.business_id)
        
        assert len(assets) >= 2
        
        # Check each asset is verified
        for asset in assets:
            assert asset.verified, f"Asset {asset.name} was not verified!"
        
        # Verify files exist
        docs_path = Path(temp_workspace) / "businesses" / business.business_id / "documents"
        assert (docs_path / "business_overview.md").exists()
        assert (docs_path / "launch_checklist.md").exists()
    
    @pytest.mark.asyncio
    async def test_business_status(self, biz_engine, temp_workspace):
        """Test business status tracking."""
        business = await biz_engine.create_business(
            business_type=BusinessType.SUBSCRIPTION,
            name="Test Subscription",
            description="A subscription service"
        )
        
        await biz_engine.create_business_documents(business.business_id)
        await biz_engine.create_landing_page(business.business_id)
        
        status = biz_engine.get_business_status(business.business_id)
        
        assert "business" in status
        assert status["assets_created"] >= 3
        assert status["assets_verified"] >= 3
        assert status["progress_percent"] > 0
    
    @pytest.mark.asyncio
    async def test_execute_next_task(self, biz_engine, temp_workspace):
        """Test task execution."""
        business = await biz_engine.create_business(
            business_type=BusinessType.AGENCY,
            name="Test Agency",
            description="A marketing agency"
        )
        
        # Execute a task
        result = await biz_engine.execute_next_task(business.business_id)
        
        assert result["success"]
        
        # Check task was moved from pending to completed
        updated_business = biz_engine.active_businesses[business.business_id]
        assert len(updated_business.tasks_completed) > 0
    
    @pytest.mark.asyncio
    async def test_marketing_materials(self, biz_engine, temp_workspace):
        """Test marketing materials creation."""
        business = await biz_engine.create_business(
            business_type=BusinessType.CONTENT,
            name="Test Content Biz",
            description="A content business"
        )
        
        assets = await biz_engine._create_marketing_materials(business.business_id)
        
        assert len(assets) >= 2
        
        # Verify files exist
        marketing_path = Path(temp_workspace) / "businesses" / business.business_id / "marketing"
        assert (marketing_path / "welcome_email.txt").exists()
        assert (marketing_path / "social_media_plan.md").exists()


class TestAntiHallucination:
    """Tests to ensure the system doesn't fabricate results."""
    
    @pytest.fixture
    def temp_workspace(self):
        temp_dir = tempfile.mkdtemp(prefix="king_ai_halluc_test_")
        yield temp_dir
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def engine(self, temp_workspace):
        return ExecutionEngine(temp_workspace)
    
    @pytest.mark.asyncio
    async def test_verification_fails_for_missing_file(self, engine, temp_workspace):
        """Test that verification catches non-existent files."""
        # Try to verify a file that doesn't exist
        result = await engine.execute(ActionRequest(
            action_type=ActionType.FILE_READ,
            params={"path": "nonexistent_file.txt"}
        ))
        
        assert not result.success
        assert "not found" in result.error.lower() or "FileNotFoundError" in result.error
    
    @pytest.mark.asyncio
    async def test_verification_required_by_default(self, engine, temp_workspace):
        """Test that verification is required by default."""
        request = ActionRequest(
            action_type=ActionType.FILE_CREATE,
            params={"path": "verified.txt", "content": "test"}
        )
        
        result = await engine.execute(request)
        
        # Should be VERIFIED not just SUCCESS
        assert result.status == ExecutionStatus.VERIFIED
        assert result.verification is not None
        assert result.verification.get("verified", False)
    
    @pytest.mark.asyncio
    async def test_all_actions_logged(self, engine, temp_workspace):
        """Test that all actions are logged for audit."""
        # Perform several actions
        await engine.execute(ActionRequest(
            action_type=ActionType.DIR_CREATE,
            params={"path": "audit_test"}
        ))
        await engine.execute(ActionRequest(
            action_type=ActionType.FILE_CREATE,
            params={"path": "audit_test/file.txt", "content": "audit"}
        ))
        
        log = engine.get_execution_log()
        
        # All actions should be in the log
        assert len(log) == 2
        
        # Each log entry should have required fields
        for entry in log:
            assert "action_id" in entry
            assert "action_type" in entry
            assert "status" in entry
            assert "started_at" in entry
            assert "completed_at" in entry


class TestConvenienceFunctions:
    """Test the convenience wrapper functions."""
    
    @pytest.fixture
    def temp_workspace(self):
        temp_dir = tempfile.mkdtemp(prefix="king_ai_conv_test_")
        # Reset the global engine
        import src.services.execution_engine as ee
        ee._execution_engine = ExecutionEngine(temp_dir)
        yield temp_dir
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        ee._execution_engine = None
    
    @pytest.mark.asyncio
    async def test_create_file_convenience(self, temp_workspace):
        """Test the create_file convenience function."""
        result = await create_file("convenience.txt", "Hello convenience!")
        
        assert result.success
        assert (Path(temp_workspace) / "convenience.txt").exists()
    
    @pytest.mark.asyncio
    async def test_run_command_convenience(self, temp_workspace):
        """Test the run_command convenience function."""
        result = await run_command("echo test", cwd=temp_workspace)
        
        assert result.success
        assert "test" in result.output.get("stdout", "")


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
