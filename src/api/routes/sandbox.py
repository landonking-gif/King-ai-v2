"""
Sandbox API Routes - REST endpoints for sandbox operations.
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

from src.utils.sandbox import SandboxManager, SandboxConfig, SandboxResult
from src.utils.logging import get_logger

logger = get_logger("sandbox_api")
router = APIRouter(prefix="/sandbox", tags=["sandbox"])

# Global sandbox manager instance
_sandbox_manager: Optional[SandboxManager] = None


def get_sandbox_manager() -> SandboxManager:
    """Get or create sandbox manager."""
    global _sandbox_manager
    if _sandbox_manager is None:
        _sandbox_manager = SandboxManager()
    return _sandbox_manager


class RunTestsRequest(BaseModel):
    """Request to run tests in sandbox."""
    source_files: Dict[str, str]
    test_files: Dict[str, str]
    memory_limit: str = "512m"
    timeout_seconds: int = 300
    network_enabled: bool = False


class SandboxInfo(BaseModel):
    """Information about a sandbox."""
    id: str
    status: str
    created_at: str


class TestResultResponse(BaseModel):
    """Response with test results."""
    sandbox_id: str
    status: str
    exit_code: int
    duration_seconds: float
    tests_passed: int
    tests_failed: int
    stdout: str
    stderr: str


@router.post("/run", response_model=TestResultResponse)
async def run_tests(request: RunTestsRequest):
    """Run tests in an isolated sandbox."""
    manager = get_sandbox_manager()
    
    config = SandboxConfig(
        memory_limit=request.memory_limit,
        timeout_seconds=request.timeout_seconds,
        network_enabled=request.network_enabled
    )
    
    try:
        result = await manager.run_tests(
            request.source_files,
            request.test_files,
            config
        )
        
        summary = result.test_summary
        
        return TestResultResponse(
            sandbox_id=result.sandbox_id,
            status=result.status.value,
            exit_code=result.exit_code,
            duration_seconds=result.duration_seconds,
            tests_passed=summary["passed"],
            tests_failed=summary["failed"],
            stdout=result.stdout[:10000],  # Limit output size
            stderr=result.stderr[:5000]
        )
        
    except Exception as e:
        logger.error(f"Sandbox run failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list", response_model=List[SandboxInfo])
async def list_sandboxes():
    """List all active sandboxes."""
    manager = get_sandbox_manager()
    
    return [
        SandboxInfo(**info)
        for info in manager.list_sandboxes()
    ]


@router.delete("/{sandbox_id}")
async def cleanup_sandbox(sandbox_id: str):
    """Clean up a specific sandbox."""
    manager = get_sandbox_manager()
    
    sandbox = manager.get_sandbox(sandbox_id)
    if not sandbox:
        raise HTTPException(status_code=404, detail="Sandbox not found")
    
    await sandbox.cleanup()
    return {"status": "cleaned", "sandbox_id": sandbox_id}


@router.post("/cleanup-all")
async def cleanup_all_sandboxes():
    """Clean up all sandboxes."""
    manager = get_sandbox_manager()
    await manager.cleanup_all()
    return {"status": "all sandboxes cleaned"}
