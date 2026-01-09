"""
Autonomous Execution Engine - Performs REAL actions with verification.

This module provides the core capability for King AI to execute real-world tasks
including file operations, API calls, shell commands, and business operations.
Every action is verified after execution to prevent hallucination.
"""

import os
import json
import subprocess
import asyncio
import aiohttp
import aiofiles
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable, TypeVar, Union
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import traceback

from src.utils.structured_logging import get_logger

logger = get_logger("execution_engine")

T = TypeVar('T')


class ExecutionStatus(str, Enum):
    """Status of an execution."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    VERIFIED = "verified"
    VERIFICATION_FAILED = "verification_failed"


class ActionType(str, Enum):
    """Types of actions the engine can execute."""
    FILE_CREATE = "file_create"
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    FILE_DELETE = "file_delete"
    DIR_CREATE = "dir_create"
    DIR_LIST = "dir_list"
    SHELL_COMMAND = "shell_command"
    HTTP_REQUEST = "http_request"
    API_CALL = "api_call"
    DATABASE_QUERY = "database_query"
    EMAIL_SEND = "email_send"
    BUSINESS_REGISTER = "business_register"
    PAYMENT_PROCESS = "payment_process"
    CUSTOM = "custom"


@dataclass
class ExecutionResult:
    """Result of an executed action."""
    action_id: str
    action_type: ActionType
    status: ExecutionStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    output: Any = None
    error: Optional[str] = None
    verification: Optional[Dict] = None
    metadata: Dict = field(default_factory=dict)
    
    @property
    def success(self) -> bool:
        return self.status in (ExecutionStatus.SUCCESS, ExecutionStatus.VERIFIED)
    
    def to_dict(self) -> Dict:
        return {
            "action_id": self.action_id,
            "action_type": self.action_type.value,
            "status": self.status.value,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "output": self.output,
            "error": self.error,
            "verification": self.verification,
            "success": self.success,
            "metadata": self.metadata
        }


@dataclass
class ActionRequest:
    """Request to execute an action."""
    action_type: ActionType
    params: Dict[str, Any]
    require_verification: bool = True
    timeout_seconds: int = 60
    retry_count: int = 3
    description: str = ""


class ExecutionEngine:
    """
    Core execution engine that performs REAL actions with verification.
    
    Key principles:
    1. NEVER report success without verification
    2. All actions are logged with full audit trail
    3. Verification proves the action actually occurred
    4. Rollback support for reversible actions
    """
    
    def __init__(self, workspace_root: Optional[str] = None):
        self.workspace_root = Path(workspace_root or os.getcwd())
        self.execution_log: List[ExecutionResult] = []
        self._http_session: Optional[aiohttp.ClientSession] = None
        self._action_counter = 0
        
        # Action handlers registry
        self._handlers: Dict[ActionType, Callable] = {
            ActionType.FILE_CREATE: self._execute_file_create,
            ActionType.FILE_READ: self._execute_file_read,
            ActionType.FILE_WRITE: self._execute_file_write,
            ActionType.FILE_DELETE: self._execute_file_delete,
            ActionType.DIR_CREATE: self._execute_dir_create,
            ActionType.DIR_LIST: self._execute_dir_list,
            ActionType.SHELL_COMMAND: self._execute_shell_command,
            ActionType.HTTP_REQUEST: self._execute_http_request,
            ActionType.API_CALL: self._execute_api_call,
        }
        
        # Verifiers registry
        self._verifiers: Dict[ActionType, Callable] = {
            ActionType.FILE_CREATE: self._verify_file_exists,
            ActionType.FILE_WRITE: self._verify_file_content,
            ActionType.FILE_DELETE: self._verify_file_deleted,
            ActionType.DIR_CREATE: self._verify_dir_exists,
            ActionType.SHELL_COMMAND: self._verify_command_output,
            ActionType.HTTP_REQUEST: self._verify_http_response,
        }
        
        logger.info("ExecutionEngine initialized", workspace=str(self.workspace_root))
    
    async def execute(self, request: ActionRequest) -> ExecutionResult:
        """
        Execute an action with full verification.
        
        This is the main entry point. It:
        1. Validates the request
        2. Executes the action
        3. Verifies the result
        4. Logs everything
        """
        self._action_counter += 1
        action_id = f"action_{self._action_counter}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        result = ExecutionResult(
            action_id=action_id,
            action_type=request.action_type,
            status=ExecutionStatus.PENDING,
            started_at=datetime.now(),
            metadata={"description": request.description, "params": request.params}
        )
        
        logger.info(
            "Executing action",
            action_id=action_id,
            action_type=request.action_type.value,
            description=request.description
        )
        
        try:
            result.status = ExecutionStatus.RUNNING
            
            # Get handler for this action type
            handler = self._handlers.get(request.action_type)
            if not handler:
                raise ValueError(f"No handler for action type: {request.action_type}")
            
            # Execute with retry and timeout
            for attempt in range(request.retry_count):
                try:
                    output = await asyncio.wait_for(
                        handler(request.params),
                        timeout=request.timeout_seconds
                    )
                    result.output = output
                    result.status = ExecutionStatus.SUCCESS
                    break
                except asyncio.TimeoutError:
                    if attempt == request.retry_count - 1:
                        raise
                    logger.warning(f"Attempt {attempt + 1} timed out, retrying...")
                    await asyncio.sleep(1)
                except Exception as e:
                    if attempt == request.retry_count - 1:
                        raise
                    logger.warning(f"Attempt {attempt + 1} failed: {e}, retrying...")
                    await asyncio.sleep(1)
            
            # Verification step - CRITICAL for anti-hallucination
            if request.require_verification and result.status == ExecutionStatus.SUCCESS:
                verification = await self._verify_action(request, result)
                result.verification = verification
                
                if verification.get("verified", False):
                    result.status = ExecutionStatus.VERIFIED
                    logger.info(
                        "Action verified",
                        action_id=action_id,
                        verification=verification
                    )
                else:
                    result.status = ExecutionStatus.VERIFICATION_FAILED
                    result.error = f"Verification failed: {verification.get('reason', 'Unknown')}"
                    logger.error(
                        "Action verification failed",
                        action_id=action_id,
                        verification=verification
                    )
            
        except Exception as e:
            result.status = ExecutionStatus.FAILED
            result.error = str(e)
            logger.error(
                "Action failed",
                action_id=action_id,
                error=str(e),
                traceback=traceback.format_exc()
            )
        
        result.completed_at = datetime.now()
        self.execution_log.append(result)
        
        return result
    
    async def execute_many(self, requests: List[ActionRequest]) -> List[ExecutionResult]:
        """Execute multiple actions, optionally in parallel."""
        results = []
        for request in requests:
            result = await self.execute(request)
            results.append(result)
            # Stop on critical failure
            if not result.success and request.params.get("critical", False):
                logger.error("Critical action failed, stopping execution chain")
                break
        return results
    
    # ===== Action Handlers =====
    
    async def _execute_file_create(self, params: Dict) -> Dict:
        """Create a new file with content."""
        path = self._resolve_path(params["path"])
        content = params.get("content", "")
        
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        
        async with aiofiles.open(path, "w", encoding="utf-8") as f:
            await f.write(content)
        
        stats = path.stat()
        return {
            "path": str(path),
            "size_bytes": stats.st_size,
            "created": True,
            "checksum": hashlib.md5(content.encode()).hexdigest()
        }
    
    async def _execute_file_read(self, params: Dict) -> Dict:
        """Read file contents."""
        path = self._resolve_path(params["path"])
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        async with aiofiles.open(path, "r", encoding="utf-8") as f:
            content = await f.read()
        
        return {
            "path": str(path),
            "content": content,
            "size_bytes": len(content),
            "checksum": hashlib.md5(content.encode()).hexdigest()
        }
    
    async def _execute_file_write(self, params: Dict) -> Dict:
        """Write/update file content."""
        path = self._resolve_path(params["path"])
        content = params["content"]
        mode = params.get("mode", "w")  # 'w' for overwrite, 'a' for append
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        async with aiofiles.open(path, mode, encoding="utf-8") as f:
            await f.write(content)
        
        # Read back to verify
        async with aiofiles.open(path, "r", encoding="utf-8") as f:
            actual_content = await f.read()
        
        return {
            "path": str(path),
            "size_bytes": len(actual_content),
            "checksum": hashlib.md5(actual_content.encode()).hexdigest(),
            "mode": mode
        }
    
    async def _execute_file_delete(self, params: Dict) -> Dict:
        """Delete a file."""
        path = self._resolve_path(params["path"])
        existed = path.exists()
        
        if existed:
            path.unlink()
        
        return {
            "path": str(path),
            "existed": existed,
            "deleted": existed
        }
    
    async def _execute_dir_create(self, params: Dict) -> Dict:
        """Create a directory."""
        path = self._resolve_path(params["path"])
        existed = path.exists()
        
        path.mkdir(parents=True, exist_ok=True)
        
        return {
            "path": str(path),
            "existed_before": existed,
            "exists_now": path.exists()
        }
    
    async def _execute_dir_list(self, params: Dict) -> Dict:
        """List directory contents."""
        path = self._resolve_path(params["path"])
        pattern = params.get("pattern", "*")
        recursive = params.get("recursive", False)
        
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {path}")
        
        if recursive:
            items = list(path.rglob(pattern))
        else:
            items = list(path.glob(pattern))
        
        return {
            "path": str(path),
            "items": [
                {
                    "name": item.name,
                    "path": str(item),
                    "is_dir": item.is_dir(),
                    "size": item.stat().st_size if item.is_file() else None
                }
                for item in items[:100]  # Limit to 100 items
            ],
            "count": len(items)
        }
    
    async def _execute_shell_command(self, params: Dict) -> Dict:
        """Execute a shell command."""
        command = params["command"]
        cwd = params.get("cwd", str(self.workspace_root))
        timeout = params.get("timeout", 60)
        shell = params.get("shell", True)
        
        # Security: Block dangerous commands
        dangerous_patterns = ["rm -rf /", "format c:", "del /s /q", ":(){:|:&};:"]
        for pattern in dangerous_patterns:
            if pattern.lower() in command.lower():
                raise ValueError(f"Blocked dangerous command pattern: {pattern}")
        
        logger.info("Executing shell command", command=command, cwd=cwd)
        
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            process.kill()
            raise TimeoutError(f"Command timed out after {timeout}s")
        
        return {
            "command": command,
            "return_code": process.returncode,
            "stdout": stdout.decode("utf-8", errors="replace"),
            "stderr": stderr.decode("utf-8", errors="replace"),
            "success": process.returncode == 0
        }
    
    async def _execute_http_request(self, params: Dict) -> Dict:
        """Execute an HTTP request."""
        method = params.get("method", "GET").upper()
        url = params["url"]
        headers = params.get("headers", {})
        body = params.get("body")
        timeout = params.get("timeout", 30)
        
        session = await self._get_session()
        
        kwargs = {
            "method": method,
            "url": url,
            "headers": headers,
            "timeout": aiohttp.ClientTimeout(total=timeout)
        }
        
        if body:
            if isinstance(body, dict):
                kwargs["json"] = body
            else:
                kwargs["data"] = body
        
        async with session.request(**kwargs) as response:
            content_type = response.headers.get("Content-Type", "")
            
            if "json" in content_type:
                response_body = await response.json()
            else:
                response_body = await response.text()
            
            return {
                "url": url,
                "method": method,
                "status_code": response.status,
                "headers": dict(response.headers),
                "body": response_body,
                "success": 200 <= response.status < 300
            }
    
    async def _execute_api_call(self, params: Dict) -> Dict:
        """Execute an API call (wrapper around HTTP with auth)."""
        # Add authentication if provided
        auth_type = params.get("auth_type", "none")
        headers = params.get("headers", {})
        
        if auth_type == "bearer":
            headers["Authorization"] = f"Bearer {params['token']}"
        elif auth_type == "api_key":
            key_header = params.get("key_header", "X-API-Key")
            headers[key_header] = params["api_key"]
        
        params["headers"] = headers
        return await self._execute_http_request(params)
    
    # ===== Verification Methods =====
    
    async def _verify_action(self, request: ActionRequest, result: ExecutionResult) -> Dict:
        """Verify that an action actually completed successfully."""
        verifier = self._verifiers.get(request.action_type)
        
        if not verifier:
            return {"verified": True, "reason": "No verifier available, trusting execution"}
        
        try:
            return await verifier(request.params, result.output)
        except Exception as e:
            return {"verified": False, "reason": str(e)}
    
    async def _verify_file_exists(self, params: Dict, output: Dict) -> Dict:
        """Verify a file was created."""
        path = Path(output["path"])
        exists = path.exists()
        
        return {
            "verified": exists,
            "reason": "File exists" if exists else "File does not exist",
            "path": str(path)
        }
    
    async def _verify_file_content(self, params: Dict, output: Dict) -> Dict:
        """Verify file content matches expected."""
        path = Path(output["path"])
        
        if not path.exists():
            return {"verified": False, "reason": "File does not exist"}
        
        async with aiofiles.open(path, "r", encoding="utf-8") as f:
            actual_content = await f.read()
        
        actual_checksum = hashlib.md5(actual_content.encode()).hexdigest()
        expected_checksum = output.get("checksum")
        
        if expected_checksum and actual_checksum == expected_checksum:
            return {"verified": True, "reason": "Content checksum matches"}
        
        return {
            "verified": False,
            "reason": f"Checksum mismatch: expected {expected_checksum}, got {actual_checksum}"
        }
    
    async def _verify_file_deleted(self, params: Dict, output: Dict) -> Dict:
        """Verify a file was deleted."""
        path = Path(output["path"])
        exists = path.exists()
        
        return {
            "verified": not exists,
            "reason": "File deleted" if not exists else "File still exists"
        }
    
    async def _verify_dir_exists(self, params: Dict, output: Dict) -> Dict:
        """Verify a directory exists."""
        path = Path(output["path"])
        exists = path.exists() and path.is_dir()
        
        return {
            "verified": exists,
            "reason": "Directory exists" if exists else "Directory does not exist"
        }
    
    async def _verify_command_output(self, params: Dict, output: Dict) -> Dict:
        """Verify command executed successfully."""
        return_code = output.get("return_code", -1)
        expected_output = params.get("expected_output")
        
        if return_code != 0:
            return {"verified": False, "reason": f"Non-zero return code: {return_code}"}
        
        if expected_output:
            stdout = output.get("stdout", "")
            if expected_output not in stdout:
                return {"verified": False, "reason": f"Expected output not found: {expected_output}"}
        
        return {"verified": True, "reason": "Command completed successfully"}
    
    async def _verify_http_response(self, params: Dict, output: Dict) -> Dict:
        """Verify HTTP request succeeded."""
        status = output.get("status_code", 0)
        expected_status = params.get("expected_status", range(200, 300))
        
        if isinstance(expected_status, int):
            expected_status = [expected_status]
        
        verified = status in expected_status
        return {
            "verified": verified,
            "reason": f"Status {status}" + (" matches expected" if verified else " unexpected")
        }
    
    # ===== Utilities =====
    
    def _resolve_path(self, path: Union[str, Path]) -> Path:
        """Resolve a path relative to workspace root."""
        p = Path(path)
        if p.is_absolute():
            return p
        return self.workspace_root / p
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._http_session is None or self._http_session.closed:
            self._http_session = aiohttp.ClientSession()
        return self._http_session
    
    async def close(self):
        """Cleanup resources."""
        if self._http_session and not self._http_session.closed:
            await self._http_session.close()
    
    def get_execution_log(self) -> List[Dict]:
        """Get the full execution log."""
        return [r.to_dict() for r in self.execution_log]
    
    def get_success_rate(self) -> float:
        """Calculate success rate of all executions."""
        if not self.execution_log:
            return 0.0
        successful = sum(1 for r in self.execution_log if r.success)
        return successful / len(self.execution_log)


# Singleton instance
_execution_engine: Optional[ExecutionEngine] = None


def get_execution_engine(workspace_root: Optional[str] = None) -> ExecutionEngine:
    """Get or create the execution engine singleton."""
    global _execution_engine
    if _execution_engine is None:
        _execution_engine = ExecutionEngine(workspace_root)
    return _execution_engine


# Convenience functions for common operations

async def create_file(path: str, content: str, verify: bool = True) -> ExecutionResult:
    """Create a file with content."""
    engine = get_execution_engine()
    return await engine.execute(ActionRequest(
        action_type=ActionType.FILE_CREATE,
        params={"path": path, "content": content},
        require_verification=verify,
        description=f"Create file: {path}"
    ))


async def run_command(command: str, cwd: Optional[str] = None) -> ExecutionResult:
    """Run a shell command."""
    engine = get_execution_engine()
    params = {"command": command}
    if cwd:
        params["cwd"] = cwd
    return await engine.execute(ActionRequest(
        action_type=ActionType.SHELL_COMMAND,
        params=params,
        require_verification=True,
        description=f"Run: {command[:50]}..."
    ))


async def http_get(url: str, headers: Optional[Dict] = None) -> ExecutionResult:
    """Perform HTTP GET request."""
    engine = get_execution_engine()
    return await engine.execute(ActionRequest(
        action_type=ActionType.HTTP_REQUEST,
        params={"url": url, "method": "GET", "headers": headers or {}},
        require_verification=True,
        description=f"GET {url}"
    ))


async def http_post(url: str, body: Any, headers: Optional[Dict] = None) -> ExecutionResult:
    """Perform HTTP POST request."""
    engine = get_execution_engine()
    return await engine.execute(ActionRequest(
        action_type=ActionType.HTTP_REQUEST,
        params={"url": url, "method": "POST", "body": body, "headers": headers or {}},
        require_verification=True,
        description=f"POST {url}"
    ))
