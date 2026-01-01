"""
Sandbox Environment Manager - Isolated execution for evolution testing.
Uses Docker containers for secure, reproducible test environments.
"""

import os
import asyncio
import tempfile
import shutil
import uuid
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json

from src.utils.logging import get_logger
from config.settings import settings

logger = get_logger("sandbox")


class SandboxStatus(str, Enum):
    """Status of a sandbox environment."""
    CREATING = "creating"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CLEANED = "cleaned"


@dataclass
class SandboxConfig:
    """Configuration for sandbox environment."""
    memory_limit: str = "512m"
    cpu_limit: float = 1.0
    timeout_seconds: int = 300
    network_enabled: bool = False
    mount_source: bool = True
    python_version: str = "3.12"
    install_dependencies: bool = True
    
    # Resource limits
    max_processes: int = 50
    max_open_files: int = 1024
    disk_quota_mb: int = 500


@dataclass
class TestResult:
    """Result of a test execution."""
    test_name: str
    passed: bool
    duration_ms: float
    output: str = ""
    error: Optional[str] = None
    traceback: Optional[str] = None


@dataclass
class SandboxResult:
    """Complete result from sandbox execution."""
    sandbox_id: str
    status: SandboxStatus
    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float
    test_results: List[TestResult] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def all_tests_passed(self) -> bool:
        """Check if all tests passed."""
        return all(t.passed for t in self.test_results)
    
    @property
    def test_summary(self) -> Dict[str, int]:
        """Get summary of test results."""
        passed = sum(1 for t in self.test_results if t.passed)
        failed = len(self.test_results) - passed
        return {"passed": passed, "failed": failed, "total": len(self.test_results)}


class SandboxEnvironment:
    """
    Manages a single sandbox environment for isolated code execution.
    """
    
    def __init__(
        self,
        sandbox_id: str,
        config: SandboxConfig,
        work_dir: Path
    ):
        """
        Initialize sandbox environment.
        
        Args:
            sandbox_id: Unique identifier for this sandbox
            config: Sandbox configuration
            work_dir: Working directory for sandbox files
        """
        self.sandbox_id = sandbox_id
        self.config = config
        self.work_dir = work_dir
        self.status = SandboxStatus.CREATING
        self.container_id: Optional[str] = None
        self.created_at = datetime.now()
        self._process: Optional[asyncio.subprocess.Process] = None
        self._use_docker: bool = True
    
    async def setup(self, source_files: Dict[str, str]) -> bool:
        """
        Set up the sandbox environment.
        
        Args:
            source_files: Dict of file_path -> content to copy into sandbox
            
        Returns:
            Success status
        """
        try:
            # Create sandbox directory structure
            self.work_dir.mkdir(parents=True, exist_ok=True)
            (self.work_dir / "src").mkdir(exist_ok=True)
            (self.work_dir / "tests").mkdir(exist_ok=True)
            (self.work_dir / "output").mkdir(exist_ok=True)
            
            # Write source files
            for file_path, content in source_files.items():
                target = self.work_dir / file_path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content)

            # Ensure importable package layout for local execution
            init_file = self.work_dir / "src" / "__init__.py"
            if (self.work_dir / "src").exists() and not init_file.exists():
                init_file.write_text("")
            
            # Create requirements file if needed
            if self.config.install_dependencies:
                await self._create_requirements()
            
            # Try Docker if enabled, otherwise use local execution
            if settings.docker_sandbox_enabled:
                # Create Dockerfile
                await self._create_dockerfile()
                
                # Build container (if Docker unavailable, fall back to local execution)
                success = await self._build_container()

                if success:
                    self._use_docker = True
                    self.status = SandboxStatus.READY
                    logger.info(f"Sandbox ready with Docker: {self.sandbox_id}")
                    return True

                # Docker build failed: run locally inside the sandbox directory.
                self._use_docker = False
                self.status = SandboxStatus.READY
                logger.warning(f"Docker unavailable; using local sandbox execution: {self.sandbox_id}")
                return True
            else:
                # Docker disabled: use local execution directly
                self._use_docker = False
                self.status = SandboxStatus.READY
                logger.info(f"Sandbox ready with local execution (Docker disabled): {self.sandbox_id}")
                return True
            
        except Exception as e:
            logger.error(f"Sandbox setup failed: {e}")
            self.status = SandboxStatus.FAILED
            return False
    
    async def _create_dockerfile(self):
        """Create Dockerfile for the sandbox."""
        dockerfile = f"""
FROM python:{self.config.python_version}-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set resource limits
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default command
CMD ["python", "-m", "pytest", "tests/", "-v", "--tb=short", "-q"]
"""
        (self.work_dir / "Dockerfile").write_text(dockerfile)
    
    async def _create_requirements(self):
        """Create requirements.txt for sandbox."""
        requirements = """
pytest>=7.0.0
pytest-asyncio>=0.21.0
pytest-timeout>=2.1.0
pytest-cov>=4.0.0
"""
        (self.work_dir / "requirements.txt").write_text(requirements)
    
    async def _build_container(self) -> bool:
        """Build the Docker container."""
        try:
            process = await asyncio.create_subprocess_exec(
                "docker", "build", "-t", f"sandbox-{self.sandbox_id}", ".",
                cwd=self.work_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=120  # 2 minute build timeout
            )
            
            if process.returncode != 0:
                logger.error(f"Docker build failed: {stderr.decode()}")
                return False
            
            return True
            
        except asyncio.TimeoutError:
            logger.error("Docker build timed out")
            return False
        except Exception as e:
            logger.error(f"Docker build error: {e}")
            return False
    
    async def run(
        self,
        command: str = None,
        env_vars: Dict[str, str] = None
    ) -> SandboxResult:
        """
        Run tests in the sandbox.
        
        Args:
            command: Optional custom command to run
            env_vars: Optional environment variables
            
        Returns:
            Sandbox execution result
        """
        self.status = SandboxStatus.RUNNING
        start_time = datetime.now()
        
        if not self._use_docker:
            return await self._run_locally(command=command, env_vars=env_vars)

        # Build docker run command
        docker_cmd = [
            "docker", "run", "--rm",
            "--name", f"sandbox-run-{self.sandbox_id}",
            "--memory", self.config.memory_limit,
            "--cpus", str(self.config.cpu_limit),
            "--pids-limit", str(self.config.max_processes),
            "--ulimit", f"nofile={self.config.max_open_files}",
        ]
        
        # Network isolation
        if not self.config.network_enabled:
            docker_cmd.extend(["--network", "none"])
        
        # Environment variables
        if env_vars:
            for key, value in env_vars.items():
                docker_cmd.extend(["-e", f"{key}={value}"])
        
        # Add pytest JSON output for parsing
        docker_cmd.extend([
            "-e", "PYTEST_ADDOPTS=--json-report --json-report-file=/app/output/report.json"
        ])
        
        # Mount output directory
        docker_cmd.extend([
            "-v", f"{self.work_dir / 'output'}:/app/output"
        ])
        
        # Image name
        docker_cmd.append(f"sandbox-{self.sandbox_id}")
        
        # Custom command
        if command:
            docker_cmd.extend(["sh", "-c", command])
        
        try:
            self._process = await asyncio.create_subprocess_exec(
                *docker_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                self._process.communicate(),
                timeout=self.config.timeout_seconds
            )
            
            duration = (datetime.now() - start_time).total_seconds()
            
            # Parse test results
            test_results = await self._parse_test_results()
            
            self.status = SandboxStatus.COMPLETED
            
            return SandboxResult(
                sandbox_id=self.sandbox_id,
                status=self.status,
                exit_code=self._process.returncode,
                stdout=stdout.decode(),
                stderr=stderr.decode(),
                duration_seconds=duration,
                test_results=test_results
            )
            
        except asyncio.TimeoutError:
            self.status = SandboxStatus.TIMEOUT
            await self._kill_container()
            
            return SandboxResult(
                sandbox_id=self.sandbox_id,
                status=self.status,
                exit_code=-1,
                stdout="",
                stderr="Execution timed out",
                duration_seconds=self.config.timeout_seconds
            )
            
        except Exception as e:
            self.status = SandboxStatus.FAILED
            
            return SandboxResult(
                sandbox_id=self.sandbox_id,
                status=self.status,
                exit_code=-1,
                stdout="",
                stderr=str(e),
                duration_seconds=(datetime.now() - start_time).total_seconds()
            )

    async def _run_locally(
        self,
        command: str = None,
        env_vars: Dict[str, str] = None,
    ) -> SandboxResult:
        """Execute tests locally in a subprocess when Docker isn't available."""
        self.status = SandboxStatus.RUNNING
        start_time = datetime.now()

        env = os.environ.copy()
        if env_vars:
            env.update(env_vars)

        # Prefer pytest json report when available, but don't require it.
        env.setdefault("PYTEST_ADDOPTS", "--tb=short -q")

        if command:
            cmd = ["sh", "-c", command]
        else:
            cmd = [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short", "-q"]

        try:
            self._process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=self.work_dir,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                self._process.communicate(),
                timeout=self.config.timeout_seconds,
            )

            duration = (datetime.now() - start_time).total_seconds()
            exit_code = self._process.returncode

            if exit_code == 0:
                self.status = SandboxStatus.COMPLETED
            else:
                self.status = SandboxStatus.FAILED

            return SandboxResult(
                sandbox_id=self.sandbox_id,
                status=self.status,
                exit_code=exit_code,
                stdout=stdout.decode(errors="replace"),
                stderr=stderr.decode(errors="replace"),
                duration_seconds=duration,
                test_results=[],
            )

        except asyncio.TimeoutError:
            self.status = SandboxStatus.TIMEOUT
            if self._process and self._process.returncode is None:
                self._process.kill()
                try:
                    await self._process.wait()
                except Exception:
                    pass

            return SandboxResult(
                sandbox_id=self.sandbox_id,
                status=self.status,
                exit_code=-1,
                stdout="",
                stderr="Execution timed out",
                duration_seconds=self.config.timeout_seconds,
                test_results=[],
            )
        except Exception as e:
            self.status = SandboxStatus.FAILED
            return SandboxResult(
                sandbox_id=self.sandbox_id,
                status=self.status,
                exit_code=-1,
                stdout="",
                stderr=str(e),
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                test_results=[],
            )
    
    async def _parse_test_results(self) -> List[TestResult]:
        """Parse pytest JSON report if available."""
        report_file = self.work_dir / "output" / "report.json"
        
        if not report_file.exists():
            return []
        
        try:
            report = json.loads(report_file.read_text())
            results = []
            
            for test in report.get("tests", []):
                results.append(TestResult(
                    test_name=test.get("nodeid", "unknown"),
                    passed=test.get("outcome") == "passed",
                    duration_ms=test.get("duration", 0) * 1000,
                    output=test.get("call", {}).get("stdout", ""),
                    error=test.get("call", {}).get("longrepr") if test.get("outcome") != "passed" else None
                ))
            
            return results
            
        except Exception as e:
            logger.warning(f"Failed to parse test report: {e}")
            return []
    
    async def _kill_container(self):
        """Kill the running container."""
        try:
            process = await asyncio.create_subprocess_exec(
                "docker", "kill", f"sandbox-run-{self.sandbox_id}",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL
            )
            await process.wait()
        except Exception:
            pass
    
    async def cleanup(self):
        """Clean up sandbox resources."""
        try:
            if self._use_docker:
                # Remove container image
                process = await asyncio.create_subprocess_exec(
                    "docker", "rmi", "-f", f"sandbox-{self.sandbox_id}",
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL
                )
                await process.wait()
            
            # Remove work directory
            if self.work_dir.exists():
                shutil.rmtree(self.work_dir)
            
            self.status = SandboxStatus.CLEANED
            logger.info(f"Sandbox cleaned: {self.sandbox_id}")
            
        except Exception as e:
            logger.warning(f"Sandbox cleanup error: {e}")


class SandboxManager:
    """
    Manages multiple sandbox environments for parallel testing.
    """
    
    def __init__(self, base_dir: str = None):
        """
        Initialize sandbox manager.
        
        Args:
            base_dir: Base directory for sandbox files
        """
        self.base_dir = Path(base_dir) if base_dir else Path(tempfile.gettempdir()) / "king-ai-sandboxes"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self._sandboxes: Dict[str, SandboxEnvironment] = {}
        self._max_concurrent = 5
        self._semaphore = asyncio.Semaphore(self._max_concurrent)
    
    async def create_sandbox(
        self,
        source_files: Dict[str, str],
        config: SandboxConfig = None
    ) -> SandboxEnvironment:
        """
        Create a new sandbox environment.
        
        Args:
            source_files: Files to include in sandbox
            config: Optional sandbox configuration
            
        Returns:
            Created sandbox environment
        """
        sandbox_id = str(uuid.uuid4())[:8]
        config = config or SandboxConfig()
        work_dir = self.base_dir / sandbox_id
        
        sandbox = SandboxEnvironment(sandbox_id, config, work_dir)
        self._sandboxes[sandbox_id] = sandbox
        
        await sandbox.setup(source_files)
        
        return sandbox
    
    async def run_tests(
        self,
        source_files: Dict[str, str],
        test_files: Dict[str, str],
        config: SandboxConfig = None
    ) -> SandboxResult:
        """
        Run tests in an isolated sandbox.
        
        Args:
            source_files: Source code files
            test_files: Test files
            config: Optional configuration
            
        Returns:
            Test execution result
        """
        async with self._semaphore:
            # Combine source and test files
            all_files = {**source_files, **test_files}
            
            # Create and run sandbox
            sandbox = await self.create_sandbox(all_files, config)
            
            try:
                result = await sandbox.run()
                return result
            finally:
                await sandbox.cleanup()
                del self._sandboxes[sandbox.sandbox_id]
    
    async def run_parallel_tests(
        self,
        test_suites: List[Tuple[Dict[str, str], Dict[str, str]]],
        config: SandboxConfig = None
    ) -> List[SandboxResult]:
        """
        Run multiple test suites in parallel.
        
        Args:
            test_suites: List of (source_files, test_files) tuples
            config: Optional configuration
            
        Returns:
            List of results for each suite
        """
        tasks = [
            self.run_tests(source, tests, config)
            for source, tests in test_suites
        ]
        
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    async def cleanup_all(self):
        """Clean up all sandboxes."""
        for sandbox in list(self._sandboxes.values()):
            await sandbox.cleanup()
        
        self._sandboxes.clear()
    
    def get_sandbox(self, sandbox_id: str) -> Optional[SandboxEnvironment]:
        """Get a sandbox by ID."""
        return self._sandboxes.get(sandbox_id)
    
    def list_sandboxes(self) -> List[Dict[str, Any]]:
        """List all active sandboxes."""
        return [
            {
                "id": s.sandbox_id,
                "status": s.status.value,
                "created_at": s.created_at.isoformat()
            }
            for s in self._sandboxes.values()
        ]


# Backward compatibility with old Sandbox class
class Sandbox:
    """
    Legacy Sandbox class for backward compatibility.
    Wraps the new SandboxManager functionality.
    """
    
    def __init__(self, project_root: str = None):
        if project_root:
            self.project_root = Path(project_root)
        else:
            self.project_root = Path(__file__).parent.parent.parent.absolute()
        
        self.temp_dir = None
        self.manager = SandboxManager()
        self._sandbox_env = None
    
    def create_sandbox(self, target_files: list[str]):
        """Creates a temp directory - legacy method."""
        self.temp_dir = Path(tempfile.mkdtemp(prefix="king_ai_sandbox_"))
        
        src_path = self.project_root / "src"
        if src_path.exists():
            shutil.copytree(src_path, self.temp_dir / "src")
        
        tests_path = self.project_root / "tests"
        if tests_path.exists():
            shutil.copytree(tests_path, self.temp_dir / "tests")
        
        pyproject = self.project_root / "pyproject.toml"
        if pyproject.exists():
            shutil.copy(pyproject, self.temp_dir)
        
        return self.temp_dir
    
    def apply_patch(self, relative_file_path: str, new_code: str) -> bool:
        """Applies code change - legacy method."""
        if not self.temp_dir:
            raise RuntimeError("Sandbox not created. Call create_sandbox() first.")
        
        from src.utils.code_patcher import CodePatcher
        target = self.temp_dir / relative_file_path
        
        try:
            patcher = CodePatcher()
            result = patcher.apply_patch(str(target.absolute()), new_code)
            return result["success"]
        except Exception as e:
            print(f"Sandbox Patch Failed: {e}")
            return False
    
    def run_tests(self) -> dict:
        """Runs tests - legacy method."""
        if not self.temp_dir:
            raise RuntimeError("Sandbox not created.")
        
        try:
            import subprocess
            env = os.environ.copy()
            env["PYTHONPATH"] = str(self.temp_dir)
            
            result = subprocess.run(
                ["pytest", "tests"],
                cwd=str(self.temp_dir),
                env=env,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            return {
                "success": result.returncode == 0,
                "output": result.stdout + "\n" + result.stderr,
                "exit_code": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "output": "Tests timed out!",
                "exit_code": -1
            }
        except Exception as e:
            return {
                "success": False,
                "output": str(e),
                "exit_code": -1
            }
    
    def cleanup(self):
        """Cleanup - legacy method."""
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
