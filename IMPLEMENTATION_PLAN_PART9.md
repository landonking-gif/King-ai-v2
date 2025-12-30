# King AI v2 - Implementation Plan Part 9
## Evolution Engine - Sandbox Testing

**Target Timeline:** Week 6-7
**Objective:** Implement isolated sandbox environments for safely testing code changes before deployment.

---

## Overview of All Parts

| Part | Title | Status |
|------|-------|--------|
| 1 | Infrastructure Layer & Core System Hardening | ✅ Complete |
| 2 | Master AI Brain - Core Enhancements | ✅ Complete |
| 3 | Master AI Brain - Context & Memory System | ✅ Complete |
| 4 | Master AI Brain - Planning & ReAct Implementation | ✅ Complete |
| 5 | Evolution Engine - Core Models & Proposal System | ✅ Complete |
| 6 | Evolution Engine - Code Analysis & AST Tools | ✅ Complete |
| 7 | Evolution Engine - Code Patching & Generation | ✅ Complete |
| 8 | Evolution Engine - Git Integration & Rollback | ✅ Complete |
| **9** | **Evolution Engine - Sandbox Testing** | 🔄 Current |
| 10 | Sub-Agent: Research (Web/API) | ⏳ Pending |
| 11 | Sub-Agent: Code Generator | ⏳ Pending |
| 12 | Sub-Agent: Content (Blog/SEO) | ⏳ Pending |
| 13 | Sub-Agent: Commerce - Shopify | ⏳ Pending |
| 14 | Sub-Agent: Commerce - Suppliers | ⏳ Pending |
| 15 | Sub-Agent: Finance - Stripe | ⏳ Pending |
| 16 | Sub-Agent: Finance - Plaid/Banking | ⏳ Pending |
| 17 | Sub-Agent: Analytics | ⏳ Pending |
| 18 | Sub-Agent: Legal | ⏳ Pending |
| 19 | Business: Lifecycle Engine | ⏳ Pending |
| 20 | Business: Playbook System | ⏳ Pending |
| 21 | Business: Portfolio Management | ⏳ Pending |
| 22 | Dashboard: React Components | ⏳ Pending |
| 23 | Dashboard: Approval Workflows | ⏳ Pending |
| 24 | Dashboard: WebSocket & Monitoring | ⏳ Pending |

---

## Part 9 Scope

This part focuses on:
1. Docker-based sandbox environment creation
2. Isolated test execution with resource limits
3. Test result collection and analysis
4. Sandbox lifecycle management
5. Security isolation and cleanup

---

## Task 9.1: Create Sandbox Environment Manager

**File:** `src/utils/sandbox.py` (REPLACE EXISTING FILE)

```python
"""
Sandbox Environment Manager - Isolated execution for evolution testing.
Uses Docker containers for secure, reproducible test environments.
"""

import os
import asyncio
import tempfile
import shutil
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json

from src.utils.structured_logging import get_logger
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
            
            # Create requirements file if needed
            if self.config.install_dependencies:
                await self._create_requirements()
            
            # Create Dockerfile
            await self._create_dockerfile()
            
            # Build container
            success = await self._build_container()
            
            if success:
                self.status = SandboxStatus.READY
                logger.info(f"Sandbox ready: {self.sandbox_id}")
            else:
                self.status = SandboxStatus.FAILED
                
            return success
            
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
```

---

## Task 9.2: Create Evolution Test Runner

**File:** `src/master_ai/evolution_test_runner.py` (CREATE NEW FILE)

```python
"""
Evolution Test Runner - Orchestrates testing of evolution proposals.
Validates code changes before they are applied to production.
"""

import asyncio
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

from src.utils.sandbox import (
    SandboxManager,
    SandboxConfig,
    SandboxResult,
    TestResult,
    SandboxStatus
)
from src.master_ai.evolution_models import EvolutionProposal
from src.utils.code_patcher import PatchSet
from src.utils.structured_logging import get_logger

logger = get_logger("evolution_test_runner")


class TestPhase(str, Enum):
    """Phases of evolution testing."""
    SYNTAX_CHECK = "syntax_check"
    UNIT_TESTS = "unit_tests"
    INTEGRATION_TESTS = "integration_tests"
    REGRESSION_TESTS = "regression_tests"


@dataclass
class TestPhaseResult:
    """Result of a test phase."""
    phase: TestPhase
    passed: bool
    duration_seconds: float
    tests_run: int = 0
    tests_passed: int = 0
    tests_failed: int = 0
    error_summary: Optional[str] = None
    details: List[TestResult] = field(default_factory=list)


@dataclass
class EvolutionTestResult:
    """Complete test result for an evolution proposal."""
    proposal_id: str
    passed: bool
    phases: List[TestPhaseResult]
    total_duration_seconds: float
    coverage_percent: Optional[float] = None
    recommendations: List[str] = field(default_factory=list)
    
    @property
    def summary(self) -> Dict[str, Any]:
        """Get test summary."""
        return {
            "passed": self.passed,
            "phases_passed": sum(1 for p in self.phases if p.passed),
            "phases_total": len(self.phases),
            "total_tests": sum(p.tests_run for p in self.phases),
            "total_passed": sum(p.tests_passed for p in self.phases),
            "total_failed": sum(p.tests_failed for p in self.phases),
            "duration": self.total_duration_seconds,
            "coverage": self.coverage_percent
        }


class EvolutionTestRunner:
    """
    Runs comprehensive tests on evolution proposals.
    Uses sandbox environments for isolation.
    """
    
    def __init__(
        self,
        sandbox_manager: SandboxManager,
        project_root: str
    ):
        """
        Initialize test runner.
        
        Args:
            sandbox_manager: Sandbox manager instance
            project_root: Root directory of the project
        """
        self.sandbox = sandbox_manager
        self.project_root = Path(project_root)
        
        # Test configuration
        self.timeout_per_phase = 120  # seconds
        self.require_all_phases = False
    
    async def test_evolution(
        self,
        proposal: EvolutionProposal,
        patchset: PatchSet,
        phases: List[TestPhase] = None
    ) -> EvolutionTestResult:
        """
        Run all test phases on an evolution proposal.
        
        Args:
            proposal: The evolution proposal
            patchset: Patchset with code changes
            phases: Optional list of phases to run
            
        Returns:
            Complete test result
        """
        if phases is None:
            phases = list(TestPhase)
        
        start_time = datetime.now()
        phase_results: List[TestPhaseResult] = []
        
        # Prepare source files with patches applied
        source_files = await self._prepare_source_files(patchset)
        test_files = await self._collect_test_files()
        
        for phase in phases:
            logger.info(f"Running test phase: {phase.value}", proposal_id=proposal.id)
            
            result = await self._run_phase(
                phase, source_files, test_files, proposal
            )
            phase_results.append(result)
            
            # Stop on failure if required
            if not result.passed and self.require_all_phases:
                logger.warning(
                    f"Phase {phase.value} failed, stopping tests",
                    proposal_id=proposal.id
                )
                break
        
        total_duration = (datetime.now() - start_time).total_seconds()
        
        # Determine overall pass/fail
        passed = all(p.passed for p in phase_results)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(phase_results)
        
        result = EvolutionTestResult(
            proposal_id=proposal.id,
            passed=passed,
            phases=phase_results,
            total_duration_seconds=total_duration,
            recommendations=recommendations
        )
        
        logger.info(
            f"Evolution testing complete",
            proposal_id=proposal.id,
            passed=passed,
            duration=total_duration
        )
        
        return result
    
    async def _prepare_source_files(
        self,
        patchset: PatchSet
    ) -> Dict[str, str]:
        """Prepare source files with patches applied."""
        source_files = {}
        
        # Read current source files
        for patch in patchset.patches:
            file_path = self.project_root / patch.file_path
            
            if file_path.exists():
                # Apply patch to get new content
                source_files[patch.file_path] = patch.new_content
            else:
                # New file
                source_files[patch.file_path] = patch.new_content
        
        # Include essential existing files
        essential_dirs = ["src", "config"]
        for dir_name in essential_dirs:
            dir_path = self.project_root / dir_name
            if dir_path.exists():
                for file in dir_path.rglob("*.py"):
                    rel_path = str(file.relative_to(self.project_root))
                    if rel_path not in source_files:
                        source_files[rel_path] = file.read_text()
        
        return source_files
    
    async def _collect_test_files(self) -> Dict[str, str]:
        """Collect test files from project."""
        test_files = {}
        
        tests_dir = self.project_root / "tests"
        if tests_dir.exists():
            for file in tests_dir.rglob("*.py"):
                rel_path = str(file.relative_to(self.project_root))
                test_files[rel_path] = file.read_text()
        
        return test_files
    
    async def _run_phase(
        self,
        phase: TestPhase,
        source_files: Dict[str, str],
        test_files: Dict[str, str],
        proposal: EvolutionProposal
    ) -> TestPhaseResult:
        """Run a single test phase."""
        start_time = datetime.now()
        
        if phase == TestPhase.SYNTAX_CHECK:
            return await self._run_syntax_check(source_files)
        elif phase == TestPhase.UNIT_TESTS:
            return await self._run_unit_tests(source_files, test_files)
        elif phase == TestPhase.INTEGRATION_TESTS:
            return await self._run_integration_tests(source_files, test_files)
        elif phase == TestPhase.REGRESSION_TESTS:
            return await self._run_regression_tests(source_files, test_files)
        
        return TestPhaseResult(
            phase=phase,
            passed=False,
            duration_seconds=0,
            error_summary="Unknown phase"
        )
    
    async def _run_syntax_check(
        self,
        source_files: Dict[str, str]
    ) -> TestPhaseResult:
        """Check Python syntax of all files."""
        start_time = datetime.now()
        errors = []
        
        import ast
        
        for file_path, content in source_files.items():
            if not file_path.endswith('.py'):
                continue
            
            try:
                ast.parse(content)
            except SyntaxError as e:
                errors.append(f"{file_path}:{e.lineno}: {e.msg}")
        
        duration = (datetime.now() - start_time).total_seconds()
        
        return TestPhaseResult(
            phase=TestPhase.SYNTAX_CHECK,
            passed=len(errors) == 0,
            duration_seconds=duration,
            tests_run=len([f for f in source_files if f.endswith('.py')]),
            tests_passed=len(source_files) - len(errors),
            tests_failed=len(errors),
            error_summary="\n".join(errors) if errors else None
        )
    
    async def _run_unit_tests(
        self,
        source_files: Dict[str, str],
        test_files: Dict[str, str]
    ) -> TestPhaseResult:
        """Run unit tests in sandbox."""
        start_time = datetime.now()
        
        # Filter to unit tests only
        unit_tests = {
            k: v for k, v in test_files.items()
            if 'test_' in k and 'integration' not in k.lower()
        }
        
        if not unit_tests:
            return TestPhaseResult(
                phase=TestPhase.UNIT_TESTS,
                passed=True,
                duration_seconds=0,
                tests_run=0,
                error_summary="No unit tests found"
            )
        
        config = SandboxConfig(
            timeout_seconds=self.timeout_per_phase,
            network_enabled=False
        )
        
        result = await self.sandbox.run_tests(source_files, unit_tests, config)
        duration = (datetime.now() - start_time).total_seconds()
        
        return self._convert_sandbox_result(
            TestPhase.UNIT_TESTS, result, duration
        )
    
    async def _run_integration_tests(
        self,
        source_files: Dict[str, str],
        test_files: Dict[str, str]
    ) -> TestPhaseResult:
        """Run integration tests in sandbox."""
        start_time = datetime.now()
        
        # Filter to integration tests
        integration_tests = {
            k: v for k, v in test_files.items()
            if 'integration' in k.lower() or 'e2e' in k.lower()
        }
        
        if not integration_tests:
            return TestPhaseResult(
                phase=TestPhase.INTEGRATION_TESTS,
                passed=True,
                duration_seconds=0,
                tests_run=0,
                error_summary="No integration tests found"
            )
        
        config = SandboxConfig(
            timeout_seconds=self.timeout_per_phase * 2,
            network_enabled=True,  # Integration tests may need network
            memory_limit="1g"
        )
        
        result = await self.sandbox.run_tests(source_files, integration_tests, config)
        duration = (datetime.now() - start_time).total_seconds()
        
        return self._convert_sandbox_result(
            TestPhase.INTEGRATION_TESTS, result, duration
        )
    
    async def _run_regression_tests(
        self,
        source_files: Dict[str, str],
        test_files: Dict[str, str]
    ) -> TestPhaseResult:
        """Run regression tests to ensure no existing functionality is broken."""
        start_time = datetime.now()
        
        # Run all tests for regression
        config = SandboxConfig(
            timeout_seconds=self.timeout_per_phase * 2,
            network_enabled=False
        )
        
        result = await self.sandbox.run_tests(source_files, test_files, config)
        duration = (datetime.now() - start_time).total_seconds()
        
        return self._convert_sandbox_result(
            TestPhase.REGRESSION_TESTS, result, duration
        )
    
    def _convert_sandbox_result(
        self,
        phase: TestPhase,
        sandbox_result: SandboxResult,
        duration: float
    ) -> TestPhaseResult:
        """Convert sandbox result to test phase result."""
        summary = sandbox_result.test_summary
        
        # Determine pass/fail
        if sandbox_result.status == SandboxStatus.TIMEOUT:
            passed = False
            error = "Test execution timed out"
        elif sandbox_result.status == SandboxStatus.FAILED:
            passed = False
            error = sandbox_result.stderr[:500]
        elif summary["failed"] > 0:
            passed = False
            error = f"{summary['failed']} tests failed"
        else:
            passed = True
            error = None
        
        return TestPhaseResult(
            phase=phase,
            passed=passed,
            duration_seconds=duration,
            tests_run=summary["total"],
            tests_passed=summary["passed"],
            tests_failed=summary["failed"],
            error_summary=error,
            details=sandbox_result.test_results
        )
    
    def _generate_recommendations(
        self,
        phase_results: List[TestPhaseResult]
    ) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        for result in phase_results:
            if not result.passed:
                if result.phase == TestPhase.SYNTAX_CHECK:
                    recommendations.append(
                        "Fix syntax errors before proceeding with evolution"
                    )
                elif result.phase == TestPhase.UNIT_TESTS:
                    recommendations.append(
                        f"Review failing unit tests: {result.tests_failed} failures"
                    )
                elif result.phase == TestPhase.REGRESSION_TESTS:
                    recommendations.append(
                        "Evolution may break existing functionality - review carefully"
                    )
        
        # Check for slow tests
        slow_phases = [r for r in phase_results if r.duration_seconds > 60]
        if slow_phases:
            recommendations.append(
                "Consider optimizing slow tests for faster feedback"
            )
        
        return recommendations
    
    async def quick_validate(
        self,
        patchset: PatchSet
    ) -> Tuple[bool, str]:
        """
        Quick validation of a patchset (syntax only).
        
        Returns:
            Tuple of (valid, error_message)
        """
        source_files = await self._prepare_source_files(patchset)
        result = await self._run_syntax_check(source_files)
        
        return result.passed, result.error_summary or ""
```

---

## Task 9.3: Create Test Generation Helper

**File:** `src/master_ai/test_generator.py` (CREATE NEW FILE)

```python
"""
Test Generator - Generates tests for evolution proposals.
Uses LLM to create test cases for new or modified code.
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum

from src.utils.ollama_client import OllamaClient
from src.utils.code_analyzer import CodeAnalyzer, FunctionInfo, ClassInfo
from src.utils.structured_logging import get_logger

logger = get_logger("test_generator")


class TestType(str, Enum):
    """Types of tests to generate."""
    UNIT = "unit"
    INTEGRATION = "integration"
    EDGE_CASE = "edge_case"
    ERROR_HANDLING = "error_handling"


@dataclass
class GeneratedTest:
    """A generated test case."""
    name: str
    test_type: TestType
    code: str
    description: str
    target_function: str
    confidence: float


class TestGenerator:
    """
    Generates test cases for code changes.
    Uses LLM for intelligent test generation.
    """
    
    TEST_GENERATION_PROMPT = """Generate pytest test cases for the following Python code.

Code to test:
```python
{code}
```

Context about the code:
{context}

Requirements:
1. Generate comprehensive test cases covering:
   - Normal operation (happy path)
   - Edge cases (empty inputs, boundaries)
   - Error conditions (invalid inputs, exceptions)
2. Use pytest fixtures where appropriate
3. Include docstrings explaining each test
4. Use meaningful test names (test_<function>_<scenario>)
5. Use assert statements with clear messages

Generate {num_tests} test cases.

Output the tests as valid Python code:
```python
"""
    
    def __init__(self, llm_client: OllamaClient, code_analyzer: CodeAnalyzer):
        """
        Initialize test generator.
        
        Args:
            llm_client: LLM client for generation
            code_analyzer: Code analyzer for understanding code
        """
        self.llm = llm_client
        self.analyzer = code_analyzer
    
    async def generate_tests(
        self,
        source_code: str,
        file_path: str,
        num_tests: int = 5,
        test_types: List[TestType] = None
    ) -> List[GeneratedTest]:
        """
        Generate tests for source code.
        
        Args:
            source_code: Python source code to test
            file_path: Path of the source file
            num_tests: Number of tests to generate
            test_types: Types of tests to generate
            
        Returns:
            List of generated tests
        """
        if test_types is None:
            test_types = list(TestType)
        
        # Analyze the code
        analysis = self.analyzer.analyze_code(source_code)
        
        # Build context
        context = self._build_context(analysis)
        
        # Generate tests via LLM
        prompt = self.TEST_GENERATION_PROMPT.format(
            code=source_code[:3000],  # Limit code size
            context=context,
            num_tests=num_tests
        )
        
        response = await self.llm.generate(prompt)
        
        # Parse generated tests
        tests = self._parse_generated_tests(response, analysis)
        
        logger.info(
            f"Generated {len(tests)} tests",
            file=file_path,
            functions=len(analysis.get("functions", []))
        )
        
        return tests
    
    def _build_context(self, analysis: Dict[str, Any]) -> str:
        """Build context string from code analysis."""
        parts = []
        
        functions = analysis.get("functions", [])
        if functions:
            parts.append(f"Functions: {', '.join(f.name for f in functions)}")
            for func in functions[:5]:
                params = ", ".join(f"{p.name}: {p.type_hint or 'Any'}" for p in func.parameters)
                parts.append(f"  - {func.name}({params}) -> {func.return_type or 'None'}")
        
        classes = analysis.get("classes", [])
        if classes:
            parts.append(f"Classes: {', '.join(c.name for c in classes)}")
        
        return "\n".join(parts)
    
    def _parse_generated_tests(
        self,
        llm_response: str,
        analysis: Dict[str, Any]
    ) -> List[GeneratedTest]:
        """Parse LLM response into test objects."""
        tests = []
        
        # Extract code block
        code_start = llm_response.find("```python")
        code_end = llm_response.rfind("```")
        
        if code_start != -1 and code_end > code_start:
            test_code = llm_response[code_start + 9:code_end].strip()
        else:
            test_code = llm_response
        
        # Parse individual test functions
        import ast
        try:
            tree = ast.parse(test_code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                    # Extract test info
                    test_name = node.name
                    docstring = ast.get_docstring(node) or ""
                    
                    # Determine test type from name
                    test_type = TestType.UNIT
                    if "error" in test_name.lower() or "exception" in test_name.lower():
                        test_type = TestType.ERROR_HANDLING
                    elif "edge" in test_name.lower() or "empty" in test_name.lower():
                        test_type = TestType.EDGE_CASE
                    
                    # Get the function code
                    func_code = ast.unparse(node)
                    
                    # Try to identify target function
                    target = self._identify_target_function(test_name, analysis)
                    
                    tests.append(GeneratedTest(
                        name=test_name,
                        test_type=test_type,
                        code=func_code,
                        description=docstring,
                        target_function=target,
                        confidence=0.8
                    ))
                    
        except SyntaxError as e:
            logger.warning(f"Failed to parse generated tests: {e}")
        
        return tests
    
    def _identify_target_function(
        self,
        test_name: str,
        analysis: Dict[str, Any]
    ) -> str:
        """Identify which function a test is targeting."""
        functions = analysis.get("functions", [])
        
        # Remove test_ prefix and common suffixes
        name_parts = test_name.replace("test_", "").split("_")
        
        for func in functions:
            if func.name.lower() in test_name.lower():
                return func.name
        
        return "unknown"
    
    async def generate_tests_for_function(
        self,
        function: FunctionInfo,
        source_context: str = ""
    ) -> List[GeneratedTest]:
        """Generate tests specifically for one function."""
        prompt = f"""Generate pytest test cases for this function:

```python
{function.to_dict()}
```

{f'Context: {source_context}' if source_context else ''}

Generate 3-5 comprehensive tests covering:
1. Normal operation with typical inputs
2. Edge cases (empty, None, boundary values)
3. Error handling (invalid inputs)

Output valid pytest code:
```python
"""
        
        response = await self.llm.generate(prompt)
        
        # Simplified parsing for single function
        tests = self._parse_generated_tests(
            response,
            {"functions": [function]}
        )
        
        return tests
    
    def generate_test_file_skeleton(
        self,
        source_file: str,
        functions: List[FunctionInfo]
    ) -> str:
        """Generate a test file skeleton."""
        module_name = source_file.replace("/", ".").replace(".py", "")
        
        imports = [
            "import pytest",
            f"from {module_name} import *",
            "",
            "",
        ]
        
        test_stubs = []
        for func in functions:
            test_stubs.append(f'''
def test_{func.name}_basic():
    """Test {func.name} with basic inputs."""
    # TODO: Implement test
    pass


def test_{func.name}_edge_cases():
    """Test {func.name} with edge cases."""
    # TODO: Implement test
    pass
''')
        
        return "\n".join(imports) + "\n".join(test_stubs)
```

---

## Task 9.4: Create Sandbox API Routes

**File:** `src/api/routes/sandbox.py` (CREATE NEW FILE)

```python
"""
Sandbox API Routes - REST endpoints for sandbox operations.
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

from src.utils.sandbox import SandboxManager, SandboxConfig, SandboxResult
from src.utils.structured_logging import get_logger

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
```

---

## Testing Requirements

**File:** `tests/test_sandbox.py` (CREATE NEW FILE)

```python
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
        source_files = {
            "src/slow.py": "import time; time.sleep(1000)"
        }
        
        test_files = {
            "tests/test_slow.py": """
import sys
sys.path.insert(0, '/app')
from src.slow import *

def test_slow():
    pass
"""
        }
        
        config = SandboxConfig(timeout_seconds=5)
        
        result = await manager.run_tests(source_files, test_files, config)
        
        # Should timeout
        assert result.status == SandboxStatus.TIMEOUT
    
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
```

---

## Acceptance Criteria

- [ ] `src/utils/sandbox.py` - Docker-based sandbox environment
- [ ] `src/master_ai/evolution_test_runner.py` - Multi-phase test execution
- [ ] `src/master_ai/test_generator.py` - LLM-based test generation
- [ ] `src/api/routes/sandbox.py` - REST API for sandbox operations
- [ ] `tests/test_sandbox.py` - All tests passing
- [ ] Sandbox isolation working (network, resources)
- [ ] Test results parsed correctly
- [ ] Cleanup working properly

---

## File Summary

| Action | File Path |
|--------|-----------|
| REPLACE | `src/utils/sandbox.py` |
| CREATE | `src/master_ai/evolution_test_runner.py` |
| CREATE | `src/master_ai/test_generator.py` |
| CREATE | `src/api/routes/sandbox.py` |
| CREATE | `tests/test_sandbox.py` |

---

*End of Part 9*
