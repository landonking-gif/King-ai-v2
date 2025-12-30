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
from src.utils.logging import get_logger

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
        patchset: 'PatchSet',
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
        patchset: 'PatchSet'
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
        patchset: 'PatchSet'
    ) -> Tuple[bool, str]:
        """
        Quick validation of a patchset (syntax only).
        
        Returns:
            Tuple of (valid, error_message)
        """
        source_files = await self._prepare_source_files(patchset)
        result = await self._run_syntax_check(source_files)
        
        return result.passed, result.error_summary or ""


# Add PatchSet stub for type hints
class PatchSet:
    """Stub for PatchSet - should be defined in code_patcher.py"""
    def __init__(self):
        self.patches = []
