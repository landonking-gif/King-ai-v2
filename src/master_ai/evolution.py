"""
Evolution Engine - Manages self-modification proposals and execution.
Enhanced with confidence scoring, validation, and approval workflows.
"""

import asyncio
import json
import re
import ast
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from pathlib import Path

from src.master_ai.evolution_models import (
    EvolutionProposal, ProposalStatus, ProposalType, RiskLevel,
    ValidationResult, EvolutionHistory, EvolutionMetrics, CodeChange, ConfidenceScore
)
from src.master_ai.confidence_scorer import ConfidenceScorer
from src.master_ai.prompts import EVOLUTION_PROPOSAL_PROMPT, VALIDATION_PROMPT, EVOLUTION_PROMPT
from src.utils.llm_router import LLMRouter, TaskContext
from src.utils.structured_logging import get_logger
from src.utils.retry import with_retry, LLM_RETRY_CONFIG
from src.utils.sandbox import Sandbox
from src.database.connection import get_db
from src.database.models import EvolutionProposal as DBEvolutionProposal
from config.settings import settings

logger = get_logger("evolution_engine")


class EvolutionEngine:
    """
    Enhanced evolution engine with comprehensive proposal management.
    
    Features:
    - Proposal generation with confidence scoring
    - Validation and testing
    - Approval workflows
    - Safe execution with rollback
    - Historical tracking and metrics
    """
    
    def __init__(self, llm_router: LLMRouter, sandbox: Optional[Sandbox] = None):
        """
        Initialize the evolution engine.
        
        Args:
            llm_router: LLM router for proposal generation
            sandbox: Code testing sandbox (optional)
        """
        self.llm = llm_router
        self.sandbox = sandbox
        self.confidence_scorer = ConfidenceScorer(llm_router, self)
        
        # Metrics
        self.metrics = EvolutionMetrics()
        
        # Active proposals
        self._active_proposals: Dict[str, EvolutionProposal] = {}
        
        # Daily limit for proposals (spec: 1 per day)
        self._daily_proposal_count = 0
        self._last_proposal_date = None
        self._max_daily_proposals = 1  # Spec requirement
    
    async def propose_improvement(
        self,
        goal: str = None,
        context: str = None,
        proposal_type: ProposalType = None,
        constraints: List[str] = None
    ) -> EvolutionProposal:
        """
        Generate an evolution proposal to achieve a goal.
        
        Args:
            goal: The improvement goal (optional, for backward compatibility)
            context: Current system context (optional, for backward compatibility)
            proposal_type: Type of proposal (optional)
            constraints: Constraints to consider (optional)
            
        Returns:
            Generated proposal
        """
        # Check daily limit
        today = datetime.now().date()
        if self._last_proposal_date != today:
            self._daily_proposal_count = 0
            self._last_proposal_date = today
        
        if self._daily_proposal_count >= self._max_daily_proposals:
            raise ValueError(
                f"Daily proposal limit reached ({self._max_daily_proposals}/day). "
                "Try again tomorrow."
            )
        
        # Handle legacy interface - if called with just context string
        if goal is None and isinstance(context, str):
            # Use old EVOLUTION_PROMPT format
            from src.master_ai.kpi_monitor import kpi_monitor
            try:
                health_report = await kpi_monitor.get_system_health()
            except:
                health_report = "System health data unavailable"
            
            prompt = EVOLUTION_PROMPT.format(
                context=context or "No context provided",
                performance=health_report
            )
            
            response = await self.llm.complete(prompt)
            
            try:
                if "```json" in response:
                    response = response.split("```json")[1].split("```")[0].strip()
                elif "```" in response:
                    response = response.split("```")[1].split("```")[0].strip()
                
                data = json.loads(response)
                
                if not data.get("is_beneficial"):
                    # Create a minimal rejected proposal for backward compatibility
                    proposal = EvolutionProposal(
                        title="No improvement needed",
                        description=data.get("reason", "System is currently optimal"),
                        proposal_type=ProposalType.CODE_MODIFICATION,
                        status=ProposalStatus.REJECTED
                    )
                    return proposal
                
                # Convert old format to new format
                proposal_data = {
                    "title": data.get("description", "System improvement")[:100],
                    "description": data.get("description", ""),
                    "changes": [],
                    "config_changes": data.get("changes", {}),
                }
            except Exception as e:
                logger.warning("Error parsing legacy evolution proposal", error=str(e))
                proposal_data = {
                    "title": "Parsing error",
                    "description": str(e),
                    "changes": [],
                    "config_changes": {}
                }
        else:
            # Use new interface
            goal = goal or "Improve system performance"
            context = context or "No specific context provided"
            logger.info("Generating evolution proposal", goal=goal[:100])
            
            # Generate proposal using LLM
            proposal_data = await self._generate_proposal(goal, context, proposal_type, constraints)
        
        # Create proposal object
        proposal = EvolutionProposal(
            title=proposal_data.get("title", f"Improvement: {goal[:50] if goal else 'System'}"),
            description=proposal_data.get("description", goal or "System improvement"),
            proposal_type=proposal_type or ProposalType.CODE_MODIFICATION,
            changes=self._parse_changes(proposal_data.get("changes", [])),
            configuration_changes=proposal_data.get("config_changes", {}),
            metadata={"goal": goal or "", "context": (context or "")[:500]}
        )
        
        # Calculate risk level
        proposal.risk_level = proposal.calculate_risk_level()
        
        # Score confidence
        proposal.confidence_score = await self.confidence_scorer.score_proposal(proposal)
        
        # Determine if approval needed
        proposal.requires_approval = (
            proposal.is_high_risk() or
            not self.confidence_scorer.meets_threshold(
                proposal.confidence_score,
                proposal.risk_level.value
            )
        )
        
        # Set initial status
        if proposal.requires_approval:
            proposal.status = ProposalStatus.READY
        else:
            proposal.status = ProposalStatus.APPROVED
        
        # Persist to database
        await self._persist_proposal(proposal)
        
        # Add to active proposals
        self._active_proposals[proposal.id] = proposal
        
        # Increment counter after successful proposal
        self._daily_proposal_count += 1
        
        logger.info(
            "Proposal generated",
            proposal_id=proposal.id,
            risk=proposal.risk_level.value,
            confidence=proposal.confidence_score.overall if proposal.confidence_score else 0,
            requires_approval=proposal.requires_approval
        )
        
        return proposal
    
    @with_retry(LLM_RETRY_CONFIG)
    async def _generate_proposal(
        self,
        goal: str,
        context: str,
        proposal_type: ProposalType = None,
        constraints: List[str] = None
    ) -> Dict[str, Any]:
        """Generate proposal using LLM."""
        risk_profile = getattr(settings, "risk_profile", "moderate")
        if not isinstance(risk_profile, str):
            risk_profile = "moderate"

        prompt = EVOLUTION_PROPOSAL_PROMPT.format(
            goal=goal,
            context=context[:3000],
            proposal_type=proposal_type.value if proposal_type else "auto",
            constraints="\n".join(f"- {c}" for c in (constraints or [])),
            risk_profile=risk_profile
        )
        
        llm_context = TaskContext(
            task_type="code_generation",
            risk_level="medium",
            requires_accuracy=True,
            token_estimate=1500,
            priority="high"
        )
        
        response = await self.llm.complete(prompt, context=llm_context)
        
        # Parse JSON response
        try:
            # Extract JSON from markdown code blocks if present
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()
                
            return json.loads(response)
        except json.JSONDecodeError:
            logger.warning("Failed to parse proposal JSON", response=response[:500])
            # Fallback parsing
            return self._parse_fallback_proposal(response)
    
    def _parse_changes(self, changes_data: List[Dict]) -> List[CodeChange]:
        """Parse code changes from proposal data."""
        changes = []
        for change_data in changes_data:
            try:
                change = CodeChange(
                    file_path=change_data.get("file_path", ""),
                    change_type=change_data.get("change_type", "modify"),
                    old_content=change_data.get("old_content"),
                    new_content=change_data.get("new_content"),
                    line_start=change_data.get("line_start"),
                    line_end=change_data.get("line_end"),
                    description=change_data.get("description", "")
                )
                changes.append(change)
            except Exception as e:
                logger.warning("Failed to parse change", error=str(e))
        return changes
    
    def _parse_fallback_proposal(self, response: str) -> Dict[str, Any]:
        """Fallback parsing for malformed JSON responses."""
        # Extract title
        title_match = re.search(r'"title"\s*:\s*"([^"]+)"', response)
        title = title_match.group(1) if title_match else "Generated Proposal"
        
        # Extract description
        desc_match = re.search(r'"description"\s*:\s*"([^"]+)"', response)
        description = desc_match.group(1) if desc_match else "Auto-generated improvement"
        
        return {
            "title": title,
            "description": description,
            "changes": [],
            "config_changes": {}
        }
    
    async def validate_proposal(self, proposal: EvolutionProposal) -> ValidationResult:
        """
        Validate a proposal through testing and analysis.
        
        Args:
            proposal: The proposal to validate
            
        Returns:
            Validation results
        """
        logger.info("Validating proposal", proposal_id=proposal.id)
        
        proposal.status = ProposalStatus.VALIDATING
        
        result = ValidationResult(passed=True)
        start_time = datetime.now()
        
        try:
            # Run syntax checks
            syntax_ok = await self._validate_syntax(proposal)
            if not syntax_ok:
                result.passed = False
                result.errors.append("Syntax validation failed")
            
            # Run sandbox tests if sandbox is available
            if result.passed and self.sandbox:
                test_result = await self._run_sandbox_tests(proposal)
                result.tests_run = test_result.get("tests_run", 0)
                result.tests_passed = test_result.get("tests_passed", 0)
                result.tests_failed = test_result.get("tests_failed", 0)
                
                if result.tests_failed > 0:
                    result.passed = False
                    result.errors.extend(test_result.get("errors", []))
            
            # LLM-based validation
            if result.passed:
                llm_validation = await self._validate_with_llm(proposal)
                if not llm_validation["passed"]:
                    result.passed = False
                    result.errors.extend(llm_validation["issues"])
            
        except Exception as e:
            result.passed = False
            result.errors.append(f"Validation error: {str(e)}")
            logger.error("Proposal validation failed", error=str(e), exc_info=True)
        
        result.execution_time_seconds = (datetime.now() - start_time).total_seconds()
        proposal.validation_result = result
        
        # Update status
        if result.passed:
            proposal.status = ProposalStatus.READY
        else:
            proposal.status = ProposalStatus.DRAFT
        
        # Persist validation result
        await self._update_proposal(proposal)
        
        logger.info(
            "Validation completed",
            proposal_id=proposal.id,
            passed=result.passed,
            tests_run=result.tests_run,
            errors=len(result.errors)
        )
        
        return result
    
    async def _validate_syntax(self, proposal: EvolutionProposal) -> bool:
        """Validate syntax of code changes."""
        for change in proposal.changes:
            if change.new_content and change.file_path.endswith('.py'):
                try:
                    ast.parse(change.new_content)
                except SyntaxError as e:
                    logger.warning(
                        "Syntax error in proposal",
                        file=change.file_path,
                        error=str(e)
                    )
                    return False
        return True
    
    async def _run_sandbox_tests(self, proposal: EvolutionProposal) -> Dict[str, Any]:
        """Run tests in sandbox environment."""
        # This would integrate with the sandbox system
        # For now, return mock results
        return {
            "tests_run": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "errors": []
        }
    
    @with_retry(LLM_RETRY_CONFIG)
    async def _validate_with_llm(self, proposal: EvolutionProposal) -> Dict[str, Any]:
        """Use LLM to validate proposal quality."""
        prompt = VALIDATION_PROMPT.format(
            title=proposal.title,
            description=proposal.description,
            changes=json.dumps([
                {
                    "file": c.file_path,
                    "type": c.change_type,
                    "content": c.new_content[:200] if c.new_content else ""
                } for c in proposal.changes
            ], indent=2)
        )
        
        response = await self.llm.complete(prompt)
        
        # Parse response
        if "ISSUES FOUND" in response.upper():
            return {"passed": False, "issues": [response]}
        else:
            return {"passed": True, "issues": []}
    
    async def approve_proposal(
        self,
        proposal_id: str,
        approver: str = "system"
    ) -> bool:
        """
        Approve a proposal for execution.
        
        Args:
            proposal_id: ID of proposal to approve
            approver: Who approved it
            
        Returns:
            Success status
        """
        proposal = self._active_proposals.get(proposal_id)
        if not proposal:
            # Load from database
            proposal = await self._load_proposal(proposal_id)
            if not proposal:
                return False
        
        if proposal.status not in [ProposalStatus.READY, ProposalStatus.APPROVED]:
            return False
        
        proposal.status = ProposalStatus.APPROVED
        proposal.approved_by = approver
        proposal.approved_at = datetime.now()
        
        await self._update_proposal(proposal)
        
        logger.info("Proposal approved", proposal_id=proposal_id, approver=approver)
        return True
    
    async def execute_proposal(self, proposal: EvolutionProposal) -> Dict[str, Any]:
        """
        Execute an approved proposal.
        
        Args:
            proposal: The proposal to execute
            
        Returns:
            Execution results
        """
        if not proposal.can_execute():
            raise ValueError("Proposal is not ready for execution")
        
        logger.info("Executing proposal", proposal_id=proposal.id)
        
        proposal.status = ProposalStatus.EXECUTING
        
        # Create rollback data
        rollback_data = await self._create_rollback_data(proposal)
        proposal.rollback_data = rollback_data
        
        success = True
        errors = []
        
        try:
            # Apply changes
            for change in proposal.changes:
                await self._apply_change(change)
            
            # Apply configuration changes
            if proposal.configuration_changes:
                await self._apply_config_changes(proposal.configuration_changes)
            
            # Run post-execution validation
            validation_ok = await self._validate_execution(proposal)
            if not validation_ok:
                success = False
                errors.append("Post-execution validation failed")
        
        except Exception as e:
            success = False
            errors.append(f"Execution error: {str(e)}")
            logger.error("Proposal execution failed", error=str(e), exc_info=True)
        
        # Update status
        if success:
            proposal.status = ProposalStatus.COMPLETED
            self.metrics.update_from_proposal(proposal)
        else:
            proposal.status = ProposalStatus.FAILED
            # Attempt rollback
            await self._rollback_proposal(proposal)
        
        proposal.execution_result = {
            "success": success,
            "errors": errors,
            "executed_at": datetime.now().isoformat()
        }
        
        # Persist results
        await self._update_proposal(proposal)
        
        # Record in history
        await self._record_history(proposal, "executed" if success else "failed")
        
        logger.info(
            "Proposal execution finished",
            proposal_id=proposal.id,
            success=success,
            errors=errors
        )
        
        return proposal.execution_result
    
    async def apply_proposal(self, file_path: str, new_code: str) -> dict:
        """
        Legacy method for backward compatibility.
        Physically applies a code modification to the codebase.
        """
        from src.utils.code_patcher import CodePatcher
        import os
        
        # Get project root (two levels up from this file)
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        
        # Initialize patcher with project root
        patcher = CodePatcher(project_root)
        
        # Create and apply patch
        patch = patcher.create_patch(file_path, new_code, description="Evolution proposal")
        
        # Validate before applying
        is_valid, errors = patcher.validate_patch(patch)
        if not is_valid:
            return {
                "success": False,
                "error": f"Validation failed: {', '.join(errors)}"
            }
        
        # Apply the patch
        success = patcher.apply_patch(patch)
        
        if success:
            return {
                "success": True,
                "message": "Patch applied successfully",
                "stats": patch.stats
            }
        else:
            return {
                "success": False,
                "error": patch.error or "Unknown error"
            }
    
    async def _apply_change(self, change: CodeChange):
        """Apply a single code change."""
        # This will be implemented in Part 5.5
        # For now, just log
        logger.info(
            "Applying change",
            file=change.file_path,
            type=change.change_type
        )
    
    async def _apply_config_changes(self, config_changes: Dict[str, Any]):
        """Apply configuration changes."""
        # This will be implemented in Part 5.5
        logger.info("Applying config changes", changes=config_changes)
    
    async def _create_rollback_data(self, proposal: EvolutionProposal) -> Dict[str, Any]:
        """Create rollback data for the proposal."""
        # This will be implemented in Part 5.75
        return {"placeholder": True}
    
    async def _rollback_proposal(self, proposal: EvolutionProposal):
        """Rollback a failed proposal."""
        # This will be implemented in Part 5.75
        logger.warning("Rolling back proposal", proposal_id=proposal.id)
    
    async def _validate_execution(self, proposal: EvolutionProposal) -> bool:
        """Validate that execution was successful."""
        # This will be implemented in Part 5.5
        return True
    
    async def get_similar_proposals(self, proposal: EvolutionProposal) -> List[EvolutionProposal]:
        """Get historically similar proposals."""
        # This will be implemented with vector search in Part 3
        return []
    
    async def _persist_proposal(self, proposal: EvolutionProposal):
        """Persist proposal to database."""
        try:
            async with get_db() as db:
                db_proposal = DBEvolutionProposal(
                    id=proposal.id,
                    type=proposal.proposal_type.value,
                    description=proposal.description,
                    rationale=proposal.description,
                    proposed_changes={
                        "changes": [c.dict() for c in proposal.changes],
                        "configuration_changes": proposal.configuration_changes
                    },
                    expected_impact=str(proposal.estimated_impact),
                    confidence_score=proposal.confidence_score.overall if proposal.confidence_score else 0.0,
                    status=proposal.status.value if hasattr(proposal.status, 'value') else str(proposal.status)
                )
                db.add(db_proposal)
                await db.commit()
        except Exception as e:
            logger.warning("Failed to persist proposal to database", error=str(e))
    
    async def _update_proposal(self, proposal: EvolutionProposal):
        """Update proposal in database."""
        try:
            async with get_db() as db:
                db_proposal = await db.get(DBEvolutionProposal, proposal.id)
                if db_proposal:
                    db_proposal.status = proposal.status.value if hasattr(proposal.status, 'value') else str(proposal.status)
                    db_proposal.confidence_score = proposal.confidence_score.overall if proposal.confidence_score else 0.0
                    await db.commit()
        except Exception as e:
            logger.warning("Failed to update proposal in database", error=str(e))
    
    async def _load_proposal(self, proposal_id: str) -> Optional[EvolutionProposal]:
        """Load proposal from database."""
        try:
            async with get_db() as db:
                db_proposal = await db.get(DBEvolutionProposal, proposal_id)
                if db_proposal:
                    return EvolutionProposal(
                        id=db_proposal.id,
                        title=db_proposal.description[:100],
                        description=db_proposal.description,
                        proposal_type=ProposalType.CODE_MODIFICATION,  # Default
                        status=ProposalStatus(db_proposal.status) if hasattr(ProposalStatus, db_proposal.status.upper()) else ProposalStatus.DRAFT
                    )
        except Exception as e:
            logger.warning("Failed to load proposal from database", error=str(e))
        return None
    
    async def _record_history(self, proposal: EvolutionProposal, event_type: str):
        """Record evolution event in history."""
        # This will be implemented in Part 5.75
        pass
    
    def get_metrics(self) -> EvolutionMetrics:
        """Get evolution metrics."""
        return self.metrics
    
    def get_active_proposals(self) -> List[EvolutionProposal]:
        """Get all active proposals."""
        return list(self._active_proposals.values())

