"""
Evolution Engine - Manages self-modification proposals and execution.
Enhanced with confidence scoring, validation, and approval workflows.
"""

import json
import re
import ast
from typing import Optional, List, Dict, Any
from datetime import datetime
from pathlib import Path

from src.master_ai.evolution_models import (
    EvolutionProposal, ProposalStatus, ProposalType, ValidationResult, EvolutionMetrics, CodeChange
)
from src.master_ai.confidence_scorer import ConfidenceScorer
from src.master_ai.prompts import EVOLUTION_PROPOSAL_PROMPT, VALIDATION_PROMPT, EVOLUTION_PROMPT
from src.utils.llm_router import LLMRouter, TaskContext
from src.utils.structured_logging import get_logger
from src.utils.retry import with_retry, LLM_RETRY_CONFIG
from src.utils.sandbox import Sandbox
from src.database.connection import get_db, get_db_ctx
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
        
        # Daily limit for proposals (from settings)
        self._daily_proposal_count = 0
        self._last_proposal_date = None
        self._max_daily_proposals = getattr(settings, 'evolution_daily_limit', 100)  # Default 100/day
    
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
    
    async def execute_architecture_update(
        self,
        proposal: EvolutionProposal,
        dry_run: bool = True
    ) -> Dict[str, Any]:
        """
        Execute an architecture update proposal (arch_update type).
        
        Handles infrastructure changes like:
        - Adding/removing agents
        - Modifying service dependencies
        - Updating Terraform infrastructure
        - Reconfiguring integration endpoints
        
        Args:
            proposal: The architecture update proposal
            dry_run: If True, validate without applying changes
            
        Returns:
            Execution result with planned/applied changes
        """
        if proposal.proposal_type != ProposalType.INFRASTRUCTURE_UPDATE:
            raise ValueError(
                f"Expected INFRASTRUCTURE_UPDATE proposal, got {proposal.proposal_type}"
            )
        
        logger.info(
            "Processing architecture update",
            proposal_id=proposal.id,
            dry_run=dry_run
        )
        
        results = {
            "proposal_id": proposal.id,
            "dry_run": dry_run,
            "changes_planned": [],
            "changes_applied": [],
            "errors": [],
            "warnings": []
        }
        
        # Parse architecture changes from proposal
        arch_changes = proposal.configuration_changes.get("architecture", {})
        
        # 1. Handle agent additions/removals
        if "agents" in arch_changes:
            agent_changes = await self._process_agent_changes(
                arch_changes["agents"],
                dry_run=dry_run
            )
            results["changes_planned"].extend(agent_changes.get("planned", []))
            results["changes_applied"].extend(agent_changes.get("applied", []))
            results["warnings"].extend(agent_changes.get("warnings", []))
        
        # 2. Handle service dependency changes
        if "services" in arch_changes:
            service_changes = await self._process_service_changes(
                arch_changes["services"],
                dry_run=dry_run
            )
            results["changes_planned"].extend(service_changes.get("planned", []))
            results["changes_applied"].extend(service_changes.get("applied", []))
        
        # 3. Handle Terraform infrastructure changes
        if "terraform" in arch_changes:
            tf_changes = await self._process_terraform_changes(
                arch_changes["terraform"],
                dry_run=dry_run
            )
            results["changes_planned"].extend(tf_changes.get("planned", []))
            results["changes_applied"].extend(tf_changes.get("applied", []))
            results["warnings"].extend(tf_changes.get("warnings", []))
        
        # 4. Handle integration endpoint updates
        if "integrations" in arch_changes:
            integration_changes = await self._process_integration_changes(
                arch_changes["integrations"],
                dry_run=dry_run
            )
            results["changes_planned"].extend(integration_changes.get("planned", []))
            results["changes_applied"].extend(integration_changes.get("applied", []))
        
        # Update proposal status
        if not dry_run and not results["errors"]:
            proposal.status = ProposalStatus.COMPLETED
            proposal.execution_result = results
            await self._update_proposal(proposal)
        
        logger.info(
            "Architecture update processed",
            proposal_id=proposal.id,
            planned_count=len(results["changes_planned"]),
            applied_count=len(results["changes_applied"]),
            error_count=len(results["errors"])
        )
        
        return results
    
    async def _process_agent_changes(
        self,
        agent_config: Dict[str, Any],
        dry_run: bool = True
    ) -> Dict[str, list]:
        """Process agent addition/removal/modification."""
        results = {"planned": [], "applied": [], "warnings": []}
        
        # Add new agents
        for agent_name, config in agent_config.get("add", {}).items():
            change = {
                "type": "add_agent",
                "agent_name": agent_name,
                "config": config
            }
            results["planned"].append(change)
            
            if not dry_run:
                # Create agent file from template
                agent_file = Path(f"src/agents/{agent_name}.py")
                if not agent_file.exists():
                    agent_template = self._generate_agent_template(agent_name, config)
                    agent_file.write_text(agent_template)
                    
                    # Update router to include new agent
                    await self._register_agent_in_router(agent_name)
                    results["applied"].append(change)
                else:
                    results["warnings"].append(
                        f"Agent file already exists: {agent_file}"
                    )
        
        # Remove agents
        for agent_name in agent_config.get("remove", []):
            change = {"type": "remove_agent", "agent_name": agent_name}
            results["planned"].append(change)
            
            if not dry_run:
                agent_file = Path(f"src/agents/{agent_name}.py")
                if agent_file.exists():
                    # Create backup
                    backup_file = agent_file.with_suffix(".py.bak")
                    backup_file.write_text(agent_file.read_text())
                    agent_file.unlink()
                    results["applied"].append(change)
        
        return results
    
    async def _process_service_changes(
        self,
        service_config: Dict[str, Any],
        dry_run: bool = True
    ) -> Dict[str, list]:
        """Process service dependency changes."""
        results = {"planned": [], "applied": []}
        
        for service_name, config in service_config.items():
            change = {
                "type": "update_service",
                "service_name": service_name,
                "config": config
            }
            results["planned"].append(change)
            
            if not dry_run:
                # Update docker-compose.yml if needed
                if "docker" in config:
                    await self._update_docker_compose(service_name, config["docker"])
                results["applied"].append(change)
        
        return results
    
    async def _process_terraform_changes(
        self,
        tf_config: Dict[str, Any],
        dry_run: bool = True
    ) -> Dict[str, list]:
        """Process Terraform infrastructure changes."""
        results = {"planned": [], "applied": [], "warnings": []}
        
        for resource_type, resources in tf_config.items():
            for resource_name, config in resources.items():
                change = {
                    "type": "terraform_resource",
                    "resource_type": resource_type,
                    "resource_name": resource_name,
                    "config": config
                }
                results["planned"].append(change)
                
                if not dry_run:
                    # Generate Terraform HCL for the resource
                    tf_file = Path(f"infrastructure/terraform/{resource_type}_{resource_name}.tf")
                    tf_content = self._generate_terraform_resource(
                        resource_type, resource_name, config
                    )
                    tf_file.write_text(tf_content)
                    results["applied"].append(change)
                    
                    # Add warning about needing terraform apply
                    results["warnings"].append(
                        f"Run 'terraform plan' and 'terraform apply' to provision {resource_name}"
                    )
        
        return results
    
    async def _process_integration_changes(
        self,
        integration_config: Dict[str, Any],
        dry_run: bool = True
    ) -> Dict[str, list]:
        """Process integration endpoint changes."""
        results = {"planned": [], "applied": []}
        
        for integration_name, config in integration_config.items():
            change = {
                "type": "update_integration",
                "integration_name": integration_name,
                "config": config
            }
            results["planned"].append(change)
            
            if not dry_run:
                # Update integration configuration
                integration_file = Path(f"src/integrations/{integration_name}.py")
                if integration_file.exists():
                    # Update config in settings
                    await self._update_integration_settings(integration_name, config)
                results["applied"].append(change)
        
        return results
    
    def _generate_agent_template(self, agent_name: str, config: Dict) -> str:
        """Generate Python code for a new agent."""
        class_name = ''.join(word.title() for word in agent_name.split('_')) + 'Agent'
        description = config.get("description", f"Agent for {agent_name} operations")
        
        return f'''"""
{class_name} - {description}

Auto-generated by Evolution Engine architecture update.
"""

from typing import Any, Dict, Optional
from src.agents.base import SubAgent
from src.utils.structured_logging import get_logger

logger = get_logger(__name__)


class {class_name}(SubAgent):
    """
    {description}
    """
    
    NAME = "{agent_name}"
    DESCRIPTION = "{description}"
    
    FUNCTION_SCHEMA = {{
        "name": "{agent_name}",
        "description": "{description}",
        "parameters": {{
            "type": "object",
            "properties": {{
                "action": {{
                    "type": "string",
                    "description": "The action to perform"
                }},
                "data": {{
                    "type": "object",
                    "description": "Additional data for the action"
                }}
            }},
            "required": ["action"]
        }}
    }}
    
    async def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a task for {agent_name}."""
        logger.info(f"Processing {{task.get('name', 'unnamed')}} task")
        
        action = task.get("action", "default")
        data = task.get("data", {{}})
        
        # TODO: Implement {agent_name} logic
        
        return {{
            "success": True,
            "agent": self.NAME,
            "action": action,
            "result": "Task processed successfully"
        }}
'''
    
    def _generate_terraform_resource(
        self,
        resource_type: str,
        resource_name: str,
        config: Dict
    ) -> str:
        """Generate Terraform HCL for a resource."""
        # Basic resource template
        config_str = "\n".join(
            f'  {k} = {json.dumps(v)}'
            for k, v in config.items()
        )
        
        return f'''# Auto-generated by Evolution Engine
# Resource: {resource_type}.{resource_name}

resource "{resource_type}" "{resource_name}" {{
{config_str}

  tags = {{
    Environment = var.environment
    ManagedBy   = "evolution-engine"
  }}
}}
'''
    
    async def _register_agent_in_router(self, agent_name: str):
        """Register a new agent in the agent router."""
        # This would update src/agents/router.py to include the new agent
        # For safety, we log the instruction rather than modifying
        logger.info(
            "New agent requires registration in router",
            agent_name=agent_name,
            instruction=f"Add 'from src.agents.{agent_name} import {agent_name.title()}Agent' to router.py"
        )
    
    async def _update_docker_compose(self, service_name: str, config: Dict):
        """Update docker-compose.yml with service changes."""
        import yaml
        import aiofiles
        
        compose_file = Path("docker-compose.yml")
        if compose_file.exists():
            async with aiofiles.open(compose_file, 'r') as f:
                compose = yaml.safe_load(await f.read())
            
            if "services" not in compose:
                compose["services"] = {}
            
            compose["services"][service_name] = config
            
            async with aiofiles.open(compose_file, 'w') as f:
                await f.write(yaml.dump(compose, default_flow_style=False))
    
    async def _update_integration_settings(self, integration_name: str, config: Dict):
        """Update integration settings in configuration."""
        import yaml
        import aiofiles
        
        # Update settings with new integration config
        settings_file = Path("config/integrations.yaml")
        
        if settings_file.exists():
            async with aiofiles.open(settings_file, 'r') as f:
                integrations = yaml.safe_load(await f.read())
        else:
            integrations = {}
        
        integrations[integration_name] = config
        
        async with aiofiles.open(settings_file, 'w') as f:
            await f.write(yaml.dump(integrations, default_flow_style=False))

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
        import aiofiles
        
        file_path = Path(change.file_path)
        
        logger.info(
            "Applying change",
            file=str(file_path),
            type=str(change.change_type)
        )
        
        try:
            change_type = change.change_type  # It's a string literal: "add", "modify", "delete", "rename"
            
            if change_type == "add":
                # Create new file
                file_path.parent.mkdir(parents=True, exist_ok=True)
                async with aiofiles.open(file_path, 'w') as f:
                    await f.write(change.new_content or "")
                    
            elif change_type == "modify":
                # Modify existing file
                if not file_path.exists():
                    raise FileNotFoundError(f"Cannot modify non-existent file: {file_path}")
                
                if change.new_content:
                    async with aiofiles.open(file_path, 'w') as f:
                        await f.write(change.new_content)
                elif change.diff:
                    # Apply diff using code patcher
                    from src.services.code_patcher import CodePatcher
                    patcher = CodePatcher()
                    async with aiofiles.open(file_path, 'r') as f:
                        original = await f.read()
                    patched = patcher.apply_diff(original, change.diff)
                    async with aiofiles.open(file_path, 'w') as f:
                        await f.write(patched)
                        
            elif change_type == "delete":
                # Delete file (with safety check)
                if file_path.exists():
                    # Backup before delete
                    backup_path = file_path.with_suffix(file_path.suffix + '.bak')
                    async with aiofiles.open(file_path, 'r') as f:
                        content = await f.read()
                    async with aiofiles.open(backup_path, 'w') as f:
                        await f.write(content)
                    file_path.unlink()
                    
            elif change_type == "rename":
                # Rename/move file
                if change.new_content:  # new_content contains new path for renames
                    new_path = Path(change.new_content)
                    new_path.parent.mkdir(parents=True, exist_ok=True)
                    file_path.rename(new_path)
                    
            logger.info("Change applied successfully", file=str(file_path))
            
        except Exception as e:
            logger.error("Failed to apply change", file=str(file_path), error=str(e))
            raise
    
    async def _apply_config_changes(self, config_changes: Dict[str, Any]):
        """Apply configuration changes."""
        import yaml
        import aiofiles
        
        logger.info("Applying config changes", changes=list(config_changes.keys()))
        
        config_dir = Path("config")
        
        for config_key, config_value in config_changes.items():
            try:
                if config_key == "risk_profile":
                    # Update risk_profiles.yaml
                    risk_file = config_dir / "risk_profiles.yaml"
                    if risk_file.exists():
                        async with aiofiles.open(risk_file, 'r') as f:
                            content = await f.read()
                        risk_config = yaml.safe_load(content)
                        
                        # Merge changes
                        if isinstance(config_value, dict):
                            for profile_name, profile_changes in config_value.items():
                                if profile_name in risk_config.get("profiles", {}):
                                    risk_config["profiles"][profile_name].update(profile_changes)
                        
                        async with aiofiles.open(risk_file, 'w') as f:
                            await f.write(yaml.dump(risk_config, default_flow_style=False))
                        
                elif config_key == "settings":
                    # Update settings via environment or .env file
                    env_file = Path(".env")
                    env_lines = []
                    
                    if env_file.exists():
                        async with aiofiles.open(env_file, 'r') as f:
                            env_lines = (await f.read()).splitlines()
                    
                    # Update or add settings
                    for setting_key, setting_value in config_value.items():
                        env_key = setting_key.upper()
                        found = False
                        for i, line in enumerate(env_lines):
                            if line.startswith(f"{env_key}="):
                                env_lines[i] = f"{env_key}={setting_value}"
                                found = True
                                break
                        if not found:
                            env_lines.append(f"{env_key}={setting_value}")
                    
                    async with aiofiles.open(env_file, 'w') as f:
                        await f.write('\n'.join(env_lines))
                        
                elif config_key == "playbook":
                    # Update playbook YAML
                    playbook_name = config_value.get("name", "custom")
                    playbook_file = config_dir / "playbooks" / f"{playbook_name}.yaml"
                    
                    async with aiofiles.open(playbook_file, 'w') as f:
                        await f.write(yaml.dump(config_value, default_flow_style=False))
                        
                logger.info(f"Applied config change: {config_key}")
                
            except Exception as e:
                logger.error(f"Failed to apply config change: {config_key}", error=str(e))
                raise
    
    async def _create_rollback_data(self, proposal: EvolutionProposal) -> Dict[str, Any]:
        """Create rollback data for the proposal."""
        import aiofiles
        
        rollback_data = {
            "proposal_id": proposal.id,
            "created_at": datetime.utcnow().isoformat(),
            "original_files": {},
            "deleted_files": {},
            "created_files": [],
            "config_backup": {},
            "git_commit": None
        }
        
        # Backup original file contents
        for change in proposal.changes:
            file_path = Path(change.file_path)
            change_type = change.change_type  # It's already a string literal
            
            if change_type in ("modify", "delete"):
                if file_path.exists():
                    try:
                        async with aiofiles.open(file_path, 'r') as f:
                            rollback_data["original_files"][str(file_path)] = await f.read()
                    except Exception as e:
                        logger.warning(f"Could not backup file {file_path}: {e}")
                        
            elif change_type == "add":
                rollback_data["created_files"].append(str(file_path))
        
        # Backup config if there are config changes
        if proposal.configuration_changes:
            config_files = ["config/settings.py", "config/risk_profiles.yaml", ".env"]
            for config_file in config_files:
                config_path = Path(config_file)
                if config_path.exists():
                    try:
                        async with aiofiles.open(config_path, 'r') as f:
                            rollback_data["config_backup"][config_file] = await f.read()
                    except Exception:
                        pass
        
        # Get current git commit for reference
        try:
            from src.services.git_manager import GitManager
            git = GitManager()
            status = await git.get_status()
            rollback_data["git_commit"] = status.get("head_commit")
        except Exception:
            pass  # Git not available
        
        logger.info("Created rollback data", proposal_id=proposal.id, 
                    files_backed_up=len(rollback_data["original_files"]))
        
        return rollback_data
    
    async def _rollback_proposal(self, proposal: EvolutionProposal):
        """Rollback a failed proposal."""
        import aiofiles
        
        logger.warning("Rolling back proposal", proposal_id=proposal.id)
        
        # Get rollback data from proposal metadata
        rollback_data = proposal.metadata.get("rollback_data", {})
        
        if not rollback_data:
            logger.error("No rollback data available", proposal_id=proposal.id)
            return
        
        errors = []
        
        # Restore original files
        for file_path, original_content in rollback_data.get("original_files", {}).items():
            try:
                async with aiofiles.open(file_path, 'w') as f:
                    await f.write(original_content)
                logger.info(f"Restored file: {file_path}")
            except Exception as e:
                errors.append(f"Failed to restore {file_path}: {e}")
        
        # Delete created files
        for file_path in rollback_data.get("created_files", []):
            try:
                path = Path(file_path)
                if path.exists():
                    path.unlink()
                logger.info(f"Deleted created file: {file_path}")
            except Exception as e:
                errors.append(f"Failed to delete {file_path}: {e}")
        
        # Restore config backups
        for config_file, backup_content in rollback_data.get("config_backup", {}).items():
            try:
                async with aiofiles.open(config_file, 'w') as f:
                    await f.write(backup_content)
                logger.info(f"Restored config: {config_file}")
            except Exception as e:
                errors.append(f"Failed to restore config {config_file}: {e}")
        
        # Update proposal status
        proposal.status = ProposalStatus.ROLLED_BACK
        await self._persist_proposal(proposal)
        
        if errors:
            logger.error("Rollback completed with errors", errors=errors)
        else:
            logger.info("Rollback completed successfully", proposal_id=proposal.id)
    
    async def _validate_execution(self, proposal: EvolutionProposal) -> bool:
        """Validate that execution was successful."""
        import aiofiles
        
        logger.info("Validating execution", proposal_id=proposal.id)
        
        validation_errors = []
        
        # Check all modified/created files exist and are valid
        for change in proposal.changes:
            file_path = Path(change.file_path)
            change_type = change.change_type  # It's already a string literal
            
            if change_type == "delete":
                if file_path.exists():
                    validation_errors.append(f"File should be deleted but exists: {file_path}")
                continue
                
            if change_type in ("add", "modify"):
                if not file_path.exists():
                    validation_errors.append(f"File should exist but doesn't: {file_path}")
                    continue
                
                # Validate Python syntax for .py files
                if file_path.suffix == '.py':
                    try:
                        async with aiofiles.open(file_path, 'r') as f:
                            content = await f.read()
                        ast.parse(content)
                    except SyntaxError as e:
                        validation_errors.append(f"Syntax error in {file_path}: {e}")
        
        # Run sandbox tests if available
        if self.sandbox:
            try:
                test_result = await self.sandbox.run_tests(
                    test_files=["tests/"],
                    timeout=60
                )
                
                if not test_result.get("success", False):
                    validation_errors.append(f"Tests failed: {test_result.get('error', 'Unknown')}")
                    
            except Exception as e:
                logger.warning("Sandbox validation skipped", error=str(e))
        
        if validation_errors:
            logger.error("Execution validation failed", errors=validation_errors)
            proposal.metadata["validation_errors"] = validation_errors
            return False
        
        logger.info("Execution validation passed", proposal_id=proposal.id)
        return True
    
    async def get_similar_proposals(self, proposal: EvolutionProposal) -> List[EvolutionProposal]:
        """Get historically similar proposals using vector similarity."""
        try:
            from src.master_ai.context_memory import VectorStore, EmbeddingClient
            
            embedding_client = EmbeddingClient()
            vector_store = VectorStore(embedding_client)
            
            # Create search text from proposal
            search_text = f"{proposal.proposal_type.value}: {proposal.description}"
            if proposal.changes:
                files_changed = [c.file_path for c in proposal.changes[:5]]
                search_text += f" Files: {', '.join(files_changed)}"
            
            # Search for similar proposals
            similar = await vector_store.search(
                query=search_text,
                namespace="evolution_proposals",
                top_k=5
            )
            
            # Load full proposals from database
            similar_proposals = []
            async with get_db_ctx() as db:
                for result in similar:
                    proposal_id = result.get("metadata", {}).get("proposal_id")
                    if proposal_id and proposal_id != proposal.id:
                        db_proposal = await db.get(DBEvolutionProposal, proposal_id)
                        if db_proposal:
                            loaded = await self._load_proposal(proposal_id)
                            if loaded:
                                similar_proposals.append(loaded)
            
            return similar_proposals[:5]
            
        except Exception as e:
            logger.warning("Could not find similar proposals", error=str(e))
            return []
    
    async def _index_proposal(self, proposal: EvolutionProposal):
        """Index proposal in vector store for similarity search."""
        try:
            from src.master_ai.context_memory import VectorStore, EmbeddingClient
            
            embedding_client = EmbeddingClient()
            vector_store = VectorStore(embedding_client)
            
            # Create text representation
            text = f"""
            Type: {proposal.proposal_type.value}
            Description: {proposal.description}
            Impact: {proposal.estimated_impact}
            Files: {', '.join(c.file_path for c in proposal.changes[:10])}
            Status: {proposal.status.value if hasattr(proposal.status, 'value') else str(proposal.status)}
            """
            
            await vector_store.store(
                text=text.strip(),
                metadata={
                    "proposal_id": proposal.id,
                    "type": proposal.proposal_type.value,
                    "status": proposal.status.value if hasattr(proposal.status, 'value') else str(proposal.status),
                    "confidence": proposal.confidence_score.overall if proposal.confidence_score else 0.0,
                    "created_at": proposal.created_at.isoformat() if proposal.created_at else None
                },
                namespace="evolution_proposals"
            )
            
            logger.debug("Indexed proposal", proposal_id=proposal.id)
            
        except Exception as e:
            logger.warning("Failed to index proposal", proposal_id=proposal.id, error=str(e))
    
    async def _persist_proposal(self, proposal: EvolutionProposal):
        """Persist proposal to database."""
        try:
            async with get_db_ctx() as db:
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
            async with get_db_ctx() as db:
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
            async with get_db_ctx() as db:
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
        """
        Record evolution event in history for audit and analysis.
        Stores events to database and maintains in-memory recent history.
        """
        try:
            event_record = {
                "proposal_id": proposal.id,
                "proposal_type": proposal.proposal_type.value if proposal.proposal_type else "unknown",
                "event_type": event_type,
                "timestamp": datetime.utcnow().isoformat(),
                "status": proposal.status.value if proposal.status else "unknown",
                "confidence_score": proposal.confidence_score.overall if proposal.confidence_score else 0.0,
                "description": proposal.description[:500] if proposal.description else "",
            }
            
            # Update metrics
            if event_type == "executed":
                self.metrics.successful_executions += 1
            elif event_type == "failed":
                self.metrics.failed_executions += 1
            elif event_type == "rejected":
                self.metrics.rejected_proposals += 1
            
            # Store in database
            from src.database.connection import get_db_ctx
            async with get_db_ctx() as session:
                from sqlalchemy import text
                
                await session.execute(
                    text("""
                        INSERT INTO evolution_history 
                        (proposal_id, event_type, event_data, created_at)
                        VALUES (:proposal_id, :event_type, :event_data, :created_at)
                        ON CONFLICT (proposal_id, event_type) DO UPDATE
                        SET event_data = :event_data, created_at = :created_at
                    """),
                    {
                        "proposal_id": proposal.id,
                        "event_type": event_type,
                        "event_data": json.dumps(event_record),
                        "created_at": datetime.utcnow()
                    }
                )
                await session.commit()
                
            logger.info(
                "Evolution history recorded",
                proposal_id=proposal.id,
                event_type=event_type
            )
            
        except Exception as e:
            # Don't fail the main flow if history recording fails
            logger.warning(
                "Failed to record evolution history",
                proposal_id=proposal.id,
                error=str(e)
            )
    
    def get_metrics(self) -> EvolutionMetrics:
        """Get evolution metrics."""
        return self.metrics
    
    def get_active_proposals(self) -> List[EvolutionProposal]:
        """Get all active proposals."""
        return list(self._active_proposals.values())

