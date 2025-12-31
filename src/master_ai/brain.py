"""
Master AI Brain - The central orchestrator for the autonomous empire.

This is the ONLY component that directly interfaces with the LLM (via LLMRouter).
All strategic decisions, operational planning, and agent delegation flow through this class.
It serves as the "CEO" of the business empire.
"""

import json
import asyncio
import time
from datetime import datetime
from typing import Optional, Any
from uuid import uuid4

# Internal project imports
from src.master_ai.context import ContextManager
from src.master_ai.planner import Planner
from src.master_ai.evolution import EvolutionEngine
from src.master_ai.prompts import SYSTEM_PROMPT, INTENT_CLASSIFICATION_PROMPT
from src.master_ai.models import (
    ClassifiedIntent, IntentType, ActionType, MasterAIResponse,
    ActionResult, TokenUsage, PlanStep
)
from src.agents.router import AgentRouter
from src.database.connection import get_db
from src.database.models import ConversationMessage, Task, EvolutionProposal
from src.utils.llm_router import LLMRouter, TaskContext
from src.utils.structured_logging import get_logger, set_request_context
from src.utils.retry import with_retry, LLM_RETRY_CONFIG, TransientError
from src.utils.monitoring import monitor, trace_llm
from config.settings import settings


logger = get_logger("master_ai")

class MasterAI:
    """
    The Master AI core that manages the autonomous lifecycle of businesses.
    
    Key Responsibilities:
    1. Intent Classification: Deciding if a user wants to chat, query data, or issue commands.
    2. Planning: Breaking down complex user goals into executable subtasks.
    3. Delegation: Routing subtasks to specialized agents (e.g., ResearchAgent).
    4. Autonomous Optimization: Periodic background checks to improve business performance.
    5. Evolution: Proposing source code or structural changes to itself.
    """
    
    def __init__(self):
        """Initializes all sub-components and management engines."""
        logger.info("Initializing Master AI brain")
        
        # Primary LLM interface - handles all provider routing and fallback
        self.llm_router = LLMRouter()
        
        # Specialized manager components
        self.context = ContextManager()      # Handles RAG and context window building
        self.planner = Planner(self.llm_router)  # Uses LLM router for intelligent routing
        # EvolutionEngine expects the router interface (not a raw provider client)
        self.evolution = EvolutionEngine(self.llm_router)
        self.agent_router = AgentRouter()    # Dispatches tasks to sub-agents
        
        # State and rate limiting for autonomous features
        self.autonomous_mode = getattr(settings, 'enable_autonomous_mode', False)
        self._evolution_count_this_hour = 0
        self._hour_start = datetime.now()
        self._total_tokens_today = 0
        self._token_budget_daily = 1_000_000  # 1M tokens per day
        
        risk_profile = getattr(settings, "risk_profile", "moderate")
        if not isinstance(risk_profile, str):
            risk_profile = "moderate"

        logger.info(
            "Master AI initialized",
            autonomous_mode=self.autonomous_mode,
            risk_profile=risk_profile
        )
    
    async def process_input(self, user_input: str, request_id: str = None) -> MasterAIResponse:
        """
        Primary entry point for any user interaction (API or CLI).
        
        Logic:
        1. Build the current global context (financials, active tasks, history).
        2. Use the LLM to classify the user's intent.
        3. Dispatch to the appropriate handler (conversation, command, or query).
        
        Args:
            user_input: The user's message
            request_id: Optional request ID for tracing
            
        Returns:
            Structured MasterAIResponse
        """
        request_id = request_id or str(uuid4())
        set_request_context(request_id)
        start_time = time.time()
        
        logger.info("Processing user input", input_length=len(user_input))
        monitor.increment("master_ai.requests")
        
        try:
            # 1. Gather all relevant data for the LLM to make informed decisions
            context = await self.context.build_context()
            
            # 2. Classify the user's intent with structured output
            intent = await self._classify_intent(user_input, context)
            logger.info(
                "Intent classified",
                intent_type=intent.type.value,
                action=intent.action.value if intent.action else None,
                confidence=intent.confidence
            )
            
            # 3. Route to appropriate handler based on intent
            if intent.type == IntentType.CONVERSATION:
                response = await self._handle_conversation(user_input, context)
                result = MasterAIResponse(
                    type="conversation",
                    response=response,
                    metadata={"intent_confidence": intent.confidence}
                )
                
            elif intent.type == IntentType.COMMAND:
                result = await self._handle_command(user_input, intent, context)
                
            elif intent.type == IntentType.QUERY:
                response = await self._handle_query(user_input, context)
                result = MasterAIResponse(
                    type="conversation",
                    response=response,
                    metadata={"intent_confidence": intent.confidence}
                )
            else:
                result = MasterAIResponse(
                    type="error",
                    response="Unknown intent type"
                )
            
            # Record metrics
            duration_ms = (time.time() - start_time) * 1000
            monitor.timing("master_ai.request_duration", duration_ms)
            monitor.increment("master_ai.requests_success")
            
            logger.info(
                "Request completed",
                duration_ms=duration_ms,
                response_type=result.type
            )
            
            return result
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            monitor.increment("master_ai.requests_error")
            logger.error("Request failed", exc_info=True, error=str(e))
            
            return MasterAIResponse(
                type="error",
                response=f"I encountered an error processing your request: {str(e)}",
                metadata={"error": str(e), "duration_ms": duration_ms}
            )

    @with_retry(LLM_RETRY_CONFIG)
    async def _classify_intent(self, user_input: str, context: str) -> ClassifiedIntent:
        """
        Uses the LLM as a router to interpret natural language into a structured intent.
        
        Returns a validated ClassifiedIntent object.
        """
        prompt = INTENT_CLASSIFICATION_PROMPT.format(
            user_input=user_input,
            context=context[:2000]  # Limit context for classification
        )
        
        # Use low-risk task context for classification
        task_context = TaskContext(
            task_type="classification",
            risk_level="low",
            requires_accuracy=False,
            token_estimate=500,
            priority="normal"
        )
        
        response = await self._call_llm(prompt, task_context=task_context)
        
        try:
            # Parse and validate the response
            parsed = self._parse_json_response(response)
            
            return ClassifiedIntent(
                type=IntentType(parsed.get("type", "conversation")),
                action=ActionType(parsed.get("action")) if parsed.get("action") else None,
                parameters=parsed.get("parameters", {}),
                confidence=parsed.get("confidence", 0.5),
                reasoning=parsed.get("reasoning")
            )
        except Exception as e:
            logger.warning(
                "Failed to parse intent, defaulting to conversation",
                error=str(e),
                response=response[:200]
            )
            return ClassifiedIntent(type=IntentType.CONVERSATION, confidence=0.3)
    
    async def _handle_conversation(self, user_input: str, context: str) -> str:
        """Handles chit-chat while maintaining awareness of the empire's state."""
        prompt = f"""{SYSTEM_PROMPT}

CURRENT CONTEXT OF THE EMPIRE:
{context}

USER MESSAGE:
{user_input}

Respond naturally as King AI, the autonomous CEO. Be professional, data-aware, and concise.
"""
        task_context = TaskContext(
            task_type="conversation",
            risk_level="low",
            requires_accuracy=False,
            token_estimate=1000,
            priority="normal"
        )
        
        return await self._call_llm(prompt, task_context=task_context)
    
    async def _handle_command(
        self,
        user_input: str,
        intent: ClassifiedIntent,
        context: str
    ) -> MasterAIResponse:
        """
        Orchestrates complex actions:
        1. Breaks goal into steps via the Planner.
        2. Checks for approval requirements based on Risk Profile.
        3. Executes immediate tasks or queues for user review.
        """
        logger.info(
            "Handling command",
            action=intent.action.value if intent.action else "unknown",
            parameters=intent.parameters
        )
        
        # Create a detailed multi-step plan
        plan = await self.planner.create_plan(
            goal=user_input,
            action=intent.action.value if intent.action else None,
            parameters=intent.parameters,
            context=context
        )
        
        actions_taken = []
        pending_approvals = []
        
        # Iterate through planned steps
        for step in plan.get("steps", []):
            # Convert dict to PlanStep if needed
            if isinstance(step, dict):
                step_model = PlanStep(
                    name=step.get("name", "unknown"),
                    description=step.get("description", ""),
                    agent=step.get("agent", "general"),
                    requires_approval=step.get("requires_approval", False),
                    dependencies=step.get("dependencies", []),
                    estimated_duration=step.get("estimated_duration", "unknown"),
                    input_data=step.get("input", {}),
                    risk_level=step.get("risk_level", "low")
                )
            else:
                step_model = step
            
            if step_model.requires_approval:
                # If high risk, save to DB for user review
                task_info = await self._create_approval_task(step_model)
                pending_approvals.append(task_info)
                logger.info("Task queued for approval", task_name=step_model.name)
            else:
                # If low risk, execute immediately via router
                start = time.time()
                result = await self.agent_router.execute(step if isinstance(step, dict) else step.dict())
                duration_ms = (time.time() - start) * 1000
                
                action_result = ActionResult(
                    step_name=step_model.name,
                    agent=step_model.agent,
                    success=result.get("success", False),
                    output=result.get("output"),
                    error=result.get("error"),
                    duration_ms=duration_ms
                )
                actions_taken.append(action_result)
                
                logger.agent_execution(
                    agent=step_model.agent,
                    task=step_model.name,
                    duration_ms=duration_ms,
                    success=result.get("success", False)
                )
        
        # Generate a final confirmation or status report for the user
        response = await self._generate_action_summary(
            user_input,
            [a.dict() for a in actions_taken],
            pending_approvals,
            context
        )
        
        return MasterAIResponse(
            type="action",
            response=response,
            actions_taken=actions_taken,
            pending_approvals=pending_approvals,
            metadata={
                "plan_steps": len(plan.get("steps", [])),
                "executed": len(actions_taken),
                "pending": len(pending_approvals)
            }
        )
    
    async def _generate_action_summary(
        self,
        user_input: str,
        actions: list,
        approvals: list,
        context: str
    ) -> str:
        """Uses LLM to summarize what was just done and what is pending."""
        prompt = f"""Summarize the actions taken and items pending approval in response to: "{user_input}"
        
Actions Completed: {json.dumps(actions, default=str)}
Pending Review: {json.dumps(approvals, default=str)}

Provide a concise confirmation message to the user. Be specific about what was done and what needs approval.
"""
        task_context = TaskContext(
            task_type="summary",
            risk_level="low",
            requires_accuracy=False,
            token_estimate=500,
            priority="normal"
        )
        
        return await self._call_llm(prompt, task_context=task_context)

    async def _handle_query(self, user_input: str, context: str) -> str:
        """Answers data queries by synthesizing context gathered from the database."""
        prompt = f"""{SYSTEM_PROMPT}

SYSTEM STATE & FINANCIALS:
{context}

USER DATA QUERY:
{user_input}

Provide a data-driven response. Use specific numbers (revenue, costs, status) found in the context.
If the data is not available, say so clearly.
"""
        task_context = TaskContext(
            task_type="query",
            risk_level="low",
            requires_accuracy=True,
            token_estimate=1000,
            priority="normal"
        )
        
        return await self._call_llm(prompt, task_context=task_context)

    @trace_llm
    async def _call_llm(
        self,
        prompt: str,
        system: str = None,
        task_context: TaskContext = None
    ) -> str:
        """
        Central LLM calling method with routing, retry, and metrics.
        All LLM calls should go through this method.
        """
        start_time = time.time()
        
        # Check token budget
        if self._total_tokens_today >= self._token_budget_daily:
            logger.warning("Daily token budget exhausted")
            raise TransientError("Daily token budget exhausted. Try again tomorrow.")
        
        try:
            response = await self.llm_router.complete(
                prompt=prompt,
                system=system,
                context=task_context
            )
            
            # Estimate tokens (rough: 4 chars per token)
            estimated_tokens = (len(prompt) + len(response)) // 4
            self._total_tokens_today += estimated_tokens
            
            duration_ms = (time.time() - start_time) * 1000
            monitor.timing("llm.call_duration", duration_ms)
            monitor.increment("llm.calls_success")
            
            logger.llm_call(
                provider="router",
                model=settings.ollama_model,
                prompt_tokens=len(prompt) // 4,
                completion_tokens=len(response) // 4,
                latency_ms=duration_ms,
                success=True
            )
            
            return response
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            monitor.increment("llm.calls_error")
            
            logger.llm_call(
                provider="router",
                model=settings.ollama_model,
                prompt_tokens=len(prompt) // 4,
                completion_tokens=0,
                latency_ms=duration_ms,
                success=False,
                error=str(e)
            )
            
            raise
    
    def _parse_json_response(self, response: str) -> dict:
        """Parse JSON from LLM response, handling common formatting issues."""
        # Clean up markdown code blocks
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            response = response.split("```")[1].split("```")[0].strip()
        
        return json.loads(response)
    
    async def run_autonomous_loop(self):
        """
        Background optimization loop.
        Runs every 6 hours to analyze business units and propose evolutions.
        """
        logger.info("Starting autonomous loop")
        
        while self.autonomous_mode:
            try:
                context = await self.context.build_context()
                
                # Step 1: Self-improvement analysis
                await self._consider_evolution(context)
                
                # Step 2: Business unit health check
                await self._check_business_health(context)
                
                logger.info("Autonomous loop iteration complete")
                
            except Exception as e:
                logger.error("Autonomous loop error", exc_info=True)
            
            # Wait for the next optimization cycle (6 hours)
            await asyncio.sleep(6 * 60 * 60)
    
    async def _check_business_health(self, context: str):
        """Analyze business unit performance and suggest optimizations."""
        # TODO: Implement in Part 3
        pass
    
    async def _consider_evolution(self, context: str):
        """
        Calls the EvolutionEngine to see if any system improvements are needed.
        Rate limited to prevent runaway self-modifications.
        """
        if not getattr(settings, 'enable_self_modification', True):
            return
        
        # Periodic reset of the hourly evolution counter
        now = datetime.now()
        if (now - self._hour_start).seconds >= 3600:
            self._evolution_count_this_hour = 0
            self._hour_start = now
        
        max_evolutions = getattr(settings, "max_evolutions_per_hour", 5)
        if not isinstance(max_evolutions, int):
            max_evolutions = 5

        # Enforce rate limit from settings
        if self._evolution_count_this_hour >= max_evolutions:
            logger.debug("Evolution rate limit reached")
            return
        
        # Build context for evolution proposal
        evolution_context = {
            "business_state": context,
            "recent_actions": [],
            "performance_metrics": {},
            "risk_profile": getattr(settings, 'risk_profile', 'moderate')
        }
        
        # Ask LLM for a proposal
        proposal = await self.evolution.propose_improvement(
            context=str(evolution_context),
            goal="Improve system performance and efficiency"
        )
        
        # Handle both EvolutionProposal object and legacy dict format
        from src.master_ai.evolution_models import EvolutionProposal, ProposalStatus
        
        if proposal:
            self._evolution_count_this_hour += 1
            
            # Check if it's the new EvolutionProposal type or legacy dict
            if isinstance(proposal, EvolutionProposal):
                is_beneficial = proposal.status != ProposalStatus.REJECTED
                confidence = proposal.confidence_score.overall if proposal.confidence_score else 0.0
                proposal_data = {
                    "id": proposal.id,
                    "type": proposal.proposal_type.value,
                    "description": proposal.description,
                    "changes": {c.file_path: c.new_content for c in proposal.changes if c.new_content},
                    "confidence": confidence,
                    "is_beneficial": is_beneficial
                }
            else:
                # Legacy dict format
                is_beneficial = proposal.get("is_beneficial", False)
                confidence = proposal.get("confidence", 0)
                proposal_data = proposal
            
            if not is_beneficial:
                logger.info("Evolution proposal not beneficial, skipping")
                return
            
            # Validate confidence threshold
            evolution_threshold = getattr(settings, "evolution_confidence_threshold", 0.8)
            if not isinstance(evolution_threshold, (int, float)):
                evolution_threshold = 0.8
            if confidence < evolution_threshold:
                logger.info(
                    "Evolution proposal rejected due to low confidence",
                    confidence=confidence,
                    threshold=evolution_threshold
                )
                return
            
            # Process in sandbox
            await self._process_evolution_proposal(proposal_data)
    
    async def _process_evolution_proposal(self, proposal: dict):
        """Process a validated evolution proposal through sandbox testing."""
        from src.utils.sandbox import Sandbox
        
        changes = proposal.get("changes", {})
        if not changes:
            return
        
        logger.info(
            "Processing evolution proposal",
            type=proposal.get("type"),
            files=list(changes.keys())
        )
        
        # Create sandbox and test
        sandbox = Sandbox()
        sandbox.create_sandbox(list(changes.keys()))
        
        try:
            all_patches_applied = True
            for file_path, new_code in changes.items():
                if not sandbox.apply_patch(file_path, new_code):
                    all_patches_applied = False
                    break
            
            if all_patches_applied:
                test_result = sandbox.run_tests()
                if test_result["success"]:
                    proposal["confidence"] = min(0.95, proposal.get("confidence", 0) + 0.1)
                    await self._save_evolution_proposal(proposal)
                    logger.info("Evolution proposal saved for approval")
                else:
                    logger.warning(
                        "Evolution proposal failed sandbox tests",
                        output=test_result.get("output", "")[:500]
                    )
        finally:
            sandbox.cleanup()
    
    async def _create_approval_task(self, step: PlanStep) -> dict:
        """Persists a high-risk task that requires human intervention."""
        async with get_db() as db:
            task = Task(
                id=str(uuid4()),
                name=step.name,
                description=step.description,
                type=step.agent,
                status="pending_approval",
                agent=step.agent,
                input_data=step.input_data,
                requires_approval=True
            )
            db.add(task)
            await db.commit()
            
            logger.info("Created approval task", task_id=task.id, name=task.name)
            
            return {
                "id": task.id,
                "name": task.name,
                "description": task.description,
                "agent": task.agent
            }
    
    async def _save_evolution_proposal(self, proposal: dict):
        """Saves a self-modification proposal for user approval."""
        async with get_db() as db:
            prop = EvolutionProposal(
                id=str(uuid4()),
                type=proposal["type"],
                description=proposal["description"],
                rationale=proposal["rationale"],
                proposed_changes=proposal["changes"],
                expected_impact=proposal.get("expected_impact"),
                confidence_score=proposal.get("confidence", 0.0)
            )
            db.add(prop)
            await db.commit()
            
            logger.info(
                "Saved evolution proposal",
                proposal_id=prop.id,
                type=prop.type
            )
