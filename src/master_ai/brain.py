"""
Master AI Brain - The central orchestrator for the autonomous empire.

This is the ONLY component that directly interfaces with the LLM (via OllamaClient).
All strategic decisions, operational planning, and agent delegation flow through this class.
It serves as the "CEO" of the business empire.
"""

import json
import asyncio
from datetime import datetime
from typing import Literal, Any
from uuid import uuid4

# Internal project imports
from src.master_ai.context import ContextManager
from src.master_ai.planner import Planner
from src.master_ai.evolution import EvolutionEngine
from src.master_ai.prompts import SYSTEM_PROMPT
from src.agents.router import AgentRouter
from src.database.connection import get_db
from src.database.models import ConversationMessage, Task, EvolutionProposal
from src.utils.ollama_client import OllamaClient
from src.utils.gemini_client import GeminiClient
from config.settings import settings
import os

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
        # LLM clients 
        self.ollama = OllamaClient(
            base_url=settings.ollama_url,
            model=settings.ollama_model
        )
        self.gemini = None
        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key:
            self.gemini = GeminiClient(api_key=gemini_key)
        
        # Specialized manager components
        self.context = ContextManager()      # Handles RAG and context window building
        self.planner = Planner(self.ollama)  # Note: Planner might still use ollama directly; consider refactoring too
        self.evolution = EvolutionEngine(self.ollama) 
        self.agent_router = AgentRouter()    # Dispatches tasks to sub-agents
        
        # State and rate limiting for autonomous features
        self.autonomous_mode = False
        self._evolution_count_this_hour = 0
        self._hour_start = datetime.now()
    
    async def process_input(self, user_input: str) -> dict:
        """
        Primary entry point for any user interaction (API or CLI).
        
        Logic:
        1. Build the current global context (financials, active tasks, history).
        2. Use the LLM to classify the user's intent.
        3. Dispatch to the appropriate handler (conversation, command, or query).
        """
        # 1. Gather all relevant data for the LLM to make informed decisions
        context = await self.context.build_context()
        
        # 2. Use a specialized classification prompt to determine the 'route'
        intent = await self._classify_intent(user_input, context)
        
        # Route 1: General Chat / Discussion
        if intent["type"] == "conversation":
            response = await self._handle_conversation(user_input, context)
            return {"type": "conversation", "response": response, "actions_taken": [], "pending_approvals": []}
        
        # Route 2: Direct Command (e.g., "Start a dropshipping business")
        elif intent["type"] == "command":
            return await self._handle_command(user_input, intent, context)
        
        # Route 3: Information Query (e.g., "How much profit did we make today?")
        elif intent["type"] == "query":
            response = await self._handle_query(user_input, context)
            return {"type": "conversation", "response": response, "actions_taken": [], "pending_approvals": []}
        
        return {"type": "conversation", "response": "Error: Unknown intent type", "actions_taken": [], "pending_approvals": []}

    async def _classify_intent(self, user_input: str, context: str) -> dict:
        """
        Uses the LLM as a router to interpret natural language into a structured intent.
        
        Returns a dict with 'type', 'action', and 'parameters'.
        """
        # Inline prompt for classification (usually shorter and more focused than system prompt)
        prompt = f"""Given this user input, classify the intent.

User input: "{user_input}"

Respond with JSON only:
{{
    "type": "conversation" | "command" | "query",
    "action": null or action name (e.g., "start_business", "stop_business", "analyze", "optimize"),
    "parameters": {{extracted parameters}}
}}

Definitions:
- "conversation": General interaction, greetings, or philosophical discussion.
- "command": Request to change the state of the empire (start/stop/modify).
- "query": Request for data, reports, or status updates.
"""
        
        response = await self._call_llm(prompt)
        try:
            # Clean LLM response if it contains markdown markers
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()
            return json.loads(response)
        except Exception:
             # Default to conversation if JSON parsing fails
             return {"type": "conversation", "action": None, "parameters": {}}
    
    async def _handle_conversation(self, user_input: str, context: str) -> str:
        """Handles chit-chat while maintaining awareness of the empire's state."""
        prompt = f"""{SYSTEM_PROMPT}

CURRENT CONTEXT OF THE EMPIRE:
{context}

USER MESSAGE:
{user_input}

Respond naturally as King AI, the autonomous CEO. Be professional, data-aware, and concise.
"""
        return await self._call_llm(prompt)
    
    async def _handle_command(self, user_input: str, intent: dict, context: str) -> dict:
        """
        Orchestrates complex actions:
        1. Breaks goal into steps via the Planner.
        2. Checks for approval requirements based on Risk Profile.
        3. Executes immediate tasks or queues for user review.
        """
        # Create a detailed multi-step plan
        plan = await self.planner.create_plan(
            goal=user_input,
            action=intent["action"],
            parameters=intent["parameters"],
            context=context
        )
        
        actions_taken = []
        pending_approvals = []
        
        # Iterate through planned steps
        for step in plan.get("steps", []):
            if step["requires_approval"]:
                # If high risk (financial/legal), save to DB for user review
                task = await self._create_approval_task(step)
                pending_approvals.append(task)
            else:
                # If low risk (research/analysis), execute immediately via router
                result = await self.agent_router.execute(step)
                actions_taken.append({
                    "step": step["name"],
                    "agent": step["agent"],
                    "result": result
                })
        
        # Generate a final confirmation or status report for the user
        response = await self._generate_action_summary(user_input, actions_taken, pending_approvals, context)
        
        return {
            "type": "action",
            "response": response,
            "actions_taken": actions_taken,
            "pending_approvals": pending_approvals
        }
    
    async def _generate_action_summary(self, user_input: str, actions: list, approvals: list, context: str) -> str:
        """Uses LLM to summarize what was just done and what is pending."""
        prompt = f"""Summarize the actions taken and items pending approval in response to: "{user_input}"
        
        Actions Completed: {json.dumps(actions)}
        Pending Review: {json.dumps(approvals)}
        
        Provide a concise confirmation message to the user.
        """
        return await self._call_llm(prompt)

    async def _handle_query(self, user_input: str, context: str) -> str:
        """Answers data queries by synthesizing context gathered from the database."""
        prompt = f"""{SYSTEM_PROMPT}

SYSTEM STATE & FINANCIALS:
{context}

USER DATA QUERY:
{user_input}

Provide a data-driven response. Use specific numbers (revenue, costs, status) found in the context.
"""
        return await self._call_llm(prompt)

    async def _call_llm(self, prompt: str, system: str | None = None) -> str:
        """
        Resilient LLM wrapper. Tries Ollama first, then falls back to Gemini.
        """
        try:
            # Try Ollama (EC2 or Local)
            return await self.ollama.complete(prompt, system)
        except Exception as e:
            # Fallback to Gemini if configured
            if self.gemini:
                try:
                    return await self.gemini.complete(prompt, system)
                except Exception as ge:
                    return f"Both AI providers failed. Ollama: {e}, Gemini: {ge}"
            else:
                return f"AI connection failed and no fallback configured: {e}"
    
    async def run_autonomous_loop(self):
        """
        Background optimization loop.
        Runs every 6 hours to analyze business units and propose evolutions.
        """
        while self.autonomous_mode:
            context = await self.context.build_context()
            
            # Step 1: Self-improvement analysis
            await self._consider_evolution(context)
            
            # TODO: Add deeper business unit performance analysis here
            
            # Wait for the next optimization cycle
            await asyncio.sleep(6 * 60 * 60)
    
    async def _consider_evolution(self, context: str):
        """
        Calls the EvolutionEngine to see if any system improvements are needed.
        Rate limited to prevent runaway self-modifications.
        """
        # Periodic reset of the hourly evolution counter
        now = datetime.now()
        if (now - self._hour_start).seconds >= 3600:
            self._evolution_count_this_hour = 0
            self._hour_start = now
        
        # Enforce rate limit from settings
        if self._evolution_count_this_hour >= settings.max_evolutions_per_hour:
            return
        
        # Ask LLM for a proposal
        proposal = await self.evolution.propose_improvement(context)
        
        if proposal and proposal["is_beneficial"]:
            self._evolution_count_this_hour += 1
            await self._save_evolution_proposal(proposal)
    
    async def _create_approval_task(self, step: dict) -> dict:
        """Persists a high-risk task that requires human intervention."""
        async with get_db() as db:
            task = Task(
                id=str(uuid4()),
                name=step["name"],
                description=step.get("description"),
                type=step.get("type", "general"),
                status="pending_approval",
                agent=step.get("agent"),
                input_data=step.get("input", {}),
                requires_approval=True
            )
            db.add(task)
            await db.commit()
            return {"id": task.id, "name": task.name, "description": task.description}
    
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
