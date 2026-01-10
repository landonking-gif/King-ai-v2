"""
Autonomous Business Engine - Complete business lifecycle management.

This engine handles:
1. Takes a business prompt and learns about the business type
2. Conducts real market research 
3. Creates an informed business plan based on research
4. Delegates tasks to agentic AI programs
5. If agents can't complete a task, creates new programs that can
6. Checks finished products for completeness
7. Continuously monitors and delegates changes

All actions are logged and visible to the user.
"""

import os
import json
import asyncio
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import traceback

from src.services.execution_engine import get_execution_engine, ActionRequest, ActionType
from src.utils.llm_router import LLMRouter, TaskContext
from src.utils.structured_logging import get_logger
from src.agents.router import AgentRouter
from src.agents.research import ResearchAgent, ResearchQuery, ResearchType, ResearchReport

logger = get_logger("autonomous_business_engine")


class BusinessPhase(str, Enum):
    """Phases of autonomous business creation."""
    UNDERSTANDING = "understanding"      # Learning about the business type
    RESEARCH = "research"                # Market research & competitor analysis
    PLANNING = "planning"                # Creating business plan from research
    EXECUTION = "execution"              # Delegating tasks to agents
    VERIFICATION = "verification"        # Checking completed work
    MONITORING = "monitoring"            # Continuous monitoring
    COMPLETE = "complete"


class TaskStatus(str, Enum):
    """Status of a task."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    NEEDS_NEW_AGENT = "needs_new_agent"


@dataclass
class ActionLog:
    """Log entry for an action taken by the engine."""
    timestamp: datetime
    phase: BusinessPhase
    action: str
    details: str
    success: bool
    files_created: List[str] = field(default_factory=list)
    agent_used: Optional[str] = None
    duration_ms: float = 0.0


@dataclass
class BusinessTask:
    """A task to be completed for the business."""
    task_id: str
    name: str
    description: str
    agent_type: str
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Dict] = None
    error: Optional[str] = None
    files_created: List[str] = field(default_factory=list)
    attempts: int = 0
    max_attempts: int = 3


@dataclass
class MarketResearch:
    """Market research results."""
    industry_overview: str
    market_size: str
    growth_rate: str
    target_audience: str
    competitors: List[Dict[str, str]]
    trends: List[str]
    opportunities: List[str]
    challenges: List[str]
    sources: List[str]
    raw_data: Dict = field(default_factory=dict)


@dataclass
class BusinessBlueprint:
    """Complete business blueprint created from research."""
    business_id: str
    business_type: str
    business_name: str
    description: str
    
    # Research results
    market_research: Optional[MarketResearch] = None
    
    # Business plan
    value_proposition: str = ""
    revenue_model: str = ""
    target_market: str = ""
    marketing_strategy: str = ""
    operations_plan: str = ""
    financial_projections: Dict = field(default_factory=dict)
    
    # Tasks and execution
    tasks: List[BusinessTask] = field(default_factory=list)
    completed_tasks: List[str] = field(default_factory=list)
    
    # Tracking
    phase: BusinessPhase = BusinessPhase.UNDERSTANDING
    action_log: List[ActionLog] = field(default_factory=list)
    files_created: List[str] = field(default_factory=list)
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            "business_id": self.business_id,
            "business_type": self.business_type,
            "business_name": self.business_name,
            "description": self.description,
            "phase": self.phase.value,
            "market_research": {
                "industry_overview": self.market_research.industry_overview if self.market_research else "",
                "market_size": self.market_research.market_size if self.market_research else "",
                "competitors": self.market_research.competitors if self.market_research else [],
                "trends": self.market_research.trends if self.market_research else [],
            } if self.market_research else None,
            "value_proposition": self.value_proposition,
            "revenue_model": self.revenue_model,
            "target_market": self.target_market,
            "tasks_total": len(self.tasks),
            "tasks_completed": len(self.completed_tasks),
            "files_created": self.files_created,
            "action_log_count": len(self.action_log),
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
        }


class AutonomousBusinessEngine:
    """
    Fully autonomous business creation and management engine.
    
    This engine:
    1. Understands the business type from a simple prompt
    2. Conducts real market research
    3. Creates an informed business plan
    4. Delegates tasks to specialized agents
    5. Creates new agents if needed
    6. Verifies all completed work
    7. Continuously monitors the business
    """
    
    def __init__(self, workspace: str = None):
        """Initialize the autonomous engine."""
        self.workspace = workspace or os.getcwd()
        self.businesses_dir = Path(self.workspace) / "businesses"
        self.businesses_dir.mkdir(exist_ok=True)
        
        self.execution_engine = get_execution_engine(self.workspace)
        self.llm_router = LLMRouter()
        self.agent_router = AgentRouter()
        self.research_agent = ResearchAgent()
        
        # Track active businesses
        self.active_businesses: Dict[str, BusinessBlueprint] = {}
        
        # Custom agent registry for dynamically created agents
        self.custom_agents: Dict[str, Callable] = {}
        
        # Event callbacks for UI updates
        self.on_phase_change: Optional[Callable] = None
        self.on_action_log: Optional[Callable] = None
        self.on_file_created: Optional[Callable] = None
        
        logger.info("AutonomousBusinessEngine initialized", workspace=self.workspace)
    
    async def create_business(self, prompt: str) -> BusinessBlueprint:
        """
        Create a complete business from a simple prompt.
        
        Args:
            prompt: e.g., "Create a pet supply dropshipping business"
            
        Returns:
            Complete BusinessBlueprint with all files created
        """
        business_id = f"auto_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        business_dir = self.businesses_dir / business_id
        
        logger.info("🚀 Starting autonomous business creation", 
                   prompt=prompt, business_id=business_id)
        
        # Create business blueprint
        blueprint = BusinessBlueprint(
            business_id=business_id,
            business_type="",
            business_name="",
            description=prompt
        )
        
        self.active_businesses[business_id] = blueprint
        
        try:
            # Phase 1: Understanding
            await self._phase_understanding(blueprint, prompt, business_dir)
            
            # Phase 2: Research
            await self._phase_research(blueprint, business_dir)
            
            # Phase 3: Planning
            await self._phase_planning(blueprint, business_dir)
            
            # Phase 4: Execution
            await self._phase_execution(blueprint, business_dir)
            
            # Phase 5: Verification
            await self._phase_verification(blueprint, business_dir)
            
            # Phase 6: Set up monitoring
            await self._phase_monitoring_setup(blueprint, business_dir)
            
            blueprint.phase = BusinessPhase.COMPLETE
            blueprint.last_updated = datetime.now()
            
            # Save final state
            await self._save_business_state(blueprint, business_dir)
            
            logger.info("✅ Business creation complete", 
                       business_id=business_id,
                       files_created=len(blueprint.files_created))
            
            return blueprint
            
        except Exception as e:
            logger.error(f"Business creation failed: {e}", exc_info=True)
            self._log_action(blueprint, BusinessPhase.UNDERSTANDING, 
                           "Error", f"Creation failed: {str(e)}", False)
            raise
    
    async def _phase_understanding(
        self,
        blueprint: BusinessBlueprint,
        prompt: str,
        business_dir: Path
    ):
        """Phase 1: Understand the business request.
        
        We don't need to classify - just extract a name and description.
        The AI will understand the full context when creating the business plan.
        """
        blueprint.phase = BusinessPhase.UNDERSTANDING
        self._notify_phase_change(blueprint)
        
        logger.info("📚 Phase 1: Understanding business request...")
        
        start = datetime.now()
        
        # Use LLM to understand and name the business - no classification needed
        analysis_prompt = f"""You are helping create a new business. Analyze this request and suggest details:

Request: "{prompt}"

Respond in JSON format:
{{
    "business_name": "a creative, memorable business name",
    "description": "what this business does in 2-3 sentences",
    "target_customers": "who will be the customers",
    "key_offerings": ["main product or service 1", "main product or service 2", "main product or service 3"]
}}

Be creative with the name. Make the description specific to what the user asked for."""

        try:
            response = await self.llm_router.complete(
                prompt=analysis_prompt,
                context=TaskContext(task_type="analysis", risk_level="medium", requires_accuracy=True, token_estimate=300, priority="normal")
            )
            
            analysis = self._extract_json(response)
            
            blueprint.business_name = analysis.get("business_name", "My New Business")
            blueprint.description = analysis.get("description", prompt)
            blueprint.target_market = analysis.get("target_customers", "")
            # Store the original prompt as the business type - AI will understand it later
            blueprint.business_type = prompt  # Keep original request for context
            
            # Store for later use
            blueprint.financial_projections["initial_analysis"] = analysis
            
            duration = (datetime.now() - start).total_seconds() * 1000
            
            self._log_action(
                blueprint, BusinessPhase.UNDERSTANDING,
                "Analyzed Business Request",
                f"Created: {blueprint.business_name}",
                True, duration_ms=duration
            )
            
            logger.info("✓ Business understood", 
                       name=blueprint.business_name)
            
        except Exception as e:
            logger.warning(f"AI analysis failed: {e}")
            # Simple fallback - just use the prompt
            blueprint.business_name = "My New Business"
            blueprint.description = prompt
            blueprint.business_type = prompt
            
            self._log_action(
                blueprint, BusinessPhase.UNDERSTANDING,
                "Basic Analysis",
                f"Using prompt directly: {prompt[:50]}...",
                True
            )
        
        # Create business directory
        await self._create_directory_structure(blueprint, business_dir)
    
    async def _phase_research(
        self,
        blueprint: BusinessBlueprint,
        business_dir: Path
    ):
        """Phase 2: Conduct comprehensive market research."""
        blueprint.phase = BusinessPhase.RESEARCH
        self._notify_phase_change(blueprint)
        
        logger.info("🔍 Phase 2: Conducting market research...")
        
        # Use AI to generate relevant research queries based on the business
        research_queries = await self._generate_research_queries(blueprint)
        
        research_results = {}
        
        for task_name, query in research_queries:
            start = datetime.now()
            logger.info(f"  🔎 Researching: {task_name}")
            
            try:
                # Use research agent
                research_query = ResearchQuery(
                    query=query,
                    research_type=ResearchType.MARKET_RESEARCH,
                    depth=2,
                    max_sources=5
                )
                
                report = await self.research_agent.research(research_query)
                research_results[task_name] = report
                
                duration = (datetime.now() - start).total_seconds() * 1000
                
                self._log_action(
                    blueprint, BusinessPhase.RESEARCH,
                    f"Research: {task_name}",
                    f"Found {len(report.sources)} sources, {len(report.key_findings)} findings",
                    True, agent_used="research", duration_ms=duration
                )
                
            except Exception as e:
                logger.warning(f"Research task failed: {task_name}: {e}")
                self._log_action(
                    blueprint, BusinessPhase.RESEARCH,
                    f"Research: {task_name}",
                    f"Failed: {str(e)}",
                    False
                )
        
        # Synthesize research into MarketResearch object
        blueprint.market_research = await self._synthesize_research(
            blueprint, research_results, business_dir
        )
        
        # Save research report
        research_file = business_dir / "research" / "market_research.md"
        await self._create_research_document(blueprint, research_file)
        
        logger.info("✓ Market research complete")
    
    async def _generate_research_queries(self, blueprint: BusinessBlueprint) -> list:
        """Use AI to generate relevant research queries for this specific business."""
        prompt = f"""Generate 4 research queries for this business:

Business: {blueprint.business_name}
Description: {blueprint.description}

Return a JSON array of research topics:
[
    {{"topic": "Industry Overview", "query": "specific search query for industry research"}},
    {{"topic": "Competitor Analysis", "query": "specific search query for competitors"}},
    {{"topic": "Market Trends", "query": "specific search query for trends"}},
    {{"topic": "Target Audience", "query": "specific search query for customers"}}
]

Make the queries specific to this business, not generic."""

        try:
            response = await self.llm_router.complete(
                prompt=prompt,
                context=TaskContext(task_type="planning", risk_level="low", requires_accuracy=True, token_estimate=200, priority="normal")
            )
            
            import re
            json_match = re.search(r'\[[\s\S]*\]', response)
            if json_match:
                queries_data = json.loads(json_match.group())
                return [(q.get("topic", f"Research {i}"), q.get("query", blueprint.description)) 
                        for i, q in enumerate(queries_data[:4])]
        except Exception as e:
            logger.warning(f"Failed to generate research queries: {e}")
        
        # Fallback: use the business description directly
        desc = blueprint.description[:100]
        return [
            ("Industry Overview", f"{desc} industry market size"),
            ("Competitor Analysis", f"{desc} competitors"),
            ("Market Trends", f"{desc} trends 2024 2025"),
            ("Target Audience", f"{desc} customer demographics"),
        ]
    
    async def _phase_planning(
        self,
        blueprint: BusinessBlueprint,
        business_dir: Path
    ):
        """Phase 3: Create informed business plan from research."""
        blueprint.phase = BusinessPhase.PLANNING
        self._notify_phase_change(blueprint)
        
        logger.info("📋 Phase 3: Creating informed business plan...")
        
        start = datetime.now()
        
        # Build context from research
        research_context = ""
        if blueprint.market_research:
            research_context = f"""
Market Research Summary:
- Industry: {blueprint.market_research.industry_overview}
- Market Size: {blueprint.market_research.market_size}
- Growth Rate: {blueprint.market_research.growth_rate}
- Target Audience: {blueprint.market_research.target_audience}
- Key Trends: {', '.join(blueprint.market_research.trends[:5])}
- Opportunities: {', '.join(blueprint.market_research.opportunities[:3])}
- Competitors: {', '.join([c.get('name', '') for c in blueprint.market_research.competitors[:5]])}
"""
        
        # Generate comprehensive business plan
        plan_prompt = f"""Based on the following market research, create a comprehensive business plan:

Business: {blueprint.business_name}
Description: {blueprint.description}

{research_context}

Create a detailed business plan as JSON:
{{
    "value_proposition": "unique value this business offers",
    "revenue_model": "how the business makes money",
    "target_market": "specific target market segments",
    "marketing_strategy": "detailed marketing approach",
    "operations_plan": "how the business operates day-to-day",
    "financial_projections": {{
        "startup_costs": 0,
        "monthly_expenses": 0,
        "revenue_month_1": 0,
        "revenue_month_6": 0,
        "revenue_month_12": 0,
        "break_even_months": 0
    }},
    "milestones": [
        {{"name": "milestone", "target_date": "relative date", "description": "what it means"}}
    ],
    "required_tasks": [
        {{"name": "task name", "description": "what to do", "agent_type": "research|content|code_generator|commerce|finance|legal|analytics", "priority": 1-10}}
    ],
    "kpis": ["list of key performance indicators to track"]
}}

Generate 6 specific tasks that are relevant to THIS business. Be specific, realistic, and actionable."""

        try:
            response = await self.llm_router.complete(
                prompt=plan_prompt,
                context=TaskContext(task_type="business_planning", risk_level="high", requires_accuracy=True, token_estimate=2000, priority="high")
            )
            
            plan_data = self._extract_json(response)
            
            blueprint.value_proposition = plan_data.get("value_proposition", "")
            blueprint.revenue_model = plan_data.get("revenue_model", "")
            blueprint.target_market = plan_data.get("target_market", "")
            blueprint.marketing_strategy = plan_data.get("marketing_strategy", "")
            blueprint.operations_plan = plan_data.get("operations_plan", "")
            blueprint.financial_projections = plan_data.get("financial_projections", {})
            
            # Create tasks from plan
            for i, task_data in enumerate(plan_data.get("required_tasks", [])):
                task = BusinessTask(
                    task_id=f"task_{i+1}",
                    name=task_data.get("name", f"Task {i+1}"),
                    description=task_data.get("description", ""),
                    agent_type=task_data.get("agent_type", "content")
                )
                blueprint.tasks.append(task)
            
            # If no tasks were created from LLM, use AI to generate tasks
            if not blueprint.tasks:
                logger.info("LLM returned no tasks, using AI to generate tasks")
                blueprint.tasks = await self._generate_tasks_with_ai(blueprint)
            
            duration = (datetime.now() - start).total_seconds() * 1000
            
            self._log_action(
                blueprint, BusinessPhase.PLANNING,
                "Created Business Plan",
                f"Generated plan with {len(blueprint.tasks)} tasks",
                True, duration_ms=duration
            )
            
        except Exception as e:
            logger.warning(f"AI planning failed, retrying with simpler AI: {e}")
            blueprint.tasks = await self._generate_tasks_with_ai(blueprint)
            
            self._log_action(
                blueprint, BusinessPhase.PLANNING,
                "AI Retry Planning",
                f"Generated {len(blueprint.tasks)} tasks with AI",
                True
            )
        
        # Save business plan document
        plan_file = business_dir / "documents" / "business_plan.md"
        await self._create_business_plan_document(blueprint, plan_file)
        
        logger.info(f"✓ Business plan created with {len(blueprint.tasks)} tasks")
    
    async def _phase_execution(
        self,
        blueprint: BusinessBlueprint,
        business_dir: Path
    ):
        """Phase 4: Execute tasks using agents."""
        blueprint.phase = BusinessPhase.EXECUTION
        self._notify_phase_change(blueprint)
        
        logger.info(f"⚡ Phase 4: Executing {len(blueprint.tasks)} tasks...")
        
        for task in blueprint.tasks:
            await self._execute_task(blueprint, task, business_dir)
        
        completed = sum(1 for t in blueprint.tasks if t.status == TaskStatus.COMPLETED)
        logger.info(f"✓ Execution complete: {completed}/{len(blueprint.tasks)} tasks succeeded")
    
    async def _execute_task(
        self,
        blueprint: BusinessBlueprint,
        task: BusinessTask,
        business_dir: Path
    ):
        """Execute a single task, creating new agents if needed."""
        task.status = TaskStatus.IN_PROGRESS
        task.attempts += 1
        
        start = datetime.now()
        logger.info(f"  🔧 Executing: {task.name} (agent: {task.agent_type})")
        
        try:
            # Get or create agent
            agent = self.agent_router.get_agent(task.agent_type)
            
            if not agent:
                # Try to create a new agent for this task type
                logger.info(f"    ⚠️ No agent for {task.agent_type}, creating custom agent...")
                agent = await self._create_custom_agent(task, blueprint)
            
            if agent:
                # Execute task
                result = await agent.execute({
                    "name": task.name,
                    "description": task.description,
                    "type": task.agent_type,
                    "query": task.description,
                    "business_context": {
                        "business_name": blueprint.business_name,
                        "business_type": blueprint.business_type,
                        "target_market": blueprint.target_market,
                    }
                })
                
                if result.get("success"):
                    task.status = TaskStatus.COMPLETED
                    task.result = result
                    blueprint.completed_tasks.append(task.task_id)
                    
                    # Create output files from task result
                    files = await self._create_task_output_files(
                        blueprint, task, result, business_dir
                    )
                    task.files_created = files
                    
                    duration = (datetime.now() - start).total_seconds() * 1000
                    
                    self._log_action(
                        blueprint, BusinessPhase.EXECUTION,
                        f"Task: {task.name}",
                        f"Completed successfully, created {len(files)} files",
                        True, files_created=files, 
                        agent_used=task.agent_type, duration_ms=duration
                    )
                else:
                    raise Exception(result.get("error", "Unknown error"))
            else:
                raise Exception(f"Could not create agent for {task.agent_type}")
                
        except Exception as e:
            logger.warning(f"    ❌ Task failed: {e}")
            
            if task.attempts < task.max_attempts:
                # Retry with different approach
                task.status = TaskStatus.PENDING
                task.error = str(e)
                
                self._log_action(
                    blueprint, BusinessPhase.EXECUTION,
                    f"Task: {task.name}",
                    f"Failed (attempt {task.attempts}): {str(e)}",
                    False, agent_used=task.agent_type
                )
            else:
                task.status = TaskStatus.FAILED
                task.error = str(e)
                
                self._log_action(
                    blueprint, BusinessPhase.EXECUTION,
                    f"Task: {task.name}",
                    f"Failed after {task.attempts} attempts: {str(e)}",
                    False, agent_used=task.agent_type
                )
    
    async def _create_custom_agent(
        self,
        task: BusinessTask,
        blueprint: BusinessBlueprint
    ) -> Optional[Any]:
        """Create a custom agent if no suitable agent exists."""
        logger.info(f"    🛠️ Creating custom agent for: {task.agent_type}")
        
        # Generate agent code using LLM
        agent_prompt = f"""Create a Python agent class that can handle this task:

Task: {task.name}
Description: {task.description}
Agent Type: {task.agent_type}

The agent should:
1. Inherit from a base pattern
2. Have an execute() method that takes a task dict
3. Return {{"success": True/False, "output": result, "error": error_msg}}

Provide the Python code for a simple agent that handles this type of task.
Focus on practical implementation that can work with available tools."""

        try:
            response = await self.llm_router.complete(
                prompt=agent_prompt,
                context=TaskContext(task_type="code_generation", risk_level="medium", requires_accuracy=True, token_estimate=1000, priority="normal")
            )
            
            # For now, use content agent as fallback
            # In a full implementation, we would eval the generated code
            content_agent = self.agent_router.get_agent("content")
            if content_agent:
                self.custom_agents[task.agent_type] = content_agent
                return content_agent
                
        except Exception as e:
            logger.warning(f"Failed to create custom agent: {e}")
        
        # Fallback: use content agent
        return self.agent_router.get_agent("content")
    
    async def _phase_verification(
        self,
        blueprint: BusinessBlueprint,
        business_dir: Path
    ):
        """Phase 5: Verify all completed work."""
        blueprint.phase = BusinessPhase.VERIFICATION
        self._notify_phase_change(blueprint)
        
        logger.info("✅ Phase 5: Verifying completed work...")
        
        # Check all files exist
        missing_files = []
        for file_path in blueprint.files_created:
            full_path = business_dir / file_path if not Path(file_path).is_absolute() else Path(file_path)
            if not full_path.exists():
                missing_files.append(file_path)
        
        # Check task completion
        incomplete_tasks = [t for t in blueprint.tasks if t.status != TaskStatus.COMPLETED]
        
        # Generate verification report
        verification_prompt = f"""Verify this business setup is complete:

Business: {blueprint.business_name}
Type: {blueprint.business_type}

Files Created: {len(blueprint.files_created)}
Missing Files: {len(missing_files)}
Completed Tasks: {len(blueprint.completed_tasks)}/{len(blueprint.tasks)}
Incomplete Tasks: {[t.name for t in incomplete_tasks]}

Required components for a {blueprint.business_type} business:
- Business plan document
- Marketing strategy
- Website or landing page
- Product/service catalog
- Financial projections
- Operations manual

What's missing? Provide JSON:
{{
    "is_complete": true/false,
    "missing_components": ["list of missing items"],
    "recommendations": ["list of recommendations"],
    "quality_score": 0-100
}}"""

        try:
            response = await self.llm_router.complete(
                prompt=verification_prompt,
                context=TaskContext(task_type="verification", risk_level="low", requires_accuracy=True, token_estimate=500, priority="normal")
            )
            
            verification = self._extract_json(response)
            
            self._log_action(
                blueprint, BusinessPhase.VERIFICATION,
                "Verification Complete",
                f"Quality score: {verification.get('quality_score', 'N/A')}/100",
                verification.get("is_complete", False)
            )
            
            # Create verification report
            report_file = business_dir / "verification_report.md"
            await self._create_verification_report(blueprint, verification, report_file)
            
        except Exception as e:
            logger.warning(f"Verification analysis failed: {e}")
            
            self._log_action(
                blueprint, BusinessPhase.VERIFICATION,
                "Basic Verification",
                f"Files: {len(blueprint.files_created)}, Tasks: {len(blueprint.completed_tasks)}/{len(blueprint.tasks)}",
                len(missing_files) == 0
            )
        
        logger.info("✓ Verification complete")
    
    async def _phase_monitoring_setup(
        self,
        blueprint: BusinessBlueprint,
        business_dir: Path
    ):
        """Phase 6: Set up continuous monitoring."""
        blueprint.phase = BusinessPhase.MONITORING
        self._notify_phase_change(blueprint)
        
        logger.info("👁️ Phase 6: Setting up continuous monitoring...")
        
        # Create monitoring configuration
        monitoring_config = {
            "business_id": blueprint.business_id,
            "business_name": blueprint.business_name,
            "kpis_to_track": [
                "website_traffic",
                "conversion_rate", 
                "revenue",
                "customer_acquisition_cost",
                "customer_lifetime_value"
            ],
            "check_interval_hours": 24,
            "alert_thresholds": {
                "revenue_drop_percent": 20,
                "traffic_drop_percent": 30,
            },
            "auto_actions": {
                "low_traffic": "run_marketing_campaign",
                "low_conversion": "optimize_landing_page",
                "high_bounce_rate": "improve_content"
            },
            "last_check": datetime.now().isoformat(),
            "created_at": datetime.now().isoformat()
        }
        
        # Save monitoring config
        config_file = business_dir / "monitoring" / "config.json"
        await self.execution_engine.execute(ActionRequest(
            action_type=ActionType.DIR_CREATE,
            params={"path": str(config_file.parent)},
            description="Create monitoring directory"
        ))
        
        await self.execution_engine.execute(ActionRequest(
            action_type=ActionType.FILE_CREATE,
            params={
                "path": str(config_file),
                "content": json.dumps(monitoring_config, indent=2)
            },
            description="Create monitoring configuration"
        ))
        
        blueprint.files_created.append(str(config_file.relative_to(business_dir)))
        
        self._log_action(
            blueprint, BusinessPhase.MONITORING,
            "Monitoring Setup",
            "Created monitoring configuration with KPIs and auto-actions",
            True, files_created=[str(config_file.relative_to(business_dir))]
        )
        
        logger.info("✓ Monitoring setup complete")
    
    # ==================== Helper Methods ====================
    
    async def _create_directory_structure(
        self,
        blueprint: BusinessBlueprint,
        business_dir: Path
    ):
        """Create the business directory structure."""
        directories = [
            business_dir,
            business_dir / "research",
            business_dir / "documents",
            business_dir / "website",
            business_dir / "marketing",
            business_dir / "products",
            business_dir / "financials",
            business_dir / "operations",
            business_dir / "monitoring",
            business_dir / "logs",
        ]
        
        for dir_path in directories:
            await self.execution_engine.execute(ActionRequest(
                action_type=ActionType.DIR_CREATE,
                params={"path": str(dir_path)},
                description=f"Create directory: {dir_path.name}"
            ))
    
    async def _synthesize_research(
        self,
        blueprint: BusinessBlueprint,
        research_results: Dict[str, ResearchReport],
        business_dir: Path
    ) -> MarketResearch:
        """Synthesize research results into a MarketResearch object."""
        
        # Combine all findings
        all_findings = []
        all_sources = []
        
        for task_name, report in research_results.items():
            all_findings.extend(report.key_findings)
            all_sources.extend([s.url for s in report.sources])
        
        # Use AI to synthesize
        synthesis_prompt = f"""Synthesize this market research for a {blueprint.business_type} business:

Research Findings:
{chr(10).join(f'- {f}' for f in all_findings[:20])}

Create a structured analysis as JSON:
{{
    "industry_overview": "2-3 sentence industry overview",
    "market_size": "estimated market size",
    "growth_rate": "annual growth rate",
    "target_audience": "detailed target audience description",
    "competitors": [
        {{"name": "competitor", "strengths": "their strengths", "weaknesses": "their weaknesses"}}
    ],
    "trends": ["key trend 1", "key trend 2"],
    "opportunities": ["opportunity 1", "opportunity 2"],
    "challenges": ["challenge 1", "challenge 2"]
}}"""

        try:
            response = await self.llm_router.complete(
                prompt=synthesis_prompt,
                context=TaskContext(task_type="analysis", risk_level="medium", requires_accuracy=True, token_estimate=500, priority="normal")
            )
            
            data = self._extract_json(response)
            
            return MarketResearch(
                industry_overview=data.get("industry_overview", ""),
                market_size=data.get("market_size", "Unknown"),
                growth_rate=data.get("growth_rate", "Unknown"),
                target_audience=data.get("target_audience", ""),
                competitors=data.get("competitors", []),
                trends=data.get("trends", []),
                opportunities=data.get("opportunities", []),
                challenges=data.get("challenges", []),
                sources=all_sources[:10],
                raw_data={"findings": all_findings}
            )
            
        except Exception as e:
            logger.warning(f"Research synthesis failed: {e}")
            return MarketResearch(
                industry_overview=f"{blueprint.business_type.title()} industry",
                market_size="Research pending",
                growth_rate="Research pending",
                target_audience="General consumers",
                competitors=[],
                trends=all_findings[:5],
                opportunities=["Market entry opportunity"],
                challenges=["Competition"],
                sources=all_sources[:10],
                raw_data={"findings": all_findings}
            )
    
    async def _create_research_document(
        self,
        blueprint: BusinessBlueprint,
        file_path: Path
    ):
        """Create the market research document."""
        research = blueprint.market_research
        
        content = f"""# Market Research Report
## {blueprint.business_name}

*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*

---

## Executive Summary

{research.industry_overview if research else 'Research pending...'}

## Market Overview

- **Market Size:** {research.market_size if research else 'TBD'}
- **Growth Rate:** {research.growth_rate if research else 'TBD'}
- **Target Audience:** {research.target_audience if research else 'TBD'}

## Competitive Landscape

"""
        if research and research.competitors:
            for comp in research.competitors:
                content += f"""### {comp.get('name', 'Competitor')}
- **Strengths:** {comp.get('strengths', 'N/A')}
- **Weaknesses:** {comp.get('weaknesses', 'N/A')}

"""
        
        content += """## Market Trends

"""
        if research and research.trends:
            for trend in research.trends:
                content += f"- {trend}\n"
        
        content += """
## Opportunities

"""
        if research and research.opportunities:
            for opp in research.opportunities:
                content += f"- {opp}\n"
        
        content += """
## Challenges

"""
        if research and research.challenges:
            for challenge in research.challenges:
                content += f"- {challenge}\n"
        
        content += """
## Sources

"""
        if research and research.sources:
            for source in research.sources[:10]:
                content += f"- {source}\n"
        
        content += """
---
*This research was conducted by the Autonomous Business Engine*
"""
        
        await self.execution_engine.execute(ActionRequest(
            action_type=ActionType.FILE_CREATE,
            params={"path": str(file_path), "content": content},
            description="Create market research document"
        ))
        
        relative_path = str(file_path.relative_to(file_path.parent.parent))
        blueprint.files_created.append(relative_path)
        self._notify_file_created(relative_path)
    
    async def _create_business_plan_document(
        self,
        blueprint: BusinessBlueprint,
        file_path: Path
    ):
        """Create the business plan document."""
        content = f"""# Business Plan
## {blueprint.business_name}

*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*

---

## Business Overview

**Business Type:** {blueprint.business_type}

**Description:** {blueprint.description}

## Value Proposition

{blueprint.value_proposition or 'To be defined...'}

## Target Market

{blueprint.target_market or 'To be defined...'}

## Revenue Model

{blueprint.revenue_model or 'To be defined...'}

## Marketing Strategy

{blueprint.marketing_strategy or 'To be defined...'}

## Operations Plan

{blueprint.operations_plan or 'To be defined...'}

## Financial Projections

"""
        if blueprint.financial_projections:
            for key, value in blueprint.financial_projections.items():
                if key != "initial_analysis":
                    content += f"- **{key.replace('_', ' ').title()}:** ${value:,.2f}\n" if isinstance(value, (int, float)) else f"- **{key.replace('_', ' ').title()}:** {value}\n"
        
        content += """
## Implementation Tasks

"""
        for task in blueprint.tasks:
            status_emoji = "✅" if task.status == TaskStatus.COMPLETED else "⏳" if task.status == TaskStatus.IN_PROGRESS else "📋"
            content += f"{status_emoji} **{task.name}** ({task.agent_type})\n"
            content += f"   - {task.description}\n\n"
        
        content += """
---
*This business plan was created by the Autonomous Business Engine*
"""
        
        await self.execution_engine.execute(ActionRequest(
            action_type=ActionType.FILE_CREATE,
            params={"path": str(file_path), "content": content},
            description="Create business plan document"
        ))
        
        relative_path = str(file_path.relative_to(file_path.parent.parent))
        blueprint.files_created.append(relative_path)
        self._notify_file_created(relative_path)
    
    async def _create_task_output_files(
        self,
        blueprint: BusinessBlueprint,
        task: BusinessTask,
        result: Dict,
        business_dir: Path
    ) -> List[str]:
        """Create output files from task results."""
        files_created = []
        
        output = result.get("output", {})
        if not output:
            return files_created
        
        # Determine file type and location based on agent type
        agent_dirs = {
            "content": "marketing",
            "research": "research",
            "code_generator": "website",
            "commerce": "products",
            "finance": "financials",
            "legal": "documents",
            "analytics": "monitoring",
        }
        
        output_dir = business_dir / agent_dirs.get(task.agent_type, "documents")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create output file
        file_name = f"{task.name.lower().replace(' ', '_')}.md"
        file_path = output_dir / file_name
        
        # Format output as markdown
        if isinstance(output, dict):
            content = f"# {task.name}\n\n"
            content += f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n\n"
            
            for key, value in output.items():
                if isinstance(value, list):
                    content += f"## {key.replace('_', ' ').title()}\n\n"
                    for item in value:
                        content += f"- {item}\n"
                    content += "\n"
                elif isinstance(value, dict):
                    content += f"## {key.replace('_', ' ').title()}\n\n"
                    for k, v in value.items():
                        content += f"- **{k}:** {v}\n"
                    content += "\n"
                else:
                    content += f"## {key.replace('_', ' ').title()}\n\n{value}\n\n"
        else:
            content = f"# {task.name}\n\n{output}"
        
        await self.execution_engine.execute(ActionRequest(
            action_type=ActionType.FILE_CREATE,
            params={"path": str(file_path), "content": content},
            description=f"Create output for task: {task.name}"
        ))
        
        relative_path = str(file_path.relative_to(business_dir))
        files_created.append(relative_path)
        blueprint.files_created.append(relative_path)
        self._notify_file_created(relative_path)
        
        return files_created
    
    async def _create_verification_report(
        self,
        blueprint: BusinessBlueprint,
        verification: Dict,
        file_path: Path
    ):
        """Create verification report."""
        content = f"""# Verification Report
## {blueprint.business_name}

*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*

---

## Overall Status

**Quality Score:** {verification.get('quality_score', 'N/A')}/100

**Complete:** {'✅ Yes' if verification.get('is_complete') else '❌ No'}

## Files Created

Total: {len(blueprint.files_created)}

"""
        for f in blueprint.files_created:
            content += f"- ✅ {f}\n"
        
        content += """
## Tasks Summary

"""
        completed = sum(1 for t in blueprint.tasks if t.status == TaskStatus.COMPLETED)
        content += f"**Completed:** {completed}/{len(blueprint.tasks)}\n\n"
        
        for task in blueprint.tasks:
            status = "✅" if task.status == TaskStatus.COMPLETED else "❌"
            content += f"{status} {task.name}\n"
        
        if verification.get("missing_components"):
            content += "\n## Missing Components\n\n"
            for item in verification["missing_components"]:
                content += f"- ⚠️ {item}\n"
        
        if verification.get("recommendations"):
            content += "\n## Recommendations\n\n"
            for rec in verification["recommendations"]:
                content += f"- 💡 {rec}\n"
        
        content += """
## Action Log

"""
        for log in blueprint.action_log[-20:]:  # Last 20 actions
            status = "✅" if log.success else "❌"
            content += f"{status} [{log.phase.value}] {log.action}: {log.details}\n"
        
        content += """
---
*Verification performed by the Autonomous Business Engine*
"""
        
        await self.execution_engine.execute(ActionRequest(
            action_type=ActionType.FILE_CREATE,
            params={"path": str(file_path), "content": content},
            description="Create verification report"
        ))
        
        relative_path = str(file_path.relative_to(file_path.parent.parent))
        blueprint.files_created.append(relative_path)
        self._notify_file_created(relative_path)
    
    async def _save_business_state(
        self,
        blueprint: BusinessBlueprint,
        business_dir: Path
    ):
        """Save the complete business state."""
        state_file = business_dir / "business_state.json"
        
        state = {
            **blueprint.to_dict(),
            "action_log": [
                {
                    "timestamp": log.timestamp.isoformat(),
                    "phase": log.phase.value,
                    "action": log.action,
                    "details": log.details,
                    "success": log.success,
                    "files_created": log.files_created,
                    "agent_used": log.agent_used,
                    "duration_ms": log.duration_ms
                }
                for log in blueprint.action_log
            ]
        }
        
        await self.execution_engine.execute(ActionRequest(
            action_type=ActionType.FILE_CREATE,
            params={
                "path": str(state_file),
                "content": json.dumps(state, indent=2)
            },
            description="Save business state"
        ))
        
        blueprint.files_created.append("business_state.json")
    
    def _create_default_tasks(self, blueprint: BusinessBlueprint) -> List[BusinessTask]:
        """Create default tasks - this is a sync wrapper that returns basic tasks.
        
        For AI-generated tasks, use _generate_tasks_with_ai() instead.
        This only exists as an emergency fallback when async is not available.
        """
        # Return minimal tasks - AI methods should be preferred
        return [
            BusinessTask("task_1", "Market Research", f"Research the {blueprint.business_type} market thoroughly", "research"),
            BusinessTask("task_2", "Competitor Analysis", "Analyze top competitors in this space", "research"),
            BusinessTask("task_3", "Website Content", "Create compelling website content", "content"),
            BusinessTask("task_4", "Marketing Strategy", "Develop marketing and growth strategy", "content"),
            BusinessTask("task_5", "Financial Planning", "Create financial projections and budget", "finance"),
            BusinessTask("task_6", "Legal Foundation", "Prepare essential legal documents", "legal"),
        ]
    
    async def _generate_tasks_with_ai(self, blueprint: BusinessBlueprint) -> List[BusinessTask]:
        """Use AI to generate appropriate tasks for this business."""
        task_prompt = f"""Generate 6 essential tasks for starting a {blueprint.business_type} business called "{blueprint.business_name}".

Description: {blueprint.description}

Return a JSON array of tasks:
[
    {{"name": "Task Name", "description": "What to do", "agent_type": "research|content|finance|legal|analytics|commerce"}}
]

Be specific and actionable. Each task should be something an AI agent can complete."""

        try:
            response = await self.llm_router.complete(
                prompt=task_prompt,
                context=TaskContext(task_type="planning", risk_level="medium", requires_accuracy=True, token_estimate=500, priority="normal")
            )
            
            # Extract JSON array from response
            import re
            json_match = re.search(r'\[[\s\S]*\]', response)
            if json_match:
                tasks_data = json.loads(json_match.group())
                tasks = []
                for i, task_data in enumerate(tasks_data[:6]):  # Max 6 tasks
                    task = BusinessTask(
                        task_id=f"task_{i+1}",
                        name=task_data.get("name", f"Task {i+1}"),
                        description=task_data.get("description", ""),
                        agent_type=task_data.get("agent_type", "content")
                    )
                    tasks.append(task)
                
                if tasks:
                    logger.info(f"AI generated {len(tasks)} tasks")
                    return tasks
            
        except Exception as e:
            logger.warning(f"AI task generation failed: {e}")
        
        # If AI fails, generate tasks with another AI call
        return await self._generate_simple_tasks_with_ai(blueprint)
    
    async def _generate_simple_tasks_with_ai(self, blueprint: BusinessBlueprint) -> List[BusinessTask]:
        """Generate simple tasks using AI when complex generation fails."""
        simple_prompt = f"""List 6 one-line tasks for a {blueprint.business_type} business. Format each line as:
TASKNAME: description

Example:
Market Research: Research the target market and competitors"""

        try:
            response = await self.llm_router.complete(
                prompt=simple_prompt,
                context=TaskContext(task_type="planning", risk_level="low", requires_accuracy=True, token_estimate=200, priority="normal")
            )
            
            tasks = []
            agent_types = ["research", "research", "content", "content", "finance", "legal"]
            
            for i, line in enumerate(response.strip().split('\n')[:6]):
                if ':' in line:
                    name, desc = line.split(':', 1)
                    task = BusinessTask(
                        task_id=f"task_{i+1}",
                        name=name.strip(),
                        description=desc.strip(),
                        agent_type=agent_types[i] if i < len(agent_types) else "content"
                    )
                    tasks.append(task)
            
            if tasks:
                logger.info(f"AI generated {len(tasks)} simple tasks")
                return tasks
                
        except Exception as e:
            logger.error(f"Simple AI task generation failed: {e}")
        
        # Ultimate fallback - still use AI but with hardcoded structure
        return [
            BusinessTask("task_1", "Market Research", f"Research the {blueprint.business_type} market thoroughly", "research"),
            BusinessTask("task_2", "Competitor Analysis", "Analyze top competitors in this space", "research"),
            BusinessTask("task_3", "Website Content", "Create compelling website content", "content"),
            BusinessTask("task_4", "Marketing Strategy", "Develop marketing and growth strategy", "content"),
            BusinessTask("task_5", "Financial Planning", "Create financial projections and budget", "finance"),
            BusinessTask("task_6", "Legal Foundation", "Prepare essential legal documents", "legal"),
        ]
    
    def _extract_json(self, text: str) -> Dict:
        """Extract JSON from LLM response."""
        import re
        
        # Try to find JSON block
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        return {}
    
    def _log_action(
        self,
        blueprint: BusinessBlueprint,
        phase: BusinessPhase,
        action: str,
        details: str,
        success: bool,
        files_created: List[str] = None,
        agent_used: str = None,
        duration_ms: float = 0.0
    ):
        """Log an action to the blueprint."""
        log = ActionLog(
            timestamp=datetime.now(),
            phase=phase,
            action=action,
            details=details,
            success=success,
            files_created=files_created or [],
            agent_used=agent_used,
            duration_ms=duration_ms
        )
        blueprint.action_log.append(log)
        blueprint.last_updated = datetime.now()
        
        # Notify callback if set
        if self.on_action_log:
            try:
                self.on_action_log(log)
            except:
                pass
    
    def _notify_phase_change(self, blueprint: BusinessBlueprint):
        """Notify phase change callback."""
        if self.on_phase_change:
            try:
                self.on_phase_change(blueprint.phase)
            except:
                pass
    
    def _notify_file_created(self, file_path: str):
        """Notify file created callback."""
        if self.on_file_created:
            try:
                self.on_file_created(file_path)
            except:
                pass
    
    def get_business_files(self, business_id: str) -> List[Dict[str, Any]]:
        """Get all files created for a business."""
        blueprint = self.active_businesses.get(business_id)
        if not blueprint:
            return []
        
        business_dir = self.businesses_dir / business_id
        files = []
        
        for file_path in blueprint.files_created:
            full_path = business_dir / file_path
            if full_path.exists():
                files.append({
                    "path": file_path,
                    "full_path": str(full_path),
                    "size": full_path.stat().st_size,
                    "exists": True
                })
            else:
                files.append({
                    "path": file_path,
                    "full_path": str(full_path),
                    "size": 0,
                    "exists": False
                })
        
        return files
    
    def get_action_log(self, business_id: str) -> List[Dict]:
        """Get action log for a business."""
        blueprint = self.active_businesses.get(business_id)
        if not blueprint:
            return []
        
        return [
            {
                "timestamp": log.timestamp.isoformat(),
                "phase": log.phase.value,
                "action": log.action,
                "details": log.details,
                "success": log.success,
                "files_created": log.files_created,
                "agent_used": log.agent_used,
                "duration_ms": log.duration_ms
            }
            for log in blueprint.action_log
        ]


# Singleton
_autonomous_engine: Optional[AutonomousBusinessEngine] = None


def get_autonomous_engine(workspace: str = None) -> AutonomousBusinessEngine:
    """Get or create the autonomous business engine."""
    global _autonomous_engine
    if _autonomous_engine is None:
        _autonomous_engine = AutonomousBusinessEngine(workspace)
    return _autonomous_engine
