"""
Supervisor Agent for Multi-Agent Orchestration

Coordinates multiple specialized agents to complete complex tasks.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from agent_framework.agents.base_agent import BaseAgent, AgentAction, AgentFinish


@dataclass
class AgentTask:
    """Task assigned to an agent"""
    agent_name: str
    task: str
    priority: int = 0
    dependencies: List[str] = None


@dataclass
class TaskResult:
    """Result from agent task execution"""
    agent_name: str
    task: str
    result: str
    success: bool
    error: Optional[str] = None


class SupervisorAgent:
    """
    Supervisor that orchestrates multiple agents.
    
    Responsibilities:
    - Task decomposition and assignment
    - Agent coordination
    - Result synthesis
    - Error handling and recovery
    """
    
    def __init__(self, llm: Any, max_iterations: int = 10):
        self.llm = llm
        self.max_iterations = max_iterations
        self.agents: Dict[str, BaseAgent] = {}
        self.task_history: List[TaskResult] = []
    
    def register_agent(self, agent: BaseAgent) -> None:
        """Register an agent with the supervisor"""
        self.agents[agent.name] = agent
        print(f"✅ Registered agent: {agent.name}")
    
    def unregister_agent(self, agent_name: str) -> None:
        """Unregister an agent"""
        if agent_name in self.agents:
            del self.agents[agent_name]
            print(f"❌ Unregistered agent: {agent_name}")
    
    def list_agents(self) -> List[str]:
        """List all registered agents"""
        return list(self.agents.keys())
    
    def _build_planning_prompt(self, task: str, context: str = "") -> str:
        """Build prompt for task planning"""
        agent_descriptions = "\n".join([
            f"- {name}: {agent.description}"
            for name, agent in self.agents.items()
        ])
        
        prompt = f"""You are a supervisor coordinating multiple AI agents to complete a complex task.

Available Agents:
{agent_descriptions}

Task: {task}

{f"Previous Context: {context}" if context else ""}

Analyze the task and create a plan. For each step, specify:
1. Which agent should handle it
2. What specific sub-task they should complete
3. Any dependencies on previous steps

Respond in this format:
PLAN:
Step 1: [agent_name] - [sub-task description]
Step 2: [agent_name] - [sub-task description]
...

If the task is complete, respond with:
COMPLETE: [final answer]
"""
        return prompt
    
    def _parse_plan(self, response: str) -> List[AgentTask]:
        """Parse planning response into agent tasks"""
        tasks = []
        
        if "PLAN:" in response:
            plan_section = response.split("PLAN:")[-1].strip()
            lines = plan_section.split("\n")
            
            for i, line in enumerate(lines):
                line = line.strip()
                if line.startswith("Step"):
                    # Parse "Step N: agent_name - task"
                    try:
                        parts = line.split(":", 1)[-1].strip()
                        if " - " in parts:
                            agent_name, task_desc = parts.split(" - ", 1)
                            agent_name = agent_name.strip()
                            
                            if agent_name in self.agents:
                                tasks.append(AgentTask(
                                    agent_name=agent_name,
                                    task=task_desc.strip(),
                                    priority=i
                                ))
                    except Exception:
                        continue
        
        return tasks
    
    def _execute_task(self, agent_task: AgentTask) -> TaskResult:
        """Execute a single agent task"""
        agent = self.agents.get(agent_task.agent_name)
        
        if not agent:
            return TaskResult(
                agent_name=agent_task.agent_name,
                task=agent_task.task,
                result="",
                success=False,
                error=f"Agent '{agent_task.agent_name}' not found"
            )
        
        try:
            print(f"🤖 {agent_task.agent_name} executing: {agent_task.task[:50]}...")
            result = agent.run(agent_task.task)
            
            return TaskResult(
                agent_name=agent_task.agent_name,
                task=agent_task.task,
                result=result,
                success=True
            )
        
        except Exception as e:
            return TaskResult(
                agent_name=agent_task.agent_name,
                task=agent_task.task,
                result="",
                success=False,
                error=str(e)
            )
    
    def _synthesize_results(self, task: str, results: List[TaskResult]) -> str:
        """Synthesize results from multiple agents"""
        results_text = "\n\n".join([
            f"**{r.agent_name}** ({r.task}):\n{r.result}"
            for r in results if r.success
        ])
        
        prompt = f"""You are synthesizing results from multiple AI agents.

Original Task: {task}

Agent Results:
{results_text}

Provide a comprehensive final answer that combines all the agent outputs.
Be concise but complete.

Final Answer:"""
        
        try:
            response = self.llm.generate(prompt)
            return response
        except Exception as e:
            # Fallback: just concatenate results
            return results_text
    
    def run(self, task: str) -> Dict[str, Any]:
        """
        Run the supervisor on a complex task.
        
        Args:
            task: The task to complete
            
        Returns:
            Dictionary with final result and execution details
        """
        context = ""
        all_results = []
        
        for iteration in range(self.max_iterations):
            print(f"\n📋 Iteration {iteration + 1}")
            
            # Plan next steps
            planning_prompt = self._build_planning_prompt(task, context)
            
            try:
                plan_response = self.llm.generate(planning_prompt)
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Planning failed: {str(e)}",
                    "results": all_results
                }
            
            # Check if complete
            if "COMPLETE:" in plan_response:
                final_answer = plan_response.split("COMPLETE:")[-1].strip()
                return {
                    "success": True,
                    "answer": final_answer,
                    "results": all_results,
                    "iterations": iteration + 1
                }
            
            # Parse and execute plan
            agent_tasks = self._parse_plan(plan_response)
            
            if not agent_tasks:
                # No valid tasks, try to synthesize what we have
                if all_results:
                    final_answer = self._synthesize_results(task, all_results)
                    return {
                        "success": True,
                        "answer": final_answer,
                        "results": all_results,
                        "iterations": iteration + 1
                    }
                else:
                    return {
                        "success": False,
                        "error": "Could not create execution plan",
                        "results": []
                    }
            
            # Execute tasks
            for agent_task in agent_tasks:
                result = self._execute_task(agent_task)
                all_results.append(result)
                self.task_history.append(result)
                
                # Update context
                if result.success:
                    context += f"\n{result.agent_name}: {result.result[:500]}"
        
        # Max iterations reached - synthesize what we have
        if all_results:
            final_answer = self._synthesize_results(task, all_results)
            return {
                "success": True,
                "answer": final_answer,
                "results": all_results,
                "iterations": self.max_iterations,
                "note": "Reached max iterations"
            }
        
        return {
            "success": False,
            "error": f"Reached max iterations ({self.max_iterations})",
            "results": all_results
        }
