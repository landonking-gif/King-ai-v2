"""
Code Generator Agent - Generates and modifies source code.
Uses the Master LLM to produce application logic, scripts, and frontends.
"""

from src.agents.base import SubAgent
from src.utils.metrics import TASKS_EXECUTED

class CodeGeneratorAgent(SubAgent):
    """
    Expert programmer agent capable of high-quality code production.
    """
    name = "code_generator"
    description = "Generates and modifies source code for business applications."
    
    async def execute(self, task: dict) -> dict:
        """
        Executes a code generation or refactoring task.
        """
        description = task.get("description", "Generate code")
        input_data = task.get("input_data", {})
        
        prompt = f"""
        ### TASK: CODE GENERATION
        {description}
        
        ### CONTEXT / REQUIREMENTS:
        {input_data}
        
        ### INSTRUCTION:
        Provide clean, production-ready code or a clear diff.
        Use standard best practices for the requested language.
        """
        
        try:
            result = await self._ask_llm(prompt)
            TASKS_EXECUTED.labels(agent=self.name, status="success").inc()
            return {
                "success": True, 
                "output": result, 
                "metadata": {"type": "code_gen"}
            }
        except Exception as e:
            TASKS_EXECUTED.labels(agent=self.name, status="failed").inc()
            return {"success": False, "error": str(e)}
