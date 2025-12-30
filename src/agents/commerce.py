"""
Commerce Agent - Handles e-commerce and retail operations.
Expert in product sourcing, pricing, and fulfillment logic.
"""

from src.agents.base import SubAgent
from src.utils.metrics import TASKS_EXECUTED

class CommerceAgent(SubAgent):
    """
    Operational agent for physical or digital product sales.
    """
    name = "commerce"
    description = "Handles e-commerce operations like product sourcing and store management."
    
    async def execute(self, task: dict) -> dict:
        """
        Executes a commerce operational task.
        """
        description = task.get("description", "Commerce task")
        input_data = task.get("input_data", {})
        
        prompt = f"""
        ### TASK: COMMERCE OPERATION
        {description}
        
        ### DETAILS:
        {input_data}
        
        ### INSTRUCTION:
        Recommend the most efficient way to handle this commerce task.
        Include pricing strategies, sourcing options, or fulfillment steps.
        """
        
        try:
            result = await self._ask_llm(prompt)
            TASKS_EXECUTED.labels(agent=self.name, status="success").inc()
            return {
                "success": True, 
                "output": result, 
                "metadata": {"type": "commerce_op"}
            }
        except Exception as e:
            TASKS_EXECUTED.labels(agent=self.name, status="failed").inc()
            return {"success": False, "error": str(e)}
