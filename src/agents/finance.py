"""
Finance Agent - Manages financial reporting and profit analysis.
Expert in budgeting, forecasting, and investment tracking.
"""

from src.agents.base import SubAgent
from src.utils.metrics import TASKS_EXECUTED

class FinanceAgent(SubAgent):
    """
    The 'CFO' sub-agent for the empire.
    """
    name = "finance"
    description = "Manages financial tracking, budgeting, and profit analysis."
    
    async def execute(self, task: dict) -> dict:
        """
        Executes a financial analysis or planning task.
        """
        description = task.get("description", "Financial task")
        input_data = task.get("input_data", {})
        
        prompt = f"""
        ### TASK: FINANCIAL ANALYSIS
        {description}
        
        ### FINANCIAL DATA:
        {input_data}
        
        ### INSTRUCTION:
        Analyze the financial implications and provide recommendations.
        Focus on ROI, cost-cutting, and revenue growth.
        """
        
        try:
            result = await self._ask_llm(prompt)
            TASKS_EXECUTED.labels(agent=self.name, status="success").inc()
            return {
                "success": True, 
                "output": result, 
                "metadata": {"type": "finance_analysis"}
            }
        except Exception as e:
            TASKS_EXECUTED.labels(agent=self.name, status="failed").inc()
            return {"success": False, "error": str(e)}
