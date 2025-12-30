"""
Analytics Agent - Data-driven insight generation.
Expert in pattern recognition, market trends, and KPI monitoring.
"""

from src.agents.base import SubAgent
from src.utils.metrics import TASKS_EXECUTED

class AnalyticsAgent(SubAgent):
    """
    Data scientist agent for business intelligence.
    """
    name = "analytics"
    description = "Performs data analysis and provides actionable business insights."
    
    async def execute(self, task: dict) -> dict:
        """
        Executes a data analysis task.
        """
        description = task.get("description", "Analysis task")
        input_data = task.get("input_data", {})
        
        prompt = f"""
        ### TASK: DATA ANALYSIS
        {description}
        
        ### RAW DATA:
        {input_data}
        
        ### INSTRUCTION:
        Identify trends, anomalies, and actionable insights.
        Provide data-backed conclusions for strategic decision making.
        """
        
        try:
            result = await self._ask_llm(prompt)
            TASKS_EXECUTED.labels(agent=self.name, status="success").inc()
            return {
                "success": True, 
                "output": result, 
                "metadata": {"type": "data_analysis"}
            }
        except Exception as e:
            TASKS_EXECUTED.labels(agent=self.name, status="failed").inc()
            return {"success": False, "error": str(e)}
