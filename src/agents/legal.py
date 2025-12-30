"""
Legal Agent - Compliance and risk mitigration.
Expert in contract analysis, GDPR/CCPA compliance, and intellectual property.
"""

from src.agents.base import SubAgent
from src.utils.metrics import TASKS_EXECUTED

class LegalAgent(SubAgent):
    """
    Compliance and risk management agent.
    """
    name = "legal"
    description = "Ensures business compliance and analyzes legal documents."
    
    async def execute(self, task: dict) -> dict:
        """
        Executes a legal check or document analysis.
        """
        description = task.get("description", "Legal check")
        input_data = task.get("input_data", {})
        
        prompt = f"""
        ### TASK: LEGAL & COMPLIANCE
        {description}
        
        ### DATA / DOCUMENT:
        {input_data}
        
        ### INSTRUCTION:
        Evaluate for risk, compliance issues, and legal obligations.
        Provide a concise summary of concerns and recommended mitigations.
        """
        
        try:
            result = await self._ask_llm(prompt)
            TASKS_EXECUTED.labels(agent=self.name, status="success").inc()
            return {
                "success": True, 
                "output": result, 
                "metadata": {"type": "legal_check"}
            }
        except Exception as e:
            TASKS_EXECUTED.labels(agent=self.name, status="failed").inc()
            return {"success": False, "error": str(e)}
