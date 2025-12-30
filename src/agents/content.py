"""
Content Agent - Generates marketing and operational content.
Expert in copywritting, SEO, and social media engagement.
"""

from src.agents.base import SubAgent
from src.utils.metrics import TASKS_EXECUTED

class ContentAgent(SubAgent):
    """
    Creative agent for brand voice and customer communication.
    """
    name = "content"
    description = "Generates marketing copy, blog posts, and social media content."
    
    async def execute(self, task: dict) -> dict:
        """
        Executes content creation task.
        """
        description = task.get("description", "Create content")
        input_data = task.get("input_data", {})
        
        prompt = f"""
        ### TASK: CONTENT CREATION
        {description}
        
        ### INPUT DATA:
        {input_data}
        
        ### INSTRUCTION:
        Generate high-conversion, professional content.
        Ensure it aligns with the brand voice and target audience needs.
        """
        
        try:
            result = await self._ask_llm(prompt)
            TASKS_EXECUTED.labels(agent=self.name, status="success").inc()
            return {
                "success": True, 
                "output": result, 
                "metadata": {"type": "content_creation"}
            }
        except Exception as e:
            TASKS_EXECUTED.labels(agent=self.name, status="failed").inc()
            return {"success": False, "error": str(e)}
