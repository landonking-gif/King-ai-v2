import json
from src.utils.ollama_client import OllamaClient
from src.master_ai.prompts import PLANNING_PROMPT

class Planner:
    """
    Translates vague user requests into a structured DAG of tasks.
    Example: "Start a store" -> [Research niche, Pick name, Setup backend]
    """
    
    def __init__(self, ollama: OllamaClient):
        """
        Initializes the planner with an LLM client.
        :param ollama: The OllamaClient instance for prompt execution.
        """
        self.ollama = ollama
        
    async def create_plan(self, goal: str, action: str | None, parameters: dict, context: str) -> dict:
        """
        Create a detailed execution plan for a goal.
        """
        # Construct the prompt
        prompt = PLANNING_PROMPT.format(
            goal=goal,
            context=context
        )
        
        # Get LLM response
        response = await self.ollama.complete(prompt)
        
        # Parse JSON
        try:
            # Clean up potential markdown code blocks
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()
                
            return json.loads(response)
        except Exception as e:
            # Fallback for parsing error
            print(f"Error parsing plan: {e}")
            return {
                "goal": goal,
                "steps": [
                    {
                        "name": "Planner Error Fallback",
                        "description": f"Failed to parse plan. Original goal: {goal}",
                        "agent": "manager", # Fallback agent
                        "requires_approval": True,
                        "dependencies": [],
                        "estimated_duration": "unknown"
                    }
                ]
            }
