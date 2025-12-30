import json
from src.utils.ollama_client import OllamaClient
from src.master_ai.prompts import EVOLUTION_PROMPT

class EvolutionEngine:
    """
    The 'Self-Correction' module.
    Identifies bottlenecks or outdated logic and suggests code or config changes.
    """
    
    def __init__(self, ollama: OllamaClient):
        """
        :param ollama: OllamaClient for LLM interaction.
        """
        self.ollama = ollama
        
    async def propose_improvement(self, context: str) -> dict | None:
        """
        Analyze system and propose beneficial changes.
        """
        prompt = EVOLUTION_PROMPT.format(
            context=context,
            performance="Recent performance metrics would go here..." # Placeholder
        )
        
        response = await self.ollama.complete(prompt)
        
        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()
            
            data = json.loads(response)
            
            if data.get("is_beneficial"):
                return data
            return None
            
        except Exception as e:
            print(f"Error parsing evolution proposal: {e}")
            return None
    async def apply_proposal(self, file_path: str, new_code: str) -> dict:
        """
        Physically applies a code modification to the codebase.
        Used after human approval of a proposal.
        """
        from src.utils.code_patcher import CodePatcher
        
        # Absolute path resolution (base of the project)
        abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", file_path))
        
        patcher = CodePatcher()
        result = patcher.apply_patch(abs_path, new_code)
        
        return result
