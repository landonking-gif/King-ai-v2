import json
import os
from pathlib import Path
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
        from src.master_ai.kpi_monitor import kpi_monitor
        
        # Gather real-time data
        health_report = await kpi_monitor.get_system_health()
        
        prompt = EVOLUTION_PROMPT.format(
            context=context,
            performance=health_report
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
        
        # Get project root (two levels up from this file)
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        
        # Initialize patcher with project root
        patcher = CodePatcher(project_root)
        
        # Create and apply patch
        patch = patcher.create_patch(file_path, new_code, description="Evolution proposal")
        
        # Validate before applying
        is_valid, errors = patcher.validate_patch(patch)
        if not is_valid:
            return {
                "success": False,
                "error": f"Validation failed: {', '.join(errors)}"
            }
        
        # Apply the patch
        success = patcher.apply_patch(patch)
        
        if success:
            return {
                "success": True,
                "message": "Patch applied successfully",
                "stats": patch.stats
            }
        else:
            return {
                "success": False,
                "error": patch.error or "Unknown error"
            }
