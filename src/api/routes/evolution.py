"""
Evolution Routes - Management of self-modification proposals.
"""

from fastapi import APIRouter, HTTPException
from src.database.connection import get_db
from src.database.models import EvolutionProposal, EvolutionStatus
from src.master_ai.brain import MasterAI
from sqlalchemy import select

router = APIRouter()

@router.get("/proposals")
async def list_proposals():
    """Returns all system evolution proposals."""
    async with get_db() as db:
        result = await db.execute(
            select(EvolutionProposal).order_by(EvolutionProposal.created_at.desc())
        )
        return result.scalars().all()

@router.get("/proposals/{proposal_id}")
async def get_proposal(proposal_id: str):
    """Returns details for a specific evolution proposal."""
    async with get_db() as db:
        prop = await db.get(EvolutionProposal, proposal_id)
        if not prop:
            raise HTTPException(status_code=404, detail="Proposal not found")
        return prop
@router.post("/approve/{proposal_id}")
async def approve_proposal(proposal_id: str):
    """Executes an approved evolution proposal."""
    async with get_db() as db:
        prop = await db.get(EvolutionProposal, proposal_id)
        if not prop:
            raise HTTPException(status_code=404, detail="Proposal not found")
        
        if prop.status != EvolutionStatus.PENDING:
            return {"success": False, "message": f"Proposal is already {prop.status}"}

        # Execute
        from src.master_ai.evolution import EvolutionEngine
        from src.utils.ollama_client import OllamaClient
        from config.settings import settings
        
        engine = EvolutionEngine(OllamaClient(settings.ollama_url, settings.ollama_model))
        
        # In a real scenario, prop.proposed_changes would be structured JSON
        # For MVP, we assume it contains 'file_path' and 'new_code'
        try:
            # Check if proposed_changes is a string (JSON) or dict
            import json
            changes = prop.proposed_changes if isinstance(prop.proposed_changes, dict) else json.loads(prop.proposed_changes)
            
            result = await engine.apply_proposal(
                changes.get("file_path"),
                changes.get("new_code")
            )
            
            if result["success"]:
                prop.status = EvolutionStatus.APPROVED
                await db.commit()
                return {"success": True, "diff": result["diff"]}
            else:
                return {"success": False, "error": result["error"]}
                
        except Exception as e:
            return {"success": False, "error": f"Execution failed: {str(e)}"}

@router.post("/reject/{proposal_id}")
async def reject_proposal(proposal_id: str):
    """Rejects an evolution proposal."""
    async with get_db() as db:
        prop = await db.get(EvolutionProposal, proposal_id)
        if not prop:
            raise HTTPException(status_code=404, detail="Proposal not found")
        
        prop.status = EvolutionStatus.REJECTED
        await db.commit()
        return {"success": True}
