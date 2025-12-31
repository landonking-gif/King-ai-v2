"""
Business Routes - CRUD operations for managed business units.
"""

from typing import Optional
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException
from src.database.connection import get_db
from src.database.models import BusinessUnit, BusinessStatus
from src.business.unit import BusinessManager
from sqlalchemy import select

router = APIRouter()
business_manager = BusinessManager()


class CloneRequest(BaseModel):
    """Request model for cloning a business."""
    new_name: str
    new_niche: Optional[str] = None
    inherit_config: bool = True


@router.get("/")
async def list_businesses():
    """Returns all businesses in the empire portfolio."""
    async with get_db() as db:
        result = await db.execute(select(BusinessUnit).order_by(BusinessUnit.created_at.desc()))
        return result.scalars().all()

@router.get("/{business_id}")
async def get_business(business_id: str):
    """Returns detailed status for a single business unit."""
    async with get_db() as db:
        unit = await db.get(BusinessUnit, business_id)
        if not unit:
            raise HTTPException(status_code=404, detail="Business not found")
        return unit

@router.post("/{business_id}/clone")
async def clone_business(business_id: str, request: CloneRequest):
    """
    Clone a successful business unit to a new market/niche.
    
    This implements the REPLICATION lifecycle stage, allowing proven
    business models to be replicated efficiently.
    """
    clone = await business_manager.clone_business(
        source_id=business_id,
        new_name=request.new_name,
        new_niche=request.new_niche,
        inherit_config=request.inherit_config,
    )
    
    if not clone:
        raise HTTPException(status_code=404, detail="Source business not found")
    
    return {
        "status": "cloned",
        "clone_id": clone.id,
        "clone_name": clone.name,
        "source_id": business_id,
    }

@router.delete("/{business_id}")
async def delete_business(business_id: str):
    """Removes a business unit from the portfolio."""
    async with get_db() as db:
        unit = await db.get(BusinessUnit, business_id)
        if not unit:
            raise HTTPException(status_code=404, detail="Business not found")
        await db.delete(unit)
        await db.commit()
        return {"status": "deleted"}
