"""
Business Routes - CRUD operations for managed business units.
"""

from fastapi import APIRouter, HTTPException
from src.database.connection import get_db
from src.database.models import BusinessUnit, BusinessStatus
from sqlalchemy import select

router = APIRouter()

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
