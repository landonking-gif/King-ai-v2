"""
Business Unit Manager - Core logic for business lifecycle operations.
Handles creation and metadata updates for empire ventures.
"""

from uuid import uuid4
from src.database.models import BusinessUnit, BusinessStatus
from src.database.connection import get_db
from src.utils.metrics import ACTIVE_BUSINESSES

class BusinessManager:
    """
    Manages individual business units in the portfolio.
    """
    
    async def create_business(self, name: str, business_type: str, playbook_id: str = None) -> BusinessUnit:
        """
        Registers a new venture in the empire.
        """
        async with get_db() as db:
            unit = BusinessUnit(
                id=str(uuid4()),
                name=name,
                type=business_type,
                status=BusinessStatus.DISCOVERY,
                playbook_id=playbook_id,
                total_revenue=0.0,
                total_expenses=0.0,
                kpis={},
                config={}
            )
            db.add(unit)
            await db.commit()
            
            # Update metrics
            ACTIVE_BUSINESSES.inc()
            
            return unit

    async def update_financials(self, business_id: str, revenue_delta: float = 0, expense_delta: float = 0):
        """
        Updates the financial health of a business unit.
        """
        async with get_db() as db:
            unit = await db.get(BusinessUnit, business_id)
            if unit:
                unit.total_revenue += revenue_delta
                unit.total_expenses += expense_delta
                await db.commit()
