"""
Business Unit Manager - Core logic for business lifecycle operations.
Handles creation, cloning, and metadata updates for empire ventures.
"""

from typing import Optional
from uuid import uuid4
from src.database.models import BusinessUnit, BusinessStatus
from src.database.connection import get_db, get_db_ctx
from src.utils.metrics import ACTIVE_BUSINESSES
from src.utils.logging import get_logger

logger = get_logger(__name__)


class BusinessManager:
    """
    Manages individual business units in the portfolio.
    """
    
    async def create_business(self, name: str, business_type: str, playbook_id: str = None) -> BusinessUnit:
        """
        Registers a new venture in the empire.
        """
        async with get_db_ctx() as db:
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

    async def get_business(self, business_id: str) -> Optional[BusinessUnit]:
        """
        Retrieve a business unit by ID.
        """
        async with get_db_ctx() as db:
            return await db.get(BusinessUnit, business_id)

    async def clone_business(
        self,
        source_id: str,
        new_name: str,
        new_niche: Optional[str] = None,
        inherit_config: bool = True,
    ) -> Optional[BusinessUnit]:
        """
        Clone a successful business unit for replication to new markets.
        
        This implements the REPLICATION stage of the business lifecycle,
        allowing profitable ventures to be replicated to new niches.
        
        Args:
            source_id: ID of the business unit to clone
            new_name: Name for the new cloned business
            new_niche: Optional new niche/market (updates config)
            inherit_config: Whether to copy configuration from source
            
        Returns:
            The newly created cloned BusinessUnit, or None if source not found
        """
        async with get_db_ctx() as db:
            source = await db.get(BusinessUnit, source_id)
            if not source:
                logger.error(f"Cannot clone: source business {source_id} not found")
                return None
            
            # Prepare cloned config
            cloned_config = dict(source.config) if inherit_config else {}
            cloned_config["cloned_from"] = source_id
            cloned_config["clone_date"] = str(uuid4())[:8]  # Unique clone marker
            
            if new_niche:
                cloned_config["niche"] = new_niche
            
            # Create the clone with fresh status
            clone = BusinessUnit(
                id=str(uuid4()),
                name=new_name,
                type=source.type,
                status=BusinessStatus.SETUP,  # Start at SETUP (validated by source success)
                playbook_id=source.playbook_id,
                total_revenue=0.0,
                total_expenses=0.0,
                kpis={
                    "source_revenue": source.total_revenue,
                    "source_roi": self._calculate_roi(source),
                },
                config=cloned_config,
            )
            db.add(clone)
            await db.commit()
            
            # Update metrics
            ACTIVE_BUSINESSES.inc()
            
            logger.info(f"Cloned business {source_id} -> {clone.id} ({new_name})")
            return clone

    def _calculate_roi(self, unit: BusinessUnit) -> float:
        """Calculate ROI for a business unit."""
        if unit.total_expenses == 0:
            return 0.0
        return ((unit.total_revenue - unit.total_expenses) / unit.total_expenses) * 100

    async def update_financials(self, business_id: str, revenue_delta: float = 0, expense_delta: float = 0):
        """
        Updates the financial health of a business unit.
        """
        async with get_db_ctx() as db:
            unit = await db.get(BusinessUnit, business_id)
            if unit:
                unit.total_revenue += revenue_delta
                unit.total_expenses += expense_delta
                await db.commit()
