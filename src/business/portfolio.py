"""
Portfolio Manager - High-level empire analysis.
Aggregates performance data across all business units.
"""

from src.database.models import BusinessUnit, BusinessStatus
from src.database.connection import get_db
from sqlalchemy import select

class PortfolioManager:
    """
    The 'Dashboard Brain' for the entire portfolio.
    """
    
    async def get_total_stats(self) -> dict:
        """
        Calculates aggregate KPIs for the entire empire.
        """
        async with get_db() as db:
            result = await db.execute(select(BusinessUnit))
            units = result.scalars().all()
            
            total_revenue = sum(u.total_revenue for u in units)
            total_expenses = sum(u.total_expenses for u in units)
            
            return {
                "total_businesses": len(units),
                "total_revenue": total_revenue,
                "total_profit": total_revenue - total_expenses,
                "active_units": [u.name for u in units if u.status != BusinessStatus.SUNSET],
                "stages": self._count_stages(units)
            }

    def _count_stages(self, units: list[BusinessUnit]) -> dict:
        """Counts how many businesses are in each stage."""
        stats = {}
        for u in units:
            stats[u.status.value] = stats.get(u.status.value, 0) + 1
        return stats
