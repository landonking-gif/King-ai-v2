"""
Portfolio Manager - Manage business portfolios.
"""
import uuid
from datetime import datetime
from typing import Optional
from src.business.portfolio_models import (
    Portfolio, PortfolioStatus, AllocationStrategy,
    ResourceAllocation, PortfolioMetrics, BusinessPerformance,
    RebalanceAction, RebalanceReport
)
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Performance calculation weights
PROFIT_WEIGHT = 0.5
HEALTH_SCORE_WEIGHT = 0.5

# Ranking weights
PROFIT_RANKING_WEIGHT = 0.4
GROWTH_RANKING_WEIGHT = 0.3
HEALTH_RANKING_WEIGHT = 0.3

# Rebalancing threshold
REBALANCE_THRESHOLD = 0.05  # 5% difference threshold

# Stage bonuses for growth strategy
STAGE_BONUSES = {
    "ideation": 1.5,
    "validation": 1.3,
    "growth": 1.2,
}


class PortfolioManager:
    """Manage portfolios of businesses."""

    def __init__(self):
        self._portfolios: dict[str, Portfolio] = {}
        self._business_data: dict[str, dict] = {}  # Mock business data

    def _count_stages(self, business_units: list) -> dict[str, int]:
        """Count businesses by lifecycle status.

        Expects items with a `.status` attribute (e.g., BusinessStatus).
        Returns lowercase status names mapped to counts.
        """
        counts: dict[str, int] = {}
        for unit in business_units:
            status = getattr(unit, "status", None)
            status_value = None
            if status is None:
                continue
            # BusinessStatus is an Enum with `.value`
            status_value = getattr(status, "value", None)
            if not isinstance(status_value, str):
                status_value = str(status).lower()
            key = status_value.lower()
            counts[key] = counts.get(key, 0) + 1
        return counts

    async def get_total_stats(self) -> dict:
        """Get aggregate stats across all businesses.

        Uses the database when available; falls back to in-memory mock metrics.
        """
        try:
            from sqlalchemy import select, func
            from src.database.models import BusinessUnit
            from src.database.connection import get_db

            async with get_db() as db:
                result = await db.execute(
                    select(
                        func.sum(BusinessUnit.total_revenue).label("total_revenue"),
                        func.sum(BusinessUnit.total_expenses).label("total_expenses"),
                        func.count(BusinessUnit.id).label("count"),
                    )
                )
                row = result.first()

                total_revenue = (row.total_revenue or 0) if row else 0
                total_expenses = (row.total_expenses or 0) if row else 0
                business_count = (row.count or 0) if row else 0

                return {
                    "total_revenue": total_revenue,
                    "total_expenses": total_expenses,
                    "total_profit": total_revenue - total_expenses,
                    "business_count": business_count,
                }
        except Exception:
            # Minimal fallback: aggregate what we can from mock data
            total_revenue = sum(v.get("revenue", 0) for v in self._business_data.values())
            total_expenses = sum(v.get("expenses", 0) for v in self._business_data.values())
            return {
                "total_revenue": total_revenue,
                "total_expenses": total_expenses,
                "total_profit": total_revenue - total_expenses,
                "business_count": len(self._business_data),
            }

    async def create_portfolio(
        self,
        name: str,
        owner_id: str,
        strategy: AllocationStrategy = AllocationStrategy.EQUAL,
        total_budget: float = 0.0,
    ) -> Portfolio:
        """Create a new portfolio."""
        portfolio = Portfolio(
            id=str(uuid.uuid4()),
            name=name,
            owner_id=owner_id,
            strategy=strategy,
            total_budget=total_budget,
        )
        self._portfolios[portfolio.id] = portfolio
        logger.info(f"Created portfolio: {portfolio.id}")
        return portfolio

    async def get_portfolio(self, portfolio_id: str) -> Optional[Portfolio]:
        """Get a portfolio by ID."""
        return self._portfolios.get(portfolio_id)

    async def get_portfolios_for_owner(self, owner_id: str) -> list[Portfolio]:
        """Get all portfolios for an owner."""
        return [p for p in self._portfolios.values() if p.owner_id == owner_id]

    async def update_portfolio(
        self,
        portfolio_id: str,
        name: str = None,
        status: PortfolioStatus = None,
        strategy: AllocationStrategy = None,
        total_budget: float = None,
    ) -> Optional[Portfolio]:
        """Update portfolio settings."""
        portfolio = self._portfolios.get(portfolio_id)
        if not portfolio:
            return None

        if name:
            portfolio.name = name
        if status:
            portfolio.status = status
        if strategy:
            portfolio.strategy = strategy
        if total_budget is not None:
            portfolio.total_budget = total_budget
        
        portfolio.updated_at = datetime.utcnow()
        return portfolio

    async def add_business(
        self, portfolio_id: str, business_id: str, weight: float = None
    ) -> bool:
        """Add a business to a portfolio."""
        portfolio = self._portfolios.get(portfolio_id)
        if not portfolio:
            return False
        
        portfolio.add_business(business_id, weight)
        
        # Rebalance if using equal strategy
        if portfolio.strategy == AllocationStrategy.EQUAL:
            await self._rebalance_equal(portfolio)
        
        return True

    async def remove_business(self, portfolio_id: str, business_id: str) -> bool:
        """Remove a business from a portfolio."""
        portfolio = self._portfolios.get(portfolio_id)
        if not portfolio:
            return False
        
        portfolio.remove_business(business_id)
        
        # Rebalance remaining
        if portfolio.strategy == AllocationStrategy.EQUAL:
            await self._rebalance_equal(portfolio)
        
        return True

    async def set_allocation(
        self,
        portfolio_id: str,
        business_id: str,
        weight: float = None,
        priority: int = None,
        locked: bool = None,
    ) -> bool:
        """Set allocation for a business."""
        portfolio = self._portfolios.get(portfolio_id)
        if not portfolio or business_id not in portfolio.business_ids:
            return False

        alloc = portfolio.allocations.get(business_id)
        if not alloc:
            return False

        if weight is not None:
            alloc.weight = weight
            alloc.budget_percent = weight * 100
            alloc.attention_hours = weight * 40
        if priority is not None:
            alloc.priority = priority
        if locked is not None:
            alloc.locked = locked

        portfolio.updated_at = datetime.utcnow()
        return True

    async def calculate_metrics(self, portfolio_id: str) -> Optional[PortfolioMetrics]:
        """Calculate aggregated portfolio metrics."""
        portfolio = self._portfolios.get(portfolio_id)
        if not portfolio:
            return None

        metrics = PortfolioMetrics()
        performances = []

        for biz_id in portfolio.business_ids:
            biz_data = await self._get_business_metrics(biz_id)
            if biz_data:
                metrics.total_revenue += biz_data.get("revenue", 0)
                metrics.total_expenses += biz_data.get("expenses", 0)
                metrics.total_customers += biz_data.get("customers", 0)
                
                perf = BusinessPerformance(
                    business_id=biz_id,
                    business_name=biz_data.get("name", "Unknown"),
                    revenue=biz_data.get("revenue", 0),
                    profit=biz_data.get("profit", 0),
                    margin=biz_data.get("margin", 0),
                    growth_rate=biz_data.get("growth_rate", 0),
                    health_score=biz_data.get("health_score", 50),
                    stage=biz_data.get("stage", "unknown"),
                )
                performances.append(perf)

        metrics.total_profit = metrics.total_revenue - metrics.total_expenses
        metrics.active_businesses = len(portfolio.business_ids)

        if performances:
            metrics.average_margin = sum(p.margin for p in performances) / len(performances)
            metrics.growth_rate = sum(p.growth_rate for p in performances) / len(performances)
            
            # Find best and worst performers
            sorted_perf = sorted(performances, key=lambda x: x.profit, reverse=True)
            if sorted_perf:
                metrics.best_performer_id = sorted_perf[0].business_id
                metrics.worst_performer_id = sorted_perf[-1].business_id

        return metrics

    async def get_performance_ranking(
        self, portfolio_id: str
    ) -> list[BusinessPerformance]:
        """Get businesses ranked by performance."""
        portfolio = self._portfolios.get(portfolio_id)
        if not portfolio:
            return []

        performances = []
        for biz_id in portfolio.business_ids:
            biz_data = await self._get_business_metrics(biz_id)
            if biz_data:
                perf = BusinessPerformance(
                    business_id=biz_id,
                    business_name=biz_data.get("name", "Unknown"),
                    revenue=biz_data.get("revenue", 0),
                    profit=biz_data.get("profit", 0),
                    margin=biz_data.get("margin", 0),
                    growth_rate=biz_data.get("growth_rate", 0),
                    health_score=biz_data.get("health_score", 50),
                    stage=biz_data.get("stage", "unknown"),
                )
                performances.append(perf)

        # Rank by composite score
        for i, perf in enumerate(sorted(
            performances,
            key=lambda x: (x.profit * PROFIT_RANKING_WEIGHT + x.growth_rate * GROWTH_RANKING_WEIGHT + x.health_score * HEALTH_RANKING_WEIGHT),
            reverse=True
        )):
            perf.rank = i + 1

        return sorted(performances, key=lambda x: x.rank)

    async def recommend_rebalance(self, portfolio_id: str) -> Optional[RebalanceReport]:
        """Generate rebalancing recommendations."""
        portfolio = self._portfolios.get(portfolio_id)
        if not portfolio:
            return None

        report = RebalanceReport(
            portfolio_id=portfolio_id,
            timestamp=datetime.utcnow(),
            strategy_used=portfolio.strategy,
        )

        performances = await self.get_performance_ranking(portfolio_id)
        if not performances:
            return report

        # Calculate recommended weights based on strategy
        if portfolio.strategy == AllocationStrategy.PERFORMANCE:
            weights = await self._calculate_performance_weights(performances)
        elif portfolio.strategy == AllocationStrategy.GROWTH:
            weights = await self._calculate_growth_weights(performances)
        elif portfolio.strategy == AllocationStrategy.REVENUE:
            weights = await self._calculate_revenue_weights(performances)
        else:
            weights = {p.business_id: 1.0 / len(performances) for p in performances}

        for perf in performances:
            current = portfolio.allocations.get(perf.business_id)
            if not current:
                continue

            recommended = weights.get(perf.business_id, current.weight)
            diff = recommended - current.weight

            if current.locked:
                action = "maintain"
                reason = "Allocation is locked"
            elif abs(diff) < REBALANCE_THRESHOLD:
                action = "maintain"
                reason = "Within acceptable range"
            elif diff > 0:
                action = "increase"
                reason = f"Performance rank: {perf.rank}"
            else:
                action = "decrease"
                reason = f"Underperforming (rank {perf.rank})"

            report.actions.append(RebalanceAction(
                business_id=perf.business_id,
                action=action,
                current_weight=current.weight,
                recommended_weight=recommended,
                reason=reason,
            ))
            report.total_reallocation += abs(diff)

        return report

    async def apply_rebalance(self, report: RebalanceReport) -> bool:
        """Apply a rebalancing report."""
        portfolio = self._portfolios.get(report.portfolio_id)
        if not portfolio:
            return False

        for action in report.actions:
            if action.action != "maintain":
                alloc = portfolio.allocations.get(action.business_id)
                if alloc and not alloc.locked:
                    alloc.weight = action.recommended_weight
                    alloc.budget_percent = action.recommended_weight * 100
                    alloc.attention_hours = action.recommended_weight * 40

        report.applied = True
        portfolio.updated_at = datetime.utcnow()
        return True

    async def _rebalance_equal(self, portfolio: Portfolio):
        """Rebalance to equal weights."""
        if not portfolio.business_ids:
            return
        
        weight = 1.0 / len(portfolio.business_ids)
        for biz_id in portfolio.business_ids:
            alloc = portfolio.allocations.get(biz_id)
            if alloc and not alloc.locked:
                alloc.weight = weight
                alloc.budget_percent = weight * 100
                alloc.attention_hours = weight * 40

    async def _calculate_performance_weights(
        self, performances: list[BusinessPerformance]
    ) -> dict[str, float]:
        """Calculate weights based on performance."""
        total_score = sum(
            p.profit * PROFIT_WEIGHT + p.health_score * HEALTH_SCORE_WEIGHT
            for p in performances
        )
        if total_score <= 0:
            return {p.business_id: 1.0 / len(performances) for p in performances}
        
        return {
            p.business_id: (p.profit * PROFIT_WEIGHT + p.health_score * HEALTH_SCORE_WEIGHT) / total_score
            for p in performances
        }

    async def _calculate_growth_weights(
        self, performances: list[BusinessPerformance]
    ) -> dict[str, float]:
        """Calculate weights based on growth potential."""
        # Favor high growth + early stage
        scores = {}
        for p in performances:
            stage_bonus = STAGE_BONUSES.get(p.stage, 1.0)
            scores[p.business_id] = max(0, p.growth_rate) * stage_bonus + 10
        
        total = sum(scores.values())
        return {bid: s / total for bid, s in scores.items()}

    async def _calculate_revenue_weights(
        self, performances: list[BusinessPerformance]
    ) -> dict[str, float]:
        """Calculate weights based on revenue."""
        total = sum(max(0, p.revenue) for p in performances)
        if total <= 0:
            return {p.business_id: 1.0 / len(performances) for p in performances}
        return {p.business_id: max(0, p.revenue) / total for p in performances}

    async def _get_business_metrics(self, business_id: str) -> dict:
        """Get metrics for a business (mock implementation)."""
        # In real implementation, fetch from business unit or analytics
        return self._business_data.get(business_id, {
            "name": f"Business {business_id[:8]}",
            "revenue": 10000,
            "expenses": 7000,
            "profit": 3000,
            "margin": 30,
            "customers": 100,
            "growth_rate": 5.0,
            "health_score": 70,
            "stage": "growth",
        })

    def set_business_data(self, business_id: str, data: dict):
        """Set mock business data for testing."""
        self._business_data[business_id] = data
