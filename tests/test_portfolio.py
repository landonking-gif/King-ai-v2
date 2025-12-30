"""Tests for Portfolio Management."""
import pytest
from src.business.portfolio_models import (
    Portfolio, PortfolioStatus, AllocationStrategy
)
from src.business.portfolio import PortfolioManager


@pytest.fixture
def manager():
    return PortfolioManager()


class TestPortfolioManager:
    @pytest.mark.asyncio
    async def test_create_portfolio(self, manager):
        portfolio = await manager.create_portfolio(
            name="Test Portfolio",
            owner_id="user_1",
        )
        
        assert portfolio.id is not None
        assert portfolio.name == "Test Portfolio"
        assert portfolio.status == PortfolioStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_add_business(self, manager):
        portfolio = await manager.create_portfolio("Test", "user_1")
        
        success = await manager.add_business(portfolio.id, "biz_1")
        
        assert success is True
        assert "biz_1" in portfolio.business_ids

    @pytest.mark.asyncio
    async def test_equal_allocation(self, manager):
        portfolio = await manager.create_portfolio(
            "Test", "user_1", AllocationStrategy.EQUAL
        )
        
        await manager.add_business(portfolio.id, "biz_1")
        await manager.add_business(portfolio.id, "biz_2")
        
        assert portfolio.allocations["biz_1"].weight == 0.5
        assert portfolio.allocations["biz_2"].weight == 0.5

    @pytest.mark.asyncio
    async def test_remove_business(self, manager):
        portfolio = await manager.create_portfolio("Test", "user_1")
        await manager.add_business(portfolio.id, "biz_1")
        
        success = await manager.remove_business(portfolio.id, "biz_1")
        
        assert success is True
        assert "biz_1" not in portfolio.business_ids

    @pytest.mark.asyncio
    async def test_calculate_metrics(self, manager):
        portfolio = await manager.create_portfolio("Test", "user_1")
        await manager.add_business(portfolio.id, "biz_1")
        
        manager.set_business_data("biz_1", {
            "name": "Business 1",
            "revenue": 50000,
            "expenses": 30000,
            "profit": 20000,
            "margin": 40,
            "customers": 500,
            "growth_rate": 10,
            "health_score": 85,
            "stage": "growth",
        })
        
        metrics = await manager.calculate_metrics(portfolio.id)
        
        assert metrics.total_revenue == 50000
        assert metrics.total_profit == 20000
        assert metrics.total_customers == 500

    @pytest.mark.asyncio
    async def test_performance_ranking(self, manager):
        portfolio = await manager.create_portfolio("Test", "user_1")
        await manager.add_business(portfolio.id, "biz_1")
        await manager.add_business(portfolio.id, "biz_2")
        
        manager.set_business_data("biz_1", {
            "name": "Business 1",
            "revenue": 10000,
            "profit": 5000,
            "margin": 50,
            "growth_rate": 5,
            "health_score": 70,
            "stage": "growth",
        })
        manager.set_business_data("biz_2", {
            "name": "Business 2",
            "revenue": 20000,
            "profit": 8000,
            "margin": 40,
            "growth_rate": 15,
            "health_score": 90,
            "stage": "scaling",
        })
        
        rankings = await manager.get_performance_ranking(portfolio.id)
        
        assert len(rankings) == 2
        assert rankings[0].rank == 1

    @pytest.mark.asyncio
    async def test_recommend_rebalance(self, manager):
        portfolio = await manager.create_portfolio(
            "Test", "user_1", AllocationStrategy.PERFORMANCE
        )
        await manager.add_business(portfolio.id, "biz_1")
        await manager.add_business(portfolio.id, "biz_2")
        
        report = await manager.recommend_rebalance(portfolio.id)
        
        assert report is not None
        assert len(report.actions) == 2

    @pytest.mark.asyncio
    async def test_locked_allocation(self, manager):
        portfolio = await manager.create_portfolio("Test", "user_1")
        await manager.add_business(portfolio.id, "biz_1")
        
        await manager.set_allocation(portfolio.id, "biz_1", locked=True)
        
        assert portfolio.allocations["biz_1"].locked is True

    @pytest.mark.asyncio
    async def test_get_portfolios_for_owner(self, manager):
        await manager.create_portfolio("P1", "user_1")
        await manager.create_portfolio("P2", "user_1")
        await manager.create_portfolio("P3", "user_2")
        
        portfolios = await manager.get_portfolios_for_owner("user_1")
        
        assert len(portfolios) == 2
