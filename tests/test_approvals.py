"""Tests for Approval System."""
import pytest
from src.approvals.models import ApprovalStatus, ApprovalType, RiskLevel, RiskFactor
from src.approvals.manager import ApprovalManager


@pytest.fixture
def manager():
    return ApprovalManager()


class TestApprovalManager:
    @pytest.mark.asyncio
    async def test_create_request(self, manager):
        request = await manager.create_request(
            business_id="biz_1",
            action_type=ApprovalType.FINANCIAL,
            title="Large Purchase",
            description="Purchase inventory",
            payload={"amount": 5000},
            risk_level=RiskLevel.HIGH,
        )
        
        assert request.id is not None
        assert request.status == ApprovalStatus.PENDING

    @pytest.mark.asyncio
    async def test_auto_approve(self, manager):
        request = await manager.create_request(
            business_id="biz_1",
            action_type=ApprovalType.FINANCIAL,
            title="Small Purchase",
            description="Minor expense",
            payload={"amount": 50},  # Below auto-approve threshold
            risk_level=RiskLevel.LOW,
        )
        
        assert request.status == ApprovalStatus.APPROVED

    @pytest.mark.asyncio
    async def test_approve_request(self, manager):
        request = await manager.create_request(
            business_id="biz_1",
            action_type=ApprovalType.OPERATIONAL,
            title="Test Action",
            description="Test",
            payload={},
            risk_level=RiskLevel.MEDIUM,
        )
        
        approved = await manager.approve(request.id, "user_1", "Looks good")
        
        assert approved.status == ApprovalStatus.APPROVED
        assert approved.reviewed_by == "user_1"

    @pytest.mark.asyncio
    async def test_reject_request(self, manager):
        request = await manager.create_request(
            business_id="biz_1",
            action_type=ApprovalType.EXTERNAL,
            title="Test Action",
            description="Test",
            payload={},
        )
        
        rejected = await manager.reject(request.id, "user_1", "Too risky")
        
        assert rejected.status == ApprovalStatus.REJECTED

    @pytest.mark.asyncio
    async def test_get_pending(self, manager):
        await manager.create_request(
            business_id="biz_1",
            action_type=ApprovalType.FINANCIAL,
            title="Action 1",
            description="",
            payload={"amount": 500},  # Above auto-approve threshold
        )
        await manager.create_request(
            business_id="biz_1",
            action_type=ApprovalType.LEGAL,
            title="Action 2",
            description="",
            payload={},
        )
        
        pending = await manager.get_pending("biz_1")
        
        assert len(pending) == 2

    @pytest.mark.asyncio
    async def test_get_stats(self, manager):
        await manager.create_request(
            business_id="biz_1",
            action_type=ApprovalType.OPERATIONAL,
            title="Test",
            description="",
            payload={},
        )
        
        stats = await manager.get_stats("biz_1")
        
        assert stats["pending"] == 1
        assert stats["total"] == 1
