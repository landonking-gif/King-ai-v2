"""Tests for Finance Agent and Stripe Client."""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime
import sys

# Mock config before importing anything that depends on it
mock_settings = MagicMock()
mock_settings.ollama_url = "http://localhost:11434"
mock_settings.ollama_model = "llama3.1:8b"
sys.modules['config.settings'] = MagicMock(settings=mock_settings)

from src.integrations.stripe_client import (
    StripeClient, StripeCustomer, PaymentIntent, Subscription,
    PaymentStatus, SubscriptionStatus
)
from src.agents.finance import FinanceAgent


@pytest.fixture
def stripe_client():
    return StripeClient("sk_test_key", "whsec_test")


@pytest.fixture
def finance_agent():
    return FinanceAgent()


class TestStripeClient:
    def test_init(self, stripe_client):
        assert stripe_client.api_key == "sk_test_key"
        assert stripe_client.webhook_secret == "whsec_test"

    def test_parse_customer(self, stripe_client):
        data = {"id": "cus_123", "email": "test@example.com", "name": "Test", "created": 1609459200}
        customer = stripe_client._parse_customer(data)
        assert customer.id == "cus_123"
        assert customer.email == "test@example.com"
        assert customer.name == "Test"

    def test_parse_payment_intent(self, stripe_client):
        data = {"id": "pi_123", "amount": 1000, "currency": "usd", "status": "succeeded", "created": 1609459200}
        intent = stripe_client._parse_payment_intent(data)
        assert intent.id == "pi_123"
        assert intent.amount_dollars == 10.0
        assert intent.status == PaymentStatus.SUCCEEDED

    def test_parse_subscription(self, stripe_client):
        data = {
            "id": "sub_123",
            "customer": "cus_123",
            "status": "active",
            "items": {"data": [{"price": {"id": "price_123"}}]},
            "current_period_start": 1609459200,
            "current_period_end": 1612137600,
        }
        sub = stripe_client._parse_subscription(data)
        assert sub.id == "sub_123"
        assert sub.customer_id == "cus_123"
        assert sub.status == SubscriptionStatus.ACTIVE
        assert sub.price_id == "price_123"

    def test_amount_dollars_property(self):
        intent = PaymentIntent(
            id="pi_123",
            amount=5000,
            currency="usd",
            status=PaymentStatus.SUCCEEDED
        )
        assert intent.amount_dollars == 50.0


class TestFinanceAgent:
    @pytest.mark.asyncio
    async def test_initialize_stripe(self, finance_agent):
        result = await finance_agent.initialize_stripe("biz_1", "sk_test")
        assert result is True
        assert "biz_1" in finance_agent._clients

    @pytest.mark.asyncio
    async def test_no_client_configured(self, finance_agent):
        result = await finance_agent.create_customer("biz_unknown", "test@example.com")
        assert not result["success"]
        assert "not configured" in result["error"]

    @pytest.mark.asyncio
    async def test_create_customer_success(self, finance_agent):
        await finance_agent.initialize_stripe("biz_1", "sk_test")
        
        mock_customer = StripeCustomer(id="cus_123", email="test@example.com")
        finance_agent._clients["biz_1"].create_customer = AsyncMock(return_value=mock_customer)
        
        result = await finance_agent.create_customer("biz_1", "test@example.com", "Test User")
        assert result["success"]
        assert result["data"]["customer_id"] == "cus_123"

    @pytest.mark.asyncio
    async def test_create_payment_success(self, finance_agent):
        await finance_agent.initialize_stripe("biz_1", "sk_test")
        
        mock_intent = PaymentIntent(
            id="pi_123", amount=5000, currency="usd", status=PaymentStatus.PENDING
        )
        finance_agent._clients["biz_1"].create_payment_intent = AsyncMock(return_value=mock_intent)
        
        result = await finance_agent.create_payment("biz_1", 50.00)
        assert result["success"]
        assert result["data"]["payment_intent_id"] == "pi_123"
        assert result["data"]["amount"] == 50.0

    @pytest.mark.asyncio
    async def test_create_subscription_success(self, finance_agent):
        await finance_agent.initialize_stripe("biz_1", "sk_test")
        
        mock_sub = Subscription(
            id="sub_123",
            customer_id="cus_123",
            status=SubscriptionStatus.ACTIVE,
            price_id="price_123",
            current_period_start=datetime.now(),
            current_period_end=datetime.now(),
        )
        finance_agent._clients["biz_1"].create_subscription = AsyncMock(return_value=mock_sub)
        
        result = await finance_agent.create_subscription("biz_1", "cus_123", "price_123")
        assert result["success"]
        assert result["data"]["subscription_id"] == "sub_123"

    @pytest.mark.asyncio
    async def test_get_balance_success(self, finance_agent):
        await finance_agent.initialize_stripe("biz_1", "sk_test")
        
        mock_balance = {
            "available": [{"amount": 10000}],
            "pending": [{"amount": 5000}]
        }
        finance_agent._clients["biz_1"].get_balance = AsyncMock(return_value=mock_balance)
        
        result = await finance_agent.get_balance("biz_1")
        assert result["success"]
        assert result["data"]["available"] == 100.0
        assert result["data"]["pending"] == 50.0

    @pytest.mark.asyncio
    async def test_traditional_analysis_fallback(self, finance_agent):
        """Test that traditional financial analysis still works without Stripe"""
        with patch.object(finance_agent, '_ask_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "Financial analysis complete"
            
            result = await finance_agent.execute({
                "description": "Analyze Q4 revenue",
                "input_data": {"revenue": 100000}
            })
            
            assert result["success"]
            assert "output" in result
            mock_llm.assert_called_once()

    @pytest.mark.asyncio
    async def test_stripe_action_execution(self, finance_agent):
        """Test Stripe action routing through execute"""
        await finance_agent.initialize_stripe("biz_1", "sk_test")
        
        mock_balance = {
            "available": [{"amount": 10000}],
            "pending": [{"amount": 0}]
        }
        finance_agent._clients["biz_1"].get_balance = AsyncMock(return_value=mock_balance)
        
        result = await finance_agent.execute({
            "action": "balance",
            "business_id": "biz_1"
        })
        
        assert result["success"]
        assert result["data"]["available"] == 100.0
