"""Tests for Banking Agent and Plaid Client."""
import pytest
import sys
import os
from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

# Set up test environment before importing modules
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://test:test@localhost:5432/testdb")
os.environ.setdefault("OLLAMA_URL", "http://localhost:11434")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379")

from src.integrations.plaid_client import (
    PlaidClient, PlaidEnvironment, PlaidAccount, PlaidTransaction,
    AccountType, TransactionCategory
)


@pytest.fixture
def plaid_client():
    return PlaidClient("client_id", "secret", PlaidEnvironment.SANDBOX)


@pytest.fixture
def banking_agent():
    # Import here after env vars are set
    from src.agents.banking import BankingAgent
    return BankingAgent()


@pytest.fixture
def sample_account():
    return PlaidAccount(
        account_id="acc_123",
        name="Checking",
        official_name="Business Checking",
        account_type=AccountType.CHECKING,
        subtype="checking",
        mask="1234",
        current_balance=5000.00,
        available_balance=4500.00,
    )


@pytest.fixture
def sample_transaction():
    return PlaidTransaction(
        transaction_id="txn_123",
        account_id="acc_123",
        amount=50.00,
        date=date.today(),
        name="Coffee Shop",
        merchant_name="Starbucks",
        category=TransactionCategory.FOOD,
    )


class TestPlaidClient:
    def test_init(self, plaid_client):
        assert plaid_client.client_id == "client_id"
        assert plaid_client.environment == PlaidEnvironment.SANDBOX

    def test_parse_account_type(self, plaid_client):
        assert plaid_client._parse_account_type("depository") == AccountType.CHECKING
        assert plaid_client._parse_account_type("credit") == AccountType.CREDIT
        assert plaid_client._parse_account_type("unknown") == AccountType.OTHER

    def test_parse_category(self, plaid_client):
        assert plaid_client._parse_category(["Food and Drink"]) == TransactionCategory.FOOD
        assert plaid_client._parse_category(["Travel"]) == TransactionCategory.TRAVEL
        assert plaid_client._parse_category([]) == TransactionCategory.OTHER


class TestPlaidTransaction:
    def test_is_expense(self, sample_transaction):
        assert sample_transaction.is_expense is True
        assert sample_transaction.is_income is False

    def test_is_income(self):
        txn = PlaidTransaction(
            transaction_id="txn_456",
            account_id="acc_123",
            amount=-1000.00,  # Negative = credit/income
            date=date.today(),
            name="Payroll",
            merchant_name=None,
            category=TransactionCategory.INCOME,
        )
        assert txn.is_income is True
        assert txn.is_expense is False


class TestBankingAgent:
    @pytest.mark.asyncio
    async def test_initialize_plaid(self, banking_agent):
        result = await banking_agent.initialize_plaid(
            "biz_1", "client_id", "secret", "sandbox"
        )
        assert result is True
        assert "biz_1" in banking_agent._clients

    @pytest.mark.asyncio
    async def test_no_client_configured(self, banking_agent):
        result = await banking_agent.get_accounts("biz_unknown")
        assert not result["success"]
        assert "not configured" in result["error"]

    @pytest.mark.asyncio
    async def test_get_accounts_success(self, banking_agent, sample_account):
        await banking_agent.initialize_plaid("biz_1", "id", "secret")
        banking_agent._access_tokens["biz_1"] = {"item_1": "token_123"}
        
        banking_agent._clients["biz_1"].get_accounts = AsyncMock(
            return_value=[sample_account]
        )
        
        result = await banking_agent.get_accounts("biz_1")
        assert result["success"]
        assert len(result["output"]["accounts"]) == 1
        assert result["output"]["accounts"][0]["balance"] == 5000.00

    @pytest.mark.asyncio
    async def test_analyze_cash_flow(self, banking_agent):
        await banking_agent.initialize_plaid("biz_1", "id", "secret")
        
        # Mock get_transactions to return test data
        async def mock_get_transactions(*args, **kwargs):
            return {
                "success": True,
                "output": {
                    "transactions": [
                        {"amount": 100, "is_expense": True, "category": "food", "pending": False},
                        {"amount": -500, "is_expense": False, "category": "income", "pending": False},
                    ],
                    "count": 2
                }
            }
        
        banking_agent.get_transactions = mock_get_transactions
        
        result = await banking_agent.analyze_cash_flow("biz_1", 30)
        assert result["success"]
        assert result["output"]["total_income"] == 500
        assert result["output"]["total_expenses"] == 100
        assert result["output"]["net_cash_flow"] == 400

    @pytest.mark.asyncio
    async def test_execute_unknown_action(self, banking_agent):
        result = await banking_agent.execute({
            "input": {"action": "unknown", "business_id": "biz_1"}
        })
        assert not result["success"]
        assert "Unknown action" in result["error"]
