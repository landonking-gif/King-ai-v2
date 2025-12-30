"""
Banking Agent - Account management, transactions, cash flow analysis.
"""
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Optional
from collections import defaultdict
from src.agents.base import SubAgent
from src.integrations.plaid_client import (
    PlaidClient, PlaidEnvironment, PlaidAccount, PlaidTransaction,
    TransactionCategory
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CashFlowSummary:
    period_start: date
    period_end: date
    total_income: float
    total_expenses: float
    net_cash_flow: float
    income_by_category: dict[str, float]
    expenses_by_category: dict[str, float]
    daily_balances: list[dict]


@dataclass
class FinancialHealth:
    score: int  # 0-100
    cash_reserve_days: float
    income_stability: float
    expense_ratio: float
    recommendations: list[str]


class BankingAgent(SubAgent):
    """Agent for banking operations and financial analysis."""

    name = "banking"
    description = "Manages banking operations, account linking, transaction analysis, and financial health"

    def __init__(self):
        super().__init__()
        self._clients: dict[str, PlaidClient] = {}
        self._access_tokens: dict[str, dict[str, str]] = {}  # business_id -> {item_id: token}

    async def initialize_plaid(
        self,
        business_id: str,
        client_id: str,
        secret: str,
        environment: str = "sandbox",
    ) -> bool:
        """Initialize Plaid client for a business."""
        try:
            env = PlaidEnvironment(environment)
            self._clients[business_id] = PlaidClient(client_id, secret, env)
            self._access_tokens[business_id] = {}
            logger.info(f"Initialized Plaid for business {business_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Plaid: {e}")
            return False

    def _get_client(self, business_id: str) -> Optional[PlaidClient]:
        return self._clients.get(business_id)

    # Link Flow
    async def create_link_token(
        self, business_id: str, user_id: str, products: list[str] = None
    ) -> dict:
        """Create link token for Plaid Link."""
        client = self._get_client(business_id)
        if not client:
            return {"success": False, "error": "Plaid not configured"}
        
        try:
            result = await client.create_link_token(user_id, products)
            return {"success": True, "output": result}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def exchange_token(
        self, business_id: str, public_token: str
    ) -> dict:
        """Exchange public token for access token."""
        client = self._get_client(business_id)
        if not client:
            return {"success": False, "error": "Plaid not configured"}
        
        try:
            result = await client.exchange_public_token(public_token)
            item_id = result["item_id"]
            access_token = result["access_token"]
            
            if business_id not in self._access_tokens:
                self._access_tokens[business_id] = {}
            self._access_tokens[business_id][item_id] = access_token
            
            return {
                "success": True,
                "output": "Bank account connected",
                "metadata": {"item_id": item_id}
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # Account Operations
    async def get_accounts(self, business_id: str) -> dict:
        """Get all connected bank accounts."""
        client = self._get_client(business_id)
        if not client:
            return {"success": False, "error": "Plaid not configured"}
        
        all_accounts = []
        tokens = self._access_tokens.get(business_id, {})
        
        for item_id, token in tokens.items():
            try:
                accounts = await client.get_accounts(token)
                for acc in accounts:
                    all_accounts.append({
                        "item_id": item_id,
                        "account_id": acc.account_id,
                        "name": acc.name,
                        "type": acc.account_type.value,
                        "mask": acc.mask,
                        "balance": acc.current_balance,
                        "available": acc.available_balance,
                        "institution": acc.institution_name,
                    })
            except Exception as e:
                logger.error(f"Error fetching accounts for {item_id}: {e}")
        
        return {"success": True, "output": {"accounts": all_accounts}}

    async def get_balances(self, business_id: str) -> dict:
        """Get real-time balances for all accounts."""
        client = self._get_client(business_id)
        if not client:
            return {"success": False, "error": "Plaid not configured"}
        
        total_balance = 0.0
        total_available = 0.0
        accounts = []
        tokens = self._access_tokens.get(business_id, {})
        
        for item_id, token in tokens.items():
            try:
                accs = await client.get_balance(token)
                for acc in accs:
                    total_balance += acc.current_balance
                    total_available += acc.available_balance or 0
                    accounts.append({
                        "account_id": acc.account_id,
                        "name": acc.name,
                        "balance": acc.current_balance,
                        "available": acc.available_balance,
                    })
            except Exception as e:
                logger.error(f"Error fetching balance: {e}")
        
        return {
            "success": True,
            "output": {
                "total_balance": total_balance,
                "total_available": total_available,
                "accounts": accounts,
            }
        }

    # Transactions
    async def get_transactions(
        self,
        business_id: str,
        days: int = 30,
        account_id: str = None,
    ) -> dict:
        """Get recent transactions."""
        client = self._get_client(business_id)
        if not client:
            return {"success": False, "error": "Plaid not configured"}
        
        end_date = date.today()
        start_date = end_date - timedelta(days=days)
        
        all_transactions = []
        tokens = self._access_tokens.get(business_id, {})
        
        for item_id, token in tokens.items():
            try:
                account_ids = [account_id] if account_id else None
                txns, total = await client.get_transactions(
                    token, start_date, end_date, account_ids
                )
                for txn in txns:
                    all_transactions.append({
                        "id": txn.transaction_id,
                        "account_id": txn.account_id,
                        "amount": txn.amount,
                        "date": txn.date.isoformat(),
                        "name": txn.name,
                        "merchant": txn.merchant_name,
                        "category": txn.category.value,
                        "is_expense": txn.is_expense,
                        "pending": txn.pending,
                    })
            except Exception as e:
                logger.error(f"Error fetching transactions: {e}")
        
        all_transactions.sort(key=lambda x: x["date"], reverse=True)
        
        return {
            "success": True,
            "output": {"transactions": all_transactions, "count": len(all_transactions)}
        }

    # Cash Flow Analysis
    async def analyze_cash_flow(
        self, business_id: str, days: int = 30
    ) -> dict:
        """Analyze cash flow for the period."""
        txn_result = await self.get_transactions(business_id, days)
        if not txn_result["success"]:
            return txn_result
        
        transactions = txn_result["output"]["transactions"]
        
        total_income = 0.0
        total_expenses = 0.0
        income_by_cat: dict[str, float] = defaultdict(float)
        expenses_by_cat: dict[str, float] = defaultdict(float)
        
        for txn in transactions:
            if txn["pending"]:
                continue
            
            amount = abs(txn["amount"])
            category = txn["category"]
            
            if txn["is_expense"]:
                total_expenses += amount
                expenses_by_cat[category] += amount
            else:
                total_income += amount
                income_by_cat[category] += amount
        
        summary = CashFlowSummary(
            period_start=date.today() - timedelta(days=days),
            period_end=date.today(),
            total_income=total_income,
            total_expenses=total_expenses,
            net_cash_flow=total_income - total_expenses,
            income_by_category=dict(income_by_cat),
            expenses_by_category=dict(expenses_by_cat),
            daily_balances=[],
        )
        
        return {
            "success": True,
            "output": {
                "period_start": summary.period_start.isoformat(),
                "period_end": summary.period_end.isoformat(),
                "total_income": summary.total_income,
                "total_expenses": summary.total_expenses,
                "net_cash_flow": summary.net_cash_flow,
                "income_by_category": summary.income_by_category,
                "expenses_by_category": summary.expenses_by_category,
            }
        }

    async def get_financial_health(self, business_id: str) -> dict:
        """Calculate financial health score."""
        balance_result = await self.get_balances(business_id)
        cash_flow_result = await self.analyze_cash_flow(business_id, 90)
        
        if not balance_result["success"] or not cash_flow_result["success"]:
            return {"success": False, "error": "Could not gather financial data"}
        
        total_balance = balance_result["output"]["total_balance"]
        monthly_expenses = cash_flow_result["output"]["total_expenses"] / 3
        monthly_income = cash_flow_result["output"]["total_income"] / 3
        
        # Calculate metrics
        cash_reserve_days = (total_balance / (monthly_expenses / 30)) if monthly_expenses > 0 else 365
        expense_ratio = (monthly_expenses / monthly_income * 100) if monthly_income > 0 else 100
        
        # Score calculation (0-100)
        score = 50
        if cash_reserve_days >= 90:
            score += 20
        elif cash_reserve_days >= 30:
            score += 10
        
        if expense_ratio < 70:
            score += 20
        elif expense_ratio < 90:
            score += 10
        
        if cash_flow_result["output"]["net_cash_flow"] > 0:
            score += 10
        
        recommendations = []
        if cash_reserve_days < 30:
            recommendations.append("Build emergency fund - aim for 3 months expenses")
        if expense_ratio > 80:
            recommendations.append("Review expenses - spending ratio is high")
        if cash_flow_result["output"]["net_cash_flow"] < 0:
            recommendations.append("Address negative cash flow - expenses exceed income")
        
        health = FinancialHealth(
            score=min(100, max(0, score)),
            cash_reserve_days=cash_reserve_days,
            income_stability=0.8,  # Placeholder
            expense_ratio=expense_ratio,
            recommendations=recommendations,
        )
        
        return {
            "success": True,
            "output": {
                "score": health.score,
                "cash_reserve_days": round(health.cash_reserve_days, 1),
                "expense_ratio": round(health.expense_ratio, 1),
                "recommendations": health.recommendations,
            }
        }

    async def disconnect_bank(self, business_id: str, item_id: str) -> dict:
        """Disconnect a bank account."""
        client = self._get_client(business_id)
        if not client:
            return {"success": False, "error": "Plaid not configured"}
        
        tokens = self._access_tokens.get(business_id, {})
        token = tokens.get(item_id)
        if not token:
            return {"success": False, "error": "Bank not connected"}
        
        try:
            await client.remove_item(token)
            del self._access_tokens[business_id][item_id]
            return {"success": True, "output": "Bank disconnected"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def execute(self, task: dict) -> dict:
        """Execute banking tasks based on action type."""
        action = task.get("input", {}).get("action", "")
        business_id = task.get("input", {}).get("business_id", "")
        
        if action == "link":
            return await self.create_link_token(business_id, task.get("input", {}).get("user_id", ""))
        elif action == "exchange":
            return await self.exchange_token(business_id, task["input"]["public_token"])
        elif action == "accounts":
            return await self.get_accounts(business_id)
        elif action == "balances":
            return await self.get_balances(business_id)
        elif action == "transactions":
            return await self.get_transactions(business_id, task.get("input", {}).get("days", 30))
        elif action == "cash_flow":
            return await self.analyze_cash_flow(business_id, task.get("input", {}).get("days", 30))
        elif action == "health":
            return await self.get_financial_health(business_id)
        
        return {"success": False, "error": f"Unknown action: {action}"}
