# Implementation Plan Part 16: Finance - Plaid Banking Integration

| Field | Value |
|-------|-------|
| Module | Plaid Banking & Financial Data Integration |
| Priority | High |
| Estimated Effort | 4-5 hours |
| Dependencies | Part 3 (Database), Part 15 (Stripe Finance) |

---

## 1. Scope

This module implements Plaid banking integration for financial insights:

- **Plaid Client** - Link tokens, account access, transactions
- **Account Management** - Connect bank accounts, view balances
- **Transaction Sync** - Fetch and categorize transactions
- **Cash Flow Analysis** - Income/expense tracking, projections
- **Financial Health** - Business financial health scoring

---

## 2. Tasks

### Task 16.1: Plaid Client

**File: `src/integrations/plaid_client.py`**

```python
"""
Plaid Banking Integration Client.
"""
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from enum import Enum
from typing import Any, Optional
import httpx
from src.utils.logging import get_logger

logger = get_logger(__name__)


class PlaidEnvironment(Enum):
    SANDBOX = "sandbox"
    DEVELOPMENT = "development"
    PRODUCTION = "production"


class AccountType(Enum):
    CHECKING = "checking"
    SAVINGS = "savings"
    CREDIT = "credit"
    LOAN = "loan"
    INVESTMENT = "investment"
    OTHER = "other"


class TransactionCategory(Enum):
    INCOME = "income"
    TRANSFER = "transfer"
    PAYMENT = "payment"
    FOOD = "food"
    SHOPPING = "shopping"
    TRAVEL = "travel"
    UTILITIES = "utilities"
    SERVICES = "services"
    OTHER = "other"


@dataclass
class PlaidAccount:
    account_id: str
    name: str
    official_name: Optional[str]
    account_type: AccountType
    subtype: str
    mask: str  # Last 4 digits
    current_balance: float
    available_balance: Optional[float]
    currency: str = "USD"
    institution_id: Optional[str] = None
    institution_name: Optional[str] = None


@dataclass
class PlaidTransaction:
    transaction_id: str
    account_id: str
    amount: float  # Positive = debit, Negative = credit
    date: date
    name: str
    merchant_name: Optional[str]
    category: TransactionCategory
    category_detail: list[str] = field(default_factory=list)
    pending: bool = False
    currency: str = "USD"
    
    @property
    def is_expense(self) -> bool:
        return self.amount > 0
    
    @property
    def is_income(self) -> bool:
        return self.amount < 0


@dataclass
class PlaidInstitution:
    institution_id: str
    name: str
    products: list[str] = field(default_factory=list)
    logo: Optional[str] = None
    primary_color: Optional[str] = None


class PlaidClient:
    """Plaid API client for banking integration."""

    ENVIRONMENTS = {
        PlaidEnvironment.SANDBOX: "https://sandbox.plaid.com",
        PlaidEnvironment.DEVELOPMENT: "https://development.plaid.com",
        PlaidEnvironment.PRODUCTION: "https://production.plaid.com",
    }

    def __init__(
        self,
        client_id: str,
        secret: str,
        environment: PlaidEnvironment = PlaidEnvironment.SANDBOX,
    ):
        self.client_id = client_id
        self.secret = secret
        self.environment = environment
        self.base_url = self.ENVIRONMENTS[environment]
        self.client = httpx.AsyncClient(timeout=30.0)

    async def close(self):
        await self.client.aclose()

    async def _request(self, endpoint: str, data: dict = None) -> dict:
        payload = {
            "client_id": self.client_id,
            "secret": self.secret,
            **(data or {}),
        }
        try:
            resp = await self.client.post(
                f"{self.base_url}{endpoint}",
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"Plaid API error: {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Plaid request failed: {e}")
            raise

    # Link Token Management
    async def create_link_token(
        self,
        user_id: str,
        products: list[str] = None,
        country_codes: list[str] = None,
        language: str = "en",
        redirect_uri: str = None,
    ) -> dict:
        """Create a link token for Plaid Link initialization."""
        data = {
            "user": {"client_user_id": user_id},
            "client_name": "King AI",
            "products": products or ["transactions"],
            "country_codes": country_codes or ["US"],
            "language": language,
        }
        if redirect_uri:
            data["redirect_uri"] = redirect_uri
        
        result = await self._request("/link/token/create", data)
        return {
            "link_token": result["link_token"],
            "expiration": result["expiration"],
        }

    async def exchange_public_token(self, public_token: str) -> dict:
        """Exchange public token for access token."""
        result = await self._request("/item/public_token/exchange", {
            "public_token": public_token,
        })
        return {
            "access_token": result["access_token"],
            "item_id": result["item_id"],
        }

    # Account Management
    async def get_accounts(self, access_token: str) -> list[PlaidAccount]:
        """Get all accounts for an item."""
        result = await self._request("/accounts/get", {
            "access_token": access_token,
        })
        
        item = result.get("item", {})
        institution_id = item.get("institution_id")
        
        accounts = []
        for acc in result.get("accounts", []):
            balances = acc.get("balances", {})
            accounts.append(PlaidAccount(
                account_id=acc["account_id"],
                name=acc["name"],
                official_name=acc.get("official_name"),
                account_type=self._parse_account_type(acc.get("type", "other")),
                subtype=acc.get("subtype", ""),
                mask=acc.get("mask", ""),
                current_balance=balances.get("current", 0) or 0,
                available_balance=balances.get("available"),
                currency=balances.get("iso_currency_code", "USD"),
                institution_id=institution_id,
            ))
        return accounts

    async def get_balance(self, access_token: str) -> list[PlaidAccount]:
        """Get real-time balance for accounts."""
        result = await self._request("/accounts/balance/get", {
            "access_token": access_token,
        })
        
        accounts = []
        for acc in result.get("accounts", []):
            balances = acc.get("balances", {})
            accounts.append(PlaidAccount(
                account_id=acc["account_id"],
                name=acc["name"],
                official_name=acc.get("official_name"),
                account_type=self._parse_account_type(acc.get("type", "other")),
                subtype=acc.get("subtype", ""),
                mask=acc.get("mask", ""),
                current_balance=balances.get("current", 0) or 0,
                available_balance=balances.get("available"),
                currency=balances.get("iso_currency_code", "USD"),
            ))
        return accounts

    # Transactions
    async def get_transactions(
        self,
        access_token: str,
        start_date: date = None,
        end_date: date = None,
        account_ids: list[str] = None,
        count: int = 100,
        offset: int = 0,
    ) -> tuple[list[PlaidTransaction], int]:
        """Get transactions for an item."""
        if not start_date:
            start_date = date.today() - timedelta(days=30)
        if not end_date:
            end_date = date.today()
        
        data = {
            "access_token": access_token,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "options": {
                "count": count,
                "offset": offset,
            },
        }
        if account_ids:
            data["options"]["account_ids"] = account_ids
        
        result = await self._request("/transactions/get", data)
        
        transactions = []
        for txn in result.get("transactions", []):
            transactions.append(PlaidTransaction(
                transaction_id=txn["transaction_id"],
                account_id=txn["account_id"],
                amount=txn["amount"],
                date=date.fromisoformat(txn["date"]),
                name=txn["name"],
                merchant_name=txn.get("merchant_name"),
                category=self._parse_category(txn.get("category", [])),
                category_detail=txn.get("category", []),
                pending=txn.get("pending", False),
                currency=txn.get("iso_currency_code", "USD"),
            ))
        
        return transactions, result.get("total_transactions", len(transactions))

    async def sync_transactions(
        self, access_token: str, cursor: str = None
    ) -> dict:
        """Sync transactions incrementally."""
        data = {"access_token": access_token}
        if cursor:
            data["cursor"] = cursor
        
        result = await self._request("/transactions/sync", data)
        
        added = [self._parse_transaction(t) for t in result.get("added", [])]
        modified = [self._parse_transaction(t) for t in result.get("modified", [])]
        removed = [t["transaction_id"] for t in result.get("removed", [])]
        
        return {
            "added": added,
            "modified": modified,
            "removed": removed,
            "next_cursor": result.get("next_cursor"),
            "has_more": result.get("has_more", False),
        }

    # Institution Info
    async def get_institution(self, institution_id: str) -> PlaidInstitution:
        """Get institution details."""
        result = await self._request("/institutions/get_by_id", {
            "institution_id": institution_id,
            "country_codes": ["US"],
        })
        
        inst = result.get("institution", {})
        return PlaidInstitution(
            institution_id=inst["institution_id"],
            name=inst["name"],
            products=inst.get("products", []),
            logo=inst.get("logo"),
            primary_color=inst.get("primary_color"),
        )

    async def search_institutions(
        self, query: str, products: list[str] = None, limit: int = 10
    ) -> list[PlaidInstitution]:
        """Search for institutions."""
        result = await self._request("/institutions/search", {
            "query": query,
            "products": products or ["transactions"],
            "country_codes": ["US"],
            "options": {"limit": limit},
        })
        
        return [
            PlaidInstitution(
                institution_id=inst["institution_id"],
                name=inst["name"],
                products=inst.get("products", []),
            )
            for inst in result.get("institutions", [])
        ]

    # Item Management
    async def get_item(self, access_token: str) -> dict:
        """Get item info."""
        result = await self._request("/item/get", {"access_token": access_token})
        return result.get("item", {})

    async def remove_item(self, access_token: str) -> bool:
        """Remove an item (disconnect bank)."""
        await self._request("/item/remove", {"access_token": access_token})
        return True

    # Identity (requires identity product)
    async def get_identity(self, access_token: str) -> list[dict]:
        """Get account holder identity info."""
        result = await self._request("/identity/get", {"access_token": access_token})
        return result.get("accounts", [])

    # Parsing helpers
    def _parse_account_type(self, type_str: str) -> AccountType:
        mapping = {
            "depository": AccountType.CHECKING,
            "credit": AccountType.CREDIT,
            "loan": AccountType.LOAN,
            "investment": AccountType.INVESTMENT,
        }
        return mapping.get(type_str, AccountType.OTHER)

    def _parse_category(self, categories: list[str]) -> TransactionCategory:
        if not categories:
            return TransactionCategory.OTHER
        
        primary = categories[0].lower() if categories else ""
        mapping = {
            "income": TransactionCategory.INCOME,
            "transfer": TransactionCategory.TRANSFER,
            "payment": TransactionCategory.PAYMENT,
            "food and drink": TransactionCategory.FOOD,
            "shops": TransactionCategory.SHOPPING,
            "travel": TransactionCategory.TRAVEL,
            "utilities": TransactionCategory.UTILITIES,
            "service": TransactionCategory.SERVICES,
        }
        
        for key, cat in mapping.items():
            if key in primary:
                return cat
        return TransactionCategory.OTHER

    def _parse_transaction(self, txn: dict) -> PlaidTransaction:
        return PlaidTransaction(
            transaction_id=txn["transaction_id"],
            account_id=txn["account_id"],
            amount=txn["amount"],
            date=date.fromisoformat(txn["date"]),
            name=txn["name"],
            merchant_name=txn.get("merchant_name"),
            category=self._parse_category(txn.get("category", [])),
            category_detail=txn.get("category", []),
            pending=txn.get("pending", False),
            currency=txn.get("iso_currency_code", "USD"),
        )
```

---

### Task 16.2: Banking Agent

**File: `src/agents/banking.py`**

```python
"""
Banking Agent - Account management, transactions, cash flow analysis.
"""
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Optional
from collections import defaultdict
from src.agents.base import BaseAgent, AgentCapability, AgentResult
from src.integrations.plaid_client import (
    PlaidClient, PlaidEnvironment, PlaidAccount, PlaidTransaction,
    TransactionCategory
)
from src.database.connection import get_db_session
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


class BankingAgent(BaseAgent):
    """Agent for banking operations and financial analysis."""

    def __init__(self):
        super().__init__(
            name="Banking Agent",
            capabilities=[AgentCapability.ANALYSIS, AgentCapability.COMMERCE]
        )
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
    ) -> AgentResult:
        """Create link token for Plaid Link."""
        client = self._get_client(business_id)
        if not client:
            return AgentResult(success=False, message="Plaid not configured")
        
        try:
            result = await client.create_link_token(user_id, products)
            return AgentResult(success=True, data=result)
        except Exception as e:
            return AgentResult(success=False, message=str(e))

    async def exchange_token(
        self, business_id: str, public_token: str
    ) -> AgentResult:
        """Exchange public token for access token."""
        client = self._get_client(business_id)
        if not client:
            return AgentResult(success=False, message="Plaid not configured")
        
        try:
            result = await client.exchange_public_token(public_token)
            item_id = result["item_id"]
            access_token = result["access_token"]
            
            if business_id not in self._access_tokens:
                self._access_tokens[business_id] = {}
            self._access_tokens[business_id][item_id] = access_token
            
            return AgentResult(
                success=True,
                message="Bank account connected",
                data={"item_id": item_id}
            )
        except Exception as e:
            return AgentResult(success=False, message=str(e))

    # Account Operations
    async def get_accounts(self, business_id: str) -> AgentResult:
        """Get all connected bank accounts."""
        client = self._get_client(business_id)
        if not client:
            return AgentResult(success=False, message="Plaid not configured")
        
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
        
        return AgentResult(success=True, data={"accounts": all_accounts})

    async def get_balances(self, business_id: str) -> AgentResult:
        """Get real-time balances for all accounts."""
        client = self._get_client(business_id)
        if not client:
            return AgentResult(success=False, message="Plaid not configured")
        
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
        
        return AgentResult(
            success=True,
            data={
                "total_balance": total_balance,
                "total_available": total_available,
                "accounts": accounts,
            }
        )

    # Transactions
    async def get_transactions(
        self,
        business_id: str,
        days: int = 30,
        account_id: str = None,
    ) -> AgentResult:
        """Get recent transactions."""
        client = self._get_client(business_id)
        if not client:
            return AgentResult(success=False, message="Plaid not configured")
        
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
        
        return AgentResult(
            success=True,
            data={"transactions": all_transactions, "count": len(all_transactions)}
        )

    # Cash Flow Analysis
    async def analyze_cash_flow(
        self, business_id: str, days: int = 30
    ) -> AgentResult:
        """Analyze cash flow for the period."""
        txn_result = await self.get_transactions(business_id, days)
        if not txn_result.success:
            return txn_result
        
        transactions = txn_result.data["transactions"]
        
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
        
        return AgentResult(
            success=True,
            data={
                "period_start": summary.period_start.isoformat(),
                "period_end": summary.period_end.isoformat(),
                "total_income": summary.total_income,
                "total_expenses": summary.total_expenses,
                "net_cash_flow": summary.net_cash_flow,
                "income_by_category": summary.income_by_category,
                "expenses_by_category": summary.expenses_by_category,
            }
        )

    async def get_financial_health(self, business_id: str) -> AgentResult:
        """Calculate financial health score."""
        balance_result = await self.get_balances(business_id)
        cash_flow_result = await self.analyze_cash_flow(business_id, 90)
        
        if not balance_result.success or not cash_flow_result.success:
            return AgentResult(success=False, message="Could not gather financial data")
        
        total_balance = balance_result.data["total_balance"]
        monthly_expenses = cash_flow_result.data["total_expenses"] / 3
        monthly_income = cash_flow_result.data["total_income"] / 3
        
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
        
        if cash_flow_result.data["net_cash_flow"] > 0:
            score += 10
        
        recommendations = []
        if cash_reserve_days < 30:
            recommendations.append("Build emergency fund - aim for 3 months expenses")
        if expense_ratio > 80:
            recommendations.append("Review expenses - spending ratio is high")
        if cash_flow_result.data["net_cash_flow"] < 0:
            recommendations.append("Address negative cash flow - expenses exceed income")
        
        health = FinancialHealth(
            score=min(100, max(0, score)),
            cash_reserve_days=cash_reserve_days,
            income_stability=0.8,  # Placeholder
            expense_ratio=expense_ratio,
            recommendations=recommendations,
        )
        
        return AgentResult(
            success=True,
            data={
                "score": health.score,
                "cash_reserve_days": round(health.cash_reserve_days, 1),
                "expense_ratio": round(health.expense_ratio, 1),
                "recommendations": health.recommendations,
            }
        )

    async def disconnect_bank(self, business_id: str, item_id: str) -> AgentResult:
        """Disconnect a bank account."""
        client = self._get_client(business_id)
        if not client:
            return AgentResult(success=False, message="Plaid not configured")
        
        tokens = self._access_tokens.get(business_id, {})
        token = tokens.get(item_id)
        if not token:
            return AgentResult(success=False, message="Bank not connected")
        
        try:
            await client.remove_item(token)
            del self._access_tokens[business_id][item_id]
            return AgentResult(success=True, message="Bank disconnected")
        except Exception as e:
            return AgentResult(success=False, message=str(e))

    async def execute(self, task: str, context: dict) -> AgentResult:
        action = context.get("action", "")
        business_id = context.get("business_id", "")
        
        if action == "link":
            return await self.create_link_token(business_id, context.get("user_id", ""))
        elif action == "exchange":
            return await self.exchange_token(business_id, context["public_token"])
        elif action == "accounts":
            return await self.get_accounts(business_id)
        elif action == "balances":
            return await self.get_balances(business_id)
        elif action == "transactions":
            return await self.get_transactions(business_id, context.get("days", 30))
        elif action == "cash_flow":
            return await self.analyze_cash_flow(business_id, context.get("days", 30))
        elif action == "health":
            return await self.get_financial_health(business_id)
        
        return AgentResult(success=False, message=f"Unknown action: {action}")
```

---

### Task 16.3: Banking API Routes

**File: `src/api/routes/banking.py`**

```python
"""
Banking API Routes - Plaid banking integration endpoints.
"""
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from src.agents.banking import BankingAgent
from src.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/banking", tags=["banking"])

_agent: Optional[BankingAgent] = None


def get_agent() -> BankingAgent:
    global _agent
    if _agent is None:
        _agent = BankingAgent()
    return _agent


class InitPlaidRequest(BaseModel):
    business_id: str
    client_id: str
    secret: str
    environment: str = "sandbox"


class LinkTokenRequest(BaseModel):
    business_id: str
    user_id: str
    products: Optional[list[str]] = None


class ExchangeTokenRequest(BaseModel):
    business_id: str
    public_token: str


class DisconnectRequest(BaseModel):
    business_id: str
    item_id: str


@router.post("/init")
async def init_plaid(req: InitPlaidRequest, agent: BankingAgent = Depends(get_agent)):
    """Initialize Plaid connection."""
    success = await agent.initialize_plaid(
        req.business_id, req.client_id, req.secret, req.environment
    )
    if not success:
        raise HTTPException(500, "Failed to initialize Plaid")
    return {"status": "ok"}


@router.post("/link-token")
async def create_link_token(req: LinkTokenRequest, agent: BankingAgent = Depends(get_agent)):
    """Create a link token for Plaid Link."""
    result = await agent.create_link_token(req.business_id, req.user_id, req.products)
    if not result.success:
        raise HTTPException(400, result.message)
    return result.data


@router.post("/exchange-token")
async def exchange_token(req: ExchangeTokenRequest, agent: BankingAgent = Depends(get_agent)):
    """Exchange public token for access token."""
    result = await agent.exchange_token(req.business_id, req.public_token)
    if not result.success:
        raise HTTPException(400, result.message)
    return result.data


@router.get("/accounts/{business_id}")
async def get_accounts(business_id: str, agent: BankingAgent = Depends(get_agent)):
    """Get all connected bank accounts."""
    result = await agent.get_accounts(business_id)
    if not result.success:
        raise HTTPException(400, result.message)
    return result.data


@router.get("/balances/{business_id}")
async def get_balances(business_id: str, agent: BankingAgent = Depends(get_agent)):
    """Get real-time balances."""
    result = await agent.get_balances(business_id)
    if not result.success:
        raise HTTPException(400, result.message)
    return result.data


@router.get("/transactions/{business_id}")
async def get_transactions(
    business_id: str,
    days: int = 30,
    account_id: Optional[str] = None,
    agent: BankingAgent = Depends(get_agent)
):
    """Get recent transactions."""
    result = await agent.get_transactions(business_id, days, account_id)
    if not result.success:
        raise HTTPException(400, result.message)
    return result.data


@router.get("/cash-flow/{business_id}")
async def analyze_cash_flow(
    business_id: str,
    days: int = 30,
    agent: BankingAgent = Depends(get_agent)
):
    """Analyze cash flow."""
    result = await agent.analyze_cash_flow(business_id, days)
    if not result.success:
        raise HTTPException(400, result.message)
    return result.data


@router.get("/health/{business_id}")
async def get_financial_health(business_id: str, agent: BankingAgent = Depends(get_agent)):
    """Get financial health score."""
    result = await agent.get_financial_health(business_id)
    if not result.success:
        raise HTTPException(400, result.message)
    return result.data


@router.post("/disconnect")
async def disconnect_bank(req: DisconnectRequest, agent: BankingAgent = Depends(get_agent)):
    """Disconnect a bank account."""
    result = await agent.disconnect_bank(req.business_id, req.item_id)
    if not result.success:
        raise HTTPException(400, result.message)
    return {"status": "disconnected"}
```

---

### Task 16.4: Tests

**File: `tests/test_banking.py`**

```python
"""Tests for Banking Agent and Plaid Client."""
import pytest
from datetime import date
from unittest.mock import AsyncMock
from src.integrations.plaid_client import (
    PlaidClient, PlaidEnvironment, PlaidAccount, PlaidTransaction,
    AccountType, TransactionCategory
)
from src.agents.banking import BankingAgent


@pytest.fixture
def plaid_client():
    return PlaidClient("client_id", "secret", PlaidEnvironment.SANDBOX)


@pytest.fixture
def banking_agent():
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
        assert not result.success
        assert "not configured" in result.message

    @pytest.mark.asyncio
    async def test_get_accounts_success(self, banking_agent, sample_account):
        await banking_agent.initialize_plaid("biz_1", "id", "secret")
        banking_agent._access_tokens["biz_1"] = {"item_1": "token_123"}
        
        banking_agent._clients["biz_1"].get_accounts = AsyncMock(
            return_value=[sample_account]
        )
        
        result = await banking_agent.get_accounts("biz_1")
        assert result.success
        assert len(result.data["accounts"]) == 1
        assert result.data["accounts"][0]["balance"] == 5000.00

    @pytest.mark.asyncio
    async def test_analyze_cash_flow(self, banking_agent):
        await banking_agent.initialize_plaid("biz_1", "id", "secret")
        
        # Mock get_transactions to return test data
        async def mock_get_transactions(*args, **kwargs):
            from src.agents.base import AgentResult
            return AgentResult(
                success=True,
                data={
                    "transactions": [
                        {"amount": 100, "is_expense": True, "category": "food", "pending": False},
                        {"amount": -500, "is_expense": False, "category": "income", "pending": False},
                    ],
                    "count": 2
                }
            )
        
        banking_agent.get_transactions = mock_get_transactions
        
        result = await banking_agent.analyze_cash_flow("biz_1", 30)
        assert result.success
        assert result.data["total_income"] == 500
        assert result.data["total_expenses"] == 100
        assert result.data["net_cash_flow"] == 400
```

---

## 3. Acceptance Criteria

| Criteria | Validation |
|----------|------------|
| Plaid client connects | Link token creation works |
| Bank accounts link | Public token exchange successful |
| Accounts retrieved | List accounts with balances |
| Transactions fetched | Transaction history with categories |
| Cash flow analyzed | Income/expense breakdown |
| Financial health scored | Health metrics calculated |

---

## 4. File Summary

| File | Purpose |
|------|---------|
| `src/integrations/plaid_client.py` | Plaid API client |
| `src/agents/banking.py` | Banking agent for analysis |
| `src/api/routes/banking.py` | REST API endpoints |
| `tests/test_banking.py` | Unit tests |

---

## 5. Environment Variables

```
PLAID_CLIENT_ID=your_client_id
PLAID_SECRET=your_secret
PLAID_ENV=sandbox  # sandbox, development, production
```
