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
