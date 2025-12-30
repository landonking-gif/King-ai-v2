# Implementation Plan Part 15: Finance - Stripe Integration Sub-Agent

| Field | Value |
|-------|-------|
| Module | Stripe Payment Processing Integration |
| Priority | High |
| Estimated Effort | 4-5 hours |
| Dependencies | Part 3 (Database), Part 13 (Commerce) |

---

## 1. Scope

This module implements Stripe payment processing integration:

- **Stripe Client** - Payment intents, subscriptions, customers, invoices
- **Webhook Handler** - Process Stripe events securely
- **Finance Agent** - Revenue tracking, refunds, payouts, reporting
- **Subscription Management** - SaaS billing, plan upgrades, cancellations

---

## 2. Tasks

### Task 15.1: Stripe Client

**File: `src/integrations/stripe_client.py`**

```python
"""
Stripe Payment Processing Client.
"""
import hmac
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
import httpx
from src.utils.logging import get_logger

logger = get_logger(__name__)


class PaymentStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"
    REFUNDED = "refunded"


class SubscriptionStatus(Enum):
    ACTIVE = "active"
    PAST_DUE = "past_due"
    CANCELED = "canceled"
    UNPAID = "unpaid"
    TRIALING = "trialing"
    INCOMPLETE = "incomplete"


@dataclass
class StripeCustomer:
    id: str
    email: str
    name: Optional[str] = None
    phone: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    created: Optional[datetime] = None
    default_payment_method: Optional[str] = None


@dataclass
class PaymentIntent:
    id: str
    amount: int  # cents
    currency: str
    status: PaymentStatus
    customer_id: Optional[str] = None
    payment_method: Optional[str] = None
    description: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    created: Optional[datetime] = None

    @property
    def amount_dollars(self) -> float:
        return self.amount / 100


@dataclass
class Subscription:
    id: str
    customer_id: str
    status: SubscriptionStatus
    price_id: str
    current_period_start: datetime
    current_period_end: datetime
    cancel_at_period_end: bool = False
    canceled_at: Optional[datetime] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class Invoice:
    id: str
    customer_id: str
    subscription_id: Optional[str]
    amount_due: int
    amount_paid: int
    status: str
    paid: bool
    created: datetime
    hosted_invoice_url: Optional[str] = None
    pdf_url: Optional[str] = None


class StripeClient:
    """Stripe API client for payment processing."""

    BASE_URL = "https://api.stripe.com/v1"

    def __init__(self, api_key: str, webhook_secret: str = ""):
        self.api_key = api_key
        self.webhook_secret = webhook_secret
        self.client = httpx.AsyncClient(
            timeout=30.0,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/x-www-form-urlencoded",
            }
        )

    async def close(self):
        await self.client.aclose()

    async def _request(
        self, method: str, endpoint: str, data: dict = None
    ) -> dict:
        url = f"{self.BASE_URL}{endpoint}"
        try:
            if method == "GET":
                resp = await self.client.get(url, params=data)
            elif method == "POST":
                resp = await self.client.post(url, data=data)
            elif method == "DELETE":
                resp = await self.client.delete(url)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"Stripe API error: {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Stripe request failed: {e}")
            raise

    # Customer Management
    async def create_customer(
        self, email: str, name: str = None, metadata: dict = None
    ) -> StripeCustomer:
        data = {"email": email}
        if name:
            data["name"] = name
        if metadata:
            for k, v in metadata.items():
                data[f"metadata[{k}]"] = str(v)
        
        result = await self._request("POST", "/customers", data)
        return self._parse_customer(result)

    async def get_customer(self, customer_id: str) -> Optional[StripeCustomer]:
        try:
            result = await self._request("GET", f"/customers/{customer_id}")
            return self._parse_customer(result)
        except Exception:
            return None

    async def update_customer(
        self, customer_id: str, **updates
    ) -> StripeCustomer:
        data = {}
        for k, v in updates.items():
            if k == "metadata":
                for mk, mv in v.items():
                    data[f"metadata[{mk}]"] = str(mv)
            else:
                data[k] = v
        
        result = await self._request("POST", f"/customers/{customer_id}", data)
        return self._parse_customer(result)

    async def list_customers(
        self, limit: int = 10, starting_after: str = None
    ) -> list[StripeCustomer]:
        params = {"limit": limit}
        if starting_after:
            params["starting_after"] = starting_after
        
        result = await self._request("GET", "/customers", params)
        return [self._parse_customer(c) for c in result.get("data", [])]

    # Payment Intents
    async def create_payment_intent(
        self,
        amount: int,
        currency: str = "usd",
        customer_id: str = None,
        description: str = None,
        metadata: dict = None,
        automatic_payment_methods: bool = True,
    ) -> PaymentIntent:
        data = {
            "amount": amount,
            "currency": currency,
            "automatic_payment_methods[enabled]": str(automatic_payment_methods).lower(),
        }
        if customer_id:
            data["customer"] = customer_id
        if description:
            data["description"] = description
        if metadata:
            for k, v in metadata.items():
                data[f"metadata[{k}]"] = str(v)
        
        result = await self._request("POST", "/payment_intents", data)
        return self._parse_payment_intent(result)

    async def get_payment_intent(self, intent_id: str) -> Optional[PaymentIntent]:
        try:
            result = await self._request("GET", f"/payment_intents/{intent_id}")
            return self._parse_payment_intent(result)
        except Exception:
            return None

    async def confirm_payment_intent(
        self, intent_id: str, payment_method: str = None
    ) -> PaymentIntent:
        data = {}
        if payment_method:
            data["payment_method"] = payment_method
        
        result = await self._request(
            "POST", f"/payment_intents/{intent_id}/confirm", data
        )
        return self._parse_payment_intent(result)

    async def cancel_payment_intent(self, intent_id: str) -> PaymentIntent:
        result = await self._request(
            "POST", f"/payment_intents/{intent_id}/cancel", {}
        )
        return self._parse_payment_intent(result)

    # Refunds
    async def create_refund(
        self,
        payment_intent_id: str,
        amount: int = None,
        reason: str = None,
    ) -> dict:
        data = {"payment_intent": payment_intent_id}
        if amount:
            data["amount"] = amount
        if reason:
            data["reason"] = reason
        
        return await self._request("POST", "/refunds", data)

    # Subscriptions
    async def create_subscription(
        self,
        customer_id: str,
        price_id: str,
        trial_days: int = None,
        metadata: dict = None,
    ) -> Subscription:
        data = {
            "customer": customer_id,
            "items[0][price]": price_id,
        }
        if trial_days:
            data["trial_period_days"] = trial_days
        if metadata:
            for k, v in metadata.items():
                data[f"metadata[{k}]"] = str(v)
        
        result = await self._request("POST", "/subscriptions", data)
        return self._parse_subscription(result)

    async def get_subscription(self, sub_id: str) -> Optional[Subscription]:
        try:
            result = await self._request("GET", f"/subscriptions/{sub_id}")
            return self._parse_subscription(result)
        except Exception:
            return None

    async def update_subscription(
        self, sub_id: str, price_id: str = None, **updates
    ) -> Subscription:
        data = {}
        if price_id:
            data["items[0][price]"] = price_id
        data.update(updates)
        
        result = await self._request("POST", f"/subscriptions/{sub_id}", data)
        return self._parse_subscription(result)

    async def cancel_subscription(
        self, sub_id: str, at_period_end: bool = True
    ) -> Subscription:
        if at_period_end:
            data = {"cancel_at_period_end": "true"}
            result = await self._request("POST", f"/subscriptions/{sub_id}", data)
        else:
            result = await self._request("DELETE", f"/subscriptions/{sub_id}")
        return self._parse_subscription(result)

    async def list_subscriptions(
        self, customer_id: str = None, status: str = None, limit: int = 10
    ) -> list[Subscription]:
        params = {"limit": limit}
        if customer_id:
            params["customer"] = customer_id
        if status:
            params["status"] = status
        
        result = await self._request("GET", "/subscriptions", params)
        return [self._parse_subscription(s) for s in result.get("data", [])]

    # Invoices
    async def list_invoices(
        self, customer_id: str = None, subscription_id: str = None, limit: int = 10
    ) -> list[Invoice]:
        params = {"limit": limit}
        if customer_id:
            params["customer"] = customer_id
        if subscription_id:
            params["subscription"] = subscription_id
        
        result = await self._request("GET", "/invoices", params)
        return [self._parse_invoice(i) for i in result.get("data", [])]

    async def get_invoice(self, invoice_id: str) -> Optional[Invoice]:
        try:
            result = await self._request("GET", f"/invoices/{invoice_id}")
            return self._parse_invoice(result)
        except Exception:
            return None

    # Balance & Payouts
    async def get_balance(self) -> dict:
        return await self._request("GET", "/balance")

    async def list_payouts(self, limit: int = 10) -> list[dict]:
        result = await self._request("GET", "/payouts", {"limit": limit})
        return result.get("data", [])

    async def create_payout(self, amount: int, currency: str = "usd") -> dict:
        return await self._request("POST", "/payouts", {
            "amount": amount,
            "currency": currency,
        })

    # Webhook Verification
    def verify_webhook(self, payload: bytes, signature: str) -> dict:
        """Verify webhook signature and return parsed event."""
        if not self.webhook_secret:
            raise ValueError("Webhook secret not configured")
        
        parts = dict(p.split("=", 1) for p in signature.split(",") if "=" in p)
        timestamp = parts.get("t", "")
        v1_sig = parts.get("v1", "")
        
        signed_payload = f"{timestamp}.{payload.decode()}"
        expected_sig = hmac.new(
            self.webhook_secret.encode(),
            signed_payload.encode(),
            hashlib.sha256
        ).hexdigest()
        
        if not hmac.compare_digest(expected_sig, v1_sig):
            raise ValueError("Invalid webhook signature")
        
        import json
        return json.loads(payload)

    # Parsing helpers
    def _parse_customer(self, data: dict) -> StripeCustomer:
        return StripeCustomer(
            id=data["id"],
            email=data.get("email", ""),
            name=data.get("name"),
            phone=data.get("phone"),
            metadata=data.get("metadata", {}),
            created=datetime.fromtimestamp(data["created"]) if data.get("created") else None,
            default_payment_method=data.get("default_payment_method"),
        )

    def _parse_payment_intent(self, data: dict) -> PaymentIntent:
        status_map = {
            "requires_payment_method": PaymentStatus.PENDING,
            "requires_confirmation": PaymentStatus.PENDING,
            "requires_action": PaymentStatus.PENDING,
            "processing": PaymentStatus.PROCESSING,
            "succeeded": PaymentStatus.SUCCEEDED,
            "canceled": PaymentStatus.CANCELED,
        }
        return PaymentIntent(
            id=data["id"],
            amount=data["amount"],
            currency=data["currency"],
            status=status_map.get(data["status"], PaymentStatus.PENDING),
            customer_id=data.get("customer"),
            payment_method=data.get("payment_method"),
            description=data.get("description"),
            metadata=data.get("metadata", {}),
            created=datetime.fromtimestamp(data["created"]) if data.get("created") else None,
        )

    def _parse_subscription(self, data: dict) -> Subscription:
        return Subscription(
            id=data["id"],
            customer_id=data["customer"],
            status=SubscriptionStatus(data["status"]),
            price_id=data["items"]["data"][0]["price"]["id"] if data.get("items") else "",
            current_period_start=datetime.fromtimestamp(data["current_period_start"]),
            current_period_end=datetime.fromtimestamp(data["current_period_end"]),
            cancel_at_period_end=data.get("cancel_at_period_end", False),
            canceled_at=datetime.fromtimestamp(data["canceled_at"]) if data.get("canceled_at") else None,
            metadata=data.get("metadata", {}),
        )

    def _parse_invoice(self, data: dict) -> Invoice:
        return Invoice(
            id=data["id"],
            customer_id=data["customer"],
            subscription_id=data.get("subscription"),
            amount_due=data["amount_due"],
            amount_paid=data["amount_paid"],
            status=data["status"],
            paid=data["paid"],
            created=datetime.fromtimestamp(data["created"]),
            hosted_invoice_url=data.get("hosted_invoice_url"),
            pdf_url=data.get("invoice_pdf"),
        )
```

---

### Task 15.2: Finance Agent

**File: `src/agents/finance.py`**

```python
"""
Finance Agent - Payment processing, subscriptions, revenue tracking.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Optional
from src.agents.base import BaseAgent, AgentCapability, AgentResult
from src.integrations.stripe_client import (
    StripeClient, StripeCustomer, PaymentIntent, Subscription,
    PaymentStatus, SubscriptionStatus
)
from src.database.connection import get_db_session
from src.database.models import BusinessUnit
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RevenueMetrics:
    total_revenue: float
    monthly_recurring: float
    active_subscriptions: int
    churn_rate: float
    average_order_value: float
    pending_payouts: float


class FinanceAgent(BaseAgent):
    """Agent for financial operations and payment processing."""

    def __init__(self):
        super().__init__(
            name="Finance Agent",
            capabilities=[AgentCapability.COMMERCE, AgentCapability.ANALYSIS]
        )
        self._clients: dict[str, StripeClient] = {}

    async def initialize_stripe(
        self, business_id: str, api_key: str, webhook_secret: str = ""
    ) -> bool:
        """Initialize Stripe client for a business."""
        try:
            self._clients[business_id] = StripeClient(api_key, webhook_secret)
            logger.info(f"Initialized Stripe for business {business_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Stripe: {e}")
            return False

    def _get_client(self, business_id: str) -> Optional[StripeClient]:
        return self._clients.get(business_id)

    # Customer Operations
    async def create_customer(
        self, business_id: str, email: str, name: str = None, metadata: dict = None
    ) -> AgentResult:
        client = self._get_client(business_id)
        if not client:
            return AgentResult(success=False, message="Stripe not configured")
        
        try:
            customer = await client.create_customer(email, name, metadata)
            return AgentResult(
                success=True,
                message="Customer created",
                data={"customer_id": customer.id, "email": customer.email}
            )
        except Exception as e:
            return AgentResult(success=False, message=str(e))

    async def get_customer(self, business_id: str, customer_id: str) -> AgentResult:
        client = self._get_client(business_id)
        if not client:
            return AgentResult(success=False, message="Stripe not configured")
        
        customer = await client.get_customer(customer_id)
        if not customer:
            return AgentResult(success=False, message="Customer not found")
        
        return AgentResult(
            success=True,
            data={
                "id": customer.id,
                "email": customer.email,
                "name": customer.name,
                "metadata": customer.metadata,
            }
        )

    # Payment Operations
    async def create_payment(
        self,
        business_id: str,
        amount: float,
        currency: str = "usd",
        customer_id: str = None,
        description: str = None,
        metadata: dict = None,
    ) -> AgentResult:
        client = self._get_client(business_id)
        if not client:
            return AgentResult(success=False, message="Stripe not configured")
        
        try:
            amount_cents = int(amount * 100)
            intent = await client.create_payment_intent(
                amount=amount_cents,
                currency=currency,
                customer_id=customer_id,
                description=description,
                metadata=metadata,
            )
            return AgentResult(
                success=True,
                message="Payment intent created",
                data={
                    "payment_intent_id": intent.id,
                    "amount": intent.amount_dollars,
                    "status": intent.status.value,
                }
            )
        except Exception as e:
            return AgentResult(success=False, message=str(e))

    async def process_refund(
        self,
        business_id: str,
        payment_intent_id: str,
        amount: float = None,
        reason: str = None,
    ) -> AgentResult:
        client = self._get_client(business_id)
        if not client:
            return AgentResult(success=False, message="Stripe not configured")
        
        try:
            amount_cents = int(amount * 100) if amount else None
            refund = await client.create_refund(payment_intent_id, amount_cents, reason)
            return AgentResult(
                success=True,
                message="Refund processed",
                data={"refund_id": refund["id"], "amount": refund["amount"] / 100}
            )
        except Exception as e:
            return AgentResult(success=False, message=str(e))

    # Subscription Operations
    async def create_subscription(
        self,
        business_id: str,
        customer_id: str,
        price_id: str,
        trial_days: int = None,
        metadata: dict = None,
    ) -> AgentResult:
        client = self._get_client(business_id)
        if not client:
            return AgentResult(success=False, message="Stripe not configured")
        
        try:
            sub = await client.create_subscription(
                customer_id, price_id, trial_days, metadata
            )
            return AgentResult(
                success=True,
                message="Subscription created",
                data={
                    "subscription_id": sub.id,
                    "status": sub.status.value,
                    "current_period_end": sub.current_period_end.isoformat(),
                }
            )
        except Exception as e:
            return AgentResult(success=False, message=str(e))

    async def cancel_subscription(
        self, business_id: str, subscription_id: str, immediate: bool = False
    ) -> AgentResult:
        client = self._get_client(business_id)
        if not client:
            return AgentResult(success=False, message="Stripe not configured")
        
        try:
            sub = await client.cancel_subscription(subscription_id, at_period_end=not immediate)
            return AgentResult(
                success=True,
                message="Subscription canceled",
                data={
                    "subscription_id": sub.id,
                    "status": sub.status.value,
                    "cancel_at_period_end": sub.cancel_at_period_end,
                }
            )
        except Exception as e:
            return AgentResult(success=False, message=str(e))

    async def upgrade_subscription(
        self, business_id: str, subscription_id: str, new_price_id: str
    ) -> AgentResult:
        client = self._get_client(business_id)
        if not client:
            return AgentResult(success=False, message="Stripe not configured")
        
        try:
            sub = await client.update_subscription(subscription_id, price_id=new_price_id)
            return AgentResult(
                success=True,
                message="Subscription upgraded",
                data={"subscription_id": sub.id, "new_price": new_price_id}
            )
        except Exception as e:
            return AgentResult(success=False, message=str(e))

    # Revenue & Reporting
    async def get_revenue_metrics(self, business_id: str) -> AgentResult:
        client = self._get_client(business_id)
        if not client:
            return AgentResult(success=False, message="Stripe not configured")
        
        try:
            balance = await client.get_balance()
            subscriptions = await client.list_subscriptions(status="active", limit=100)
            
            available = sum(b["amount"] for b in balance.get("available", []))
            pending = sum(b["amount"] for b in balance.get("pending", []))
            mrr = len(subscriptions) * 2999  # Placeholder calculation
            
            metrics = RevenueMetrics(
                total_revenue=available / 100,
                monthly_recurring=mrr / 100,
                active_subscriptions=len(subscriptions),
                churn_rate=0.0,
                average_order_value=0.0,
                pending_payouts=pending / 100,
            )
            
            return AgentResult(
                success=True,
                data={
                    "total_revenue": metrics.total_revenue,
                    "monthly_recurring": metrics.monthly_recurring,
                    "active_subscriptions": metrics.active_subscriptions,
                    "pending_payouts": metrics.pending_payouts,
                }
            )
        except Exception as e:
            return AgentResult(success=False, message=str(e))

    async def get_balance(self, business_id: str) -> AgentResult:
        client = self._get_client(business_id)
        if not client:
            return AgentResult(success=False, message="Stripe not configured")
        
        try:
            balance = await client.get_balance()
            available = sum(b["amount"] for b in balance.get("available", [])) / 100
            pending = sum(b["amount"] for b in balance.get("pending", [])) / 100
            
            return AgentResult(
                success=True,
                data={"available": available, "pending": pending}
            )
        except Exception as e:
            return AgentResult(success=False, message=str(e))

    async def request_payout(
        self, business_id: str, amount: float, currency: str = "usd"
    ) -> AgentResult:
        client = self._get_client(business_id)
        if not client:
            return AgentResult(success=False, message="Stripe not configured")
        
        try:
            payout = await client.create_payout(int(amount * 100), currency)
            return AgentResult(
                success=True,
                message="Payout initiated",
                data={"payout_id": payout["id"], "amount": payout["amount"] / 100}
            )
        except Exception as e:
            return AgentResult(success=False, message=str(e))

    async def list_invoices(
        self, business_id: str, customer_id: str = None, limit: int = 10
    ) -> AgentResult:
        client = self._get_client(business_id)
        if not client:
            return AgentResult(success=False, message="Stripe not configured")
        
        try:
            invoices = await client.list_invoices(customer_id=customer_id, limit=limit)
            return AgentResult(
                success=True,
                data={
                    "invoices": [
                        {
                            "id": inv.id,
                            "amount_due": inv.amount_due / 100,
                            "amount_paid": inv.amount_paid / 100,
                            "status": inv.status,
                            "paid": inv.paid,
                            "pdf_url": inv.pdf_url,
                        }
                        for inv in invoices
                    ]
                }
            )
        except Exception as e:
            return AgentResult(success=False, message=str(e))

    # Webhook Processing
    async def handle_webhook(
        self, business_id: str, payload: bytes, signature: str
    ) -> AgentResult:
        client = self._get_client(business_id)
        if not client:
            return AgentResult(success=False, message="Stripe not configured")
        
        try:
            event = client.verify_webhook(payload, signature)
            event_type = event.get("type", "")
            data = event.get("data", {}).get("object", {})
            
            handlers = {
                "payment_intent.succeeded": self._handle_payment_succeeded,
                "payment_intent.payment_failed": self._handle_payment_failed,
                "customer.subscription.created": self._handle_subscription_created,
                "customer.subscription.deleted": self._handle_subscription_deleted,
                "invoice.paid": self._handle_invoice_paid,
                "invoice.payment_failed": self._handle_invoice_failed,
            }
            
            handler = handlers.get(event_type)
            if handler:
                await handler(business_id, data)
            
            return AgentResult(
                success=True,
                message=f"Processed {event_type}",
                data={"event_type": event_type, "event_id": event.get("id")}
            )
        except Exception as e:
            return AgentResult(success=False, message=str(e))

    async def _handle_payment_succeeded(self, business_id: str, data: dict):
        logger.info(f"Payment succeeded: {data.get('id')}")

    async def _handle_payment_failed(self, business_id: str, data: dict):
        logger.warning(f"Payment failed: {data.get('id')}")

    async def _handle_subscription_created(self, business_id: str, data: dict):
        logger.info(f"Subscription created: {data.get('id')}")

    async def _handle_subscription_deleted(self, business_id: str, data: dict):
        logger.info(f"Subscription deleted: {data.get('id')}")

    async def _handle_invoice_paid(self, business_id: str, data: dict):
        logger.info(f"Invoice paid: {data.get('id')}")

    async def _handle_invoice_failed(self, business_id: str, data: dict):
        logger.warning(f"Invoice payment failed: {data.get('id')}")

    async def execute(self, task: str, context: dict) -> AgentResult:
        action = context.get("action", "")
        business_id = context.get("business_id", "")
        
        if action == "create_payment":
            return await self.create_payment(
                business_id, context["amount"], context.get("currency", "usd"),
                context.get("customer_id"), context.get("description")
            )
        elif action == "refund":
            return await self.process_refund(
                business_id, context["payment_intent_id"], context.get("amount")
            )
        elif action == "create_subscription":
            return await self.create_subscription(
                business_id, context["customer_id"], context["price_id"]
            )
        elif action == "cancel_subscription":
            return await self.cancel_subscription(
                business_id, context["subscription_id"]
            )
        elif action == "metrics":
            return await self.get_revenue_metrics(business_id)
        elif action == "balance":
            return await self.get_balance(business_id)
        
        return AgentResult(success=False, message=f"Unknown action: {action}")
```

---

### Task 15.3: Finance API Routes

**File: `src/api/routes/finance.py`**

```python
"""
Finance API Routes - Payment processing endpoints.
"""
from typing import Optional
from fastapi import APIRouter, HTTPException, Request, Header, Depends
from pydantic import BaseModel, Field
from src.agents.finance import FinanceAgent
from src.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/finance", tags=["finance"])

_agent: Optional[FinanceAgent] = None


def get_agent() -> FinanceAgent:
    global _agent
    if _agent is None:
        _agent = FinanceAgent()
    return _agent


class InitStripeRequest(BaseModel):
    business_id: str
    api_key: str
    webhook_secret: str = ""


class CreateCustomerRequest(BaseModel):
    business_id: str
    email: str
    name: Optional[str] = None
    metadata: Optional[dict] = None


class PaymentRequest(BaseModel):
    business_id: str
    amount: float = Field(..., gt=0)
    currency: str = "usd"
    customer_id: Optional[str] = None
    description: Optional[str] = None


class RefundRequest(BaseModel):
    business_id: str
    payment_intent_id: str
    amount: Optional[float] = None
    reason: Optional[str] = None


class SubscriptionRequest(BaseModel):
    business_id: str
    customer_id: str
    price_id: str
    trial_days: Optional[int] = None


class CancelSubscriptionRequest(BaseModel):
    business_id: str
    subscription_id: str
    immediate: bool = False


class PayoutRequest(BaseModel):
    business_id: str
    amount: float = Field(..., gt=0)
    currency: str = "usd"


@router.post("/init")
async def init_stripe(req: InitStripeRequest, agent: FinanceAgent = Depends(get_agent)):
    success = await agent.initialize_stripe(req.business_id, req.api_key, req.webhook_secret)
    if not success:
        raise HTTPException(500, "Failed to initialize Stripe")
    return {"status": "ok"}


@router.post("/customers")
async def create_customer(req: CreateCustomerRequest, agent: FinanceAgent = Depends(get_agent)):
    result = await agent.create_customer(req.business_id, req.email, req.name, req.metadata)
    if not result.success:
        raise HTTPException(400, result.message)
    return result.data


@router.get("/customers/{business_id}/{customer_id}")
async def get_customer(business_id: str, customer_id: str, agent: FinanceAgent = Depends(get_agent)):
    result = await agent.get_customer(business_id, customer_id)
    if not result.success:
        raise HTTPException(404, result.message)
    return result.data


@router.post("/payments")
async def create_payment(req: PaymentRequest, agent: FinanceAgent = Depends(get_agent)):
    result = await agent.create_payment(
        req.business_id, req.amount, req.currency, req.customer_id, req.description
    )
    if not result.success:
        raise HTTPException(400, result.message)
    return result.data


@router.post("/refunds")
async def process_refund(req: RefundRequest, agent: FinanceAgent = Depends(get_agent)):
    result = await agent.process_refund(
        req.business_id, req.payment_intent_id, req.amount, req.reason
    )
    if not result.success:
        raise HTTPException(400, result.message)
    return result.data


@router.post("/subscriptions")
async def create_subscription(req: SubscriptionRequest, agent: FinanceAgent = Depends(get_agent)):
    result = await agent.create_subscription(
        req.business_id, req.customer_id, req.price_id, req.trial_days
    )
    if not result.success:
        raise HTTPException(400, result.message)
    return result.data


@router.post("/subscriptions/cancel")
async def cancel_subscription(req: CancelSubscriptionRequest, agent: FinanceAgent = Depends(get_agent)):
    result = await agent.cancel_subscription(req.business_id, req.subscription_id, req.immediate)
    if not result.success:
        raise HTTPException(400, result.message)
    return result.data


@router.get("/balance/{business_id}")
async def get_balance(business_id: str, agent: FinanceAgent = Depends(get_agent)):
    result = await agent.get_balance(business_id)
    if not result.success:
        raise HTTPException(400, result.message)
    return result.data


@router.get("/metrics/{business_id}")
async def get_metrics(business_id: str, agent: FinanceAgent = Depends(get_agent)):
    result = await agent.get_revenue_metrics(business_id)
    if not result.success:
        raise HTTPException(400, result.message)
    return result.data


@router.post("/payouts")
async def request_payout(req: PayoutRequest, agent: FinanceAgent = Depends(get_agent)):
    result = await agent.request_payout(req.business_id, req.amount, req.currency)
    if not result.success:
        raise HTTPException(400, result.message)
    return result.data


@router.get("/invoices/{business_id}")
async def list_invoices(
    business_id: str,
    customer_id: Optional[str] = None,
    limit: int = 10,
    agent: FinanceAgent = Depends(get_agent)
):
    result = await agent.list_invoices(business_id, customer_id, limit)
    if not result.success:
        raise HTTPException(400, result.message)
    return result.data


@router.post("/webhook/{business_id}")
async def stripe_webhook(
    business_id: str,
    request: Request,
    stripe_signature: str = Header(..., alias="Stripe-Signature"),
    agent: FinanceAgent = Depends(get_agent)
):
    payload = await request.body()
    result = await agent.handle_webhook(business_id, payload, stripe_signature)
    if not result.success:
        raise HTTPException(400, result.message)
    return {"received": True}
```

---

### Task 15.4: Tests

**File: `tests/test_finance.py`**

```python
"""Tests for Finance Agent and Stripe Client."""
import pytest
from unittest.mock import AsyncMock, patch
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

    def test_parse_payment_intent(self, stripe_client):
        data = {"id": "pi_123", "amount": 1000, "currency": "usd", "status": "succeeded", "created": 1609459200}
        intent = stripe_client._parse_payment_intent(data)
        assert intent.id == "pi_123"
        assert intent.amount_dollars == 10.0
        assert intent.status == PaymentStatus.SUCCEEDED


class TestFinanceAgent:
    @pytest.mark.asyncio
    async def test_initialize_stripe(self, finance_agent):
        result = await finance_agent.initialize_stripe("biz_1", "sk_test")
        assert result is True
        assert "biz_1" in finance_agent._clients

    @pytest.mark.asyncio
    async def test_no_client_configured(self, finance_agent):
        result = await finance_agent.create_customer("biz_unknown", "test@example.com")
        assert not result.success
        assert "not configured" in result.message

    @pytest.mark.asyncio
    async def test_create_customer_success(self, finance_agent):
        await finance_agent.initialize_stripe("biz_1", "sk_test")
        
        mock_customer = StripeCustomer(id="cus_123", email="test@example.com")
        finance_agent._clients["biz_1"].create_customer = AsyncMock(return_value=mock_customer)
        
        result = await finance_agent.create_customer("biz_1", "test@example.com", "Test User")
        assert result.success
        assert result.data["customer_id"] == "cus_123"

    @pytest.mark.asyncio
    async def test_create_payment_success(self, finance_agent):
        await finance_agent.initialize_stripe("biz_1", "sk_test")
        
        mock_intent = PaymentIntent(
            id="pi_123", amount=5000, currency="usd", status=PaymentStatus.PENDING
        )
        finance_agent._clients["biz_1"].create_payment_intent = AsyncMock(return_value=mock_intent)
        
        result = await finance_agent.create_payment("biz_1", 50.00)
        assert result.success
        assert result.data["payment_intent_id"] == "pi_123"
        assert result.data["amount"] == 50.0
```

---

## 3. Acceptance Criteria

| Criteria | Validation |
|----------|------------|
| Stripe client connects | API key authentication works |
| Customers created | Create/get customers via API |
| Payments processed | Payment intents created and tracked |
| Subscriptions managed | Create, upgrade, cancel subscriptions |
| Webhooks verified | Signature verification and event handling |
| Revenue metrics | Balance and metrics retrieved |

---

## 4. File Summary

| File | Purpose |
|------|---------|
| `src/integrations/stripe_client.py` | Stripe API client |
| `src/agents/finance.py` | Finance agent for payments |
| `src/api/routes/finance.py` | REST API endpoints |
| `tests/test_finance.py` | Unit tests |

---

## 5. Environment Variables

```
STRIPE_API_KEY=sk_live_...
STRIPE_WEBHOOK_SECRET=whsec_...
```
