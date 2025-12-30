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
