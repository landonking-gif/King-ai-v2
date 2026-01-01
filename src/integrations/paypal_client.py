"""
PayPal Payment Processing Client.

Fallback payment provider when Stripe is unavailable.
Implements the same interface as stripe_client for seamless failover.
"""
import hmac
import hashlib
import json
import base64
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional, List
import httpx
from src.utils.logging import get_logger
from src.utils.circuit_breaker import CircuitBreaker

logger = get_logger(__name__)

# PayPal circuit breaker
paypal_circuit = CircuitBreaker(
    "paypal",
    failure_threshold=5,
    timeout=30.0,
    success_threshold=3
)


class PayPalPaymentStatus(Enum):
    CREATED = "CREATED"
    SAVED = "SAVED"
    APPROVED = "APPROVED"
    VOIDED = "VOIDED"
    COMPLETED = "COMPLETED"
    PAYER_ACTION_REQUIRED = "PAYER_ACTION_REQUIRED"


@dataclass
class PayPalOrder:
    """Represents a PayPal order (equivalent to Stripe PaymentIntent)."""
    id: str
    status: PayPalPaymentStatus
    amount: int  # cents
    currency: str
    payer_id: Optional[str] = None
    payer_email: Optional[str] = None
    description: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    created_time: Optional[datetime] = None
    approve_url: Optional[str] = None

    @property
    def amount_dollars(self) -> float:
        return self.amount / 100


@dataclass
class PayPalPayout:
    """Represents a PayPal payout to a seller/vendor."""
    id: str
    batch_id: str
    status: str
    amount: int
    currency: str
    recipient_email: str
    sender_item_id: Optional[str] = None


class PayPalClient:
    """
    PayPal REST API client for payment processing.
    
    Used as fallback when Stripe is unavailable.
    Implements compatible interface for seamless failover.
    """

    # Sandbox vs Production
    SANDBOX_URL = "https://api-m.sandbox.paypal.com"
    PRODUCTION_URL = "https://api-m.paypal.com"

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        sandbox: bool = True,
        webhook_id: str = ""
    ):
        self.client_id = client_id
        self.client_secret = client_secret
        self.sandbox = sandbox
        self.webhook_id = webhook_id
        self.base_url = self.SANDBOX_URL if sandbox else self.PRODUCTION_URL
        self._access_token: Optional[str] = None
        self._token_expires: Optional[datetime] = None
        self.client = httpx.AsyncClient(timeout=30.0)

    async def _get_access_token(self) -> str:
        """Get OAuth2 access token, refreshing if expired."""
        if self._access_token and self._token_expires:
            if datetime.utcnow() < self._token_expires:
                return self._access_token

        # Request new token
        auth = base64.b64encode(
            f"{self.client_id}:{self.client_secret}".encode()
        ).decode()

        response = await self.client.post(
            f"{self.base_url}/v1/oauth2/token",
            headers={
                "Authorization": f"Basic {auth}",
                "Content-Type": "application/x-www-form-urlencoded"
            },
            data={"grant_type": "client_credentials"}
        )
        response.raise_for_status()
        data = response.json()

        self._access_token = data["access_token"]
        # Token expires in seconds, subtract buffer
        expires_in = data.get("expires_in", 3600) - 300
        from datetime import timedelta
        self._token_expires = datetime.utcnow() + timedelta(seconds=expires_in)

        return self._access_token

    async def _request(
        self,
        method: str,
        endpoint: str,
        data: dict = None,
        params: dict = None
    ) -> dict:
        """Make authenticated request to PayPal API."""
        token = await self._get_access_token()

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

        url = f"{self.base_url}{endpoint}"

        response = await self.client.request(
            method,
            url,
            headers=headers,
            json=data,
            params=params
        )
        response.raise_for_status()

        if response.status_code == 204:
            return {}
        return response.json()

    @paypal_circuit.protect
    async def create_order(
        self,
        amount: int,
        currency: str = "USD",
        description: str = None,
        return_url: str = None,
        cancel_url: str = None,
        metadata: dict = None
    ) -> PayPalOrder:
        """
        Create a PayPal order (equivalent to Stripe PaymentIntent).
        
        Args:
            amount: Amount in cents
            currency: Currency code (default USD)
            description: Order description
            return_url: URL to redirect after approval
            cancel_url: URL to redirect on cancel
            metadata: Custom metadata
            
        Returns:
            PayPalOrder object
        """
        amount_decimal = f"{amount / 100:.2f}"

        order_data = {
            "intent": "CAPTURE",
            "purchase_units": [{
                "amount": {
                    "currency_code": currency.upper(),
                    "value": amount_decimal
                },
                "description": description
            }]
        }

        if return_url and cancel_url:
            order_data["application_context"] = {
                "return_url": return_url,
                "cancel_url": cancel_url
            }

        result = await self._request("POST", "/v2/checkout/orders", data=order_data)

        # Find approval URL
        approve_url = None
        for link in result.get("links", []):
            if link.get("rel") == "approve":
                approve_url = link.get("href")
                break

        logger.info(
            "Created PayPal order",
            order_id=result["id"],
            amount=amount,
            currency=currency
        )

        return PayPalOrder(
            id=result["id"],
            status=PayPalPaymentStatus(result["status"]),
            amount=amount,
            currency=currency,
            description=description,
            metadata=metadata or {},
            created_time=datetime.utcnow(),
            approve_url=approve_url
        )

    @paypal_circuit.protect
    async def capture_order(self, order_id: str) -> PayPalOrder:
        """
        Capture an approved order (charge the payment).
        
        Args:
            order_id: The PayPal order ID
            
        Returns:
            Updated PayPalOrder
        """
        result = await self._request(
            "POST",
            f"/v2/checkout/orders/{order_id}/capture"
        )

        capture = result["purchase_units"][0]["payments"]["captures"][0]
        amount_str = capture["amount"]["value"]
        amount_cents = int(float(amount_str) * 100)

        payer = result.get("payer", {})

        logger.info(
            "Captured PayPal order",
            order_id=order_id,
            amount=amount_cents
        )

        return PayPalOrder(
            id=result["id"],
            status=PayPalPaymentStatus(result["status"]),
            amount=amount_cents,
            currency=capture["amount"]["currency_code"],
            payer_id=payer.get("payer_id"),
            payer_email=payer.get("email_address")
        )

    @paypal_circuit.protect
    async def get_order(self, order_id: str) -> PayPalOrder:
        """Get order details."""
        result = await self._request("GET", f"/v2/checkout/orders/{order_id}")

        amount_str = result["purchase_units"][0]["amount"]["value"]
        amount_cents = int(float(amount_str) * 100)

        return PayPalOrder(
            id=result["id"],
            status=PayPalPaymentStatus(result["status"]),
            amount=amount_cents,
            currency=result["purchase_units"][0]["amount"]["currency_code"]
        )

    @paypal_circuit.protect
    async def refund_capture(
        self,
        capture_id: str,
        amount: int = None,
        currency: str = "USD",
        reason: str = None
    ) -> dict:
        """
        Refund a captured payment.
        
        Args:
            capture_id: The capture ID to refund
            amount: Amount in cents (None for full refund)
            currency: Currency code
            reason: Refund reason
            
        Returns:
            Refund details
        """
        data = {}
        if amount:
            data["amount"] = {
                "currency_code": currency.upper(),
                "value": f"{amount / 100:.2f}"
            }
        if reason:
            data["note_to_payer"] = reason

        result = await self._request(
            "POST",
            f"/v2/payments/captures/{capture_id}/refund",
            data=data if data else None
        )

        logger.info(
            "Refunded PayPal capture",
            capture_id=capture_id,
            refund_id=result.get("id")
        )

        return result

    @paypal_circuit.protect
    async def create_payout(
        self,
        recipient_email: str,
        amount: int,
        currency: str = "USD",
        note: str = None,
        sender_item_id: str = None
    ) -> PayPalPayout:
        """
        Send money to a recipient (vendor payout).
        
        Args:
            recipient_email: Recipient's PayPal email
            amount: Amount in cents
            currency: Currency code
            note: Note to recipient
            sender_item_id: Your reference ID
            
        Returns:
            PayPalPayout object
        """
        import uuid
        sender_batch_id = str(uuid.uuid4())
        sender_item_id = sender_item_id or str(uuid.uuid4())

        payout_data = {
            "sender_batch_header": {
                "sender_batch_id": sender_batch_id,
                "email_subject": "You have a payment",
                "email_message": note or "You have received a payment."
            },
            "items": [{
                "recipient_type": "EMAIL",
                "amount": {
                    "value": f"{amount / 100:.2f}",
                    "currency": currency.upper()
                },
                "receiver": recipient_email,
                "note": note,
                "sender_item_id": sender_item_id
            }]
        }

        result = await self._request(
            "POST",
            "/v1/payments/payouts",
            data=payout_data
        )

        batch_header = result["batch_header"]

        logger.info(
            "Created PayPal payout",
            batch_id=batch_header["payout_batch_id"],
            recipient=recipient_email,
            amount=amount
        )

        return PayPalPayout(
            id=batch_header["payout_batch_id"],
            batch_id=sender_batch_id,
            status=batch_header["batch_status"],
            amount=amount,
            currency=currency,
            recipient_email=recipient_email,
            sender_item_id=sender_item_id
        )

    def verify_webhook_signature(
        self,
        payload: bytes,
        headers: dict
    ) -> bool:
        """
        Verify PayPal webhook signature.
        
        Note: Full verification requires calling PayPal API.
        This is a simplified check.
        """
        # PayPal uses different headers for webhook verification
        transmission_id = headers.get("paypal-transmission-id")
        timestamp = headers.get("paypal-transmission-time")
        webhook_signature = headers.get("paypal-transmission-sig")

        if not all([transmission_id, timestamp, webhook_signature]):
            return False

        # For production, call /v1/notifications/verify-webhook-signature
        # This is a placeholder that always returns True in sandbox
        if self.sandbox:
            return True

        # TODO: Implement full verification
        return True

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


# Factory function to create payment client with fallback
async def get_payment_client():
    """
    Get the best available payment client.
    
    Returns Stripe if available and healthy, otherwise PayPal.
    """
    from src.utils.circuit_breaker import stripe_circuit
    from config.settings import settings
    import os

    # Check if Stripe is healthy
    if stripe_circuit.is_closed and settings.stripe_api_key:
        from src.integrations.stripe_client import StripeClient
        return StripeClient(
            api_key=settings.stripe_api_key,
            webhook_secret=os.getenv("STRIPE_WEBHOOK_SECRET", "")
        ), "stripe"

    # Fall back to PayPal
    paypal_client_id = os.getenv("PAYPAL_CLIENT_ID")
    paypal_secret = os.getenv("PAYPAL_CLIENT_SECRET")
    paypal_sandbox = os.getenv("PAYPAL_SANDBOX", "true").lower() == "true"

    if paypal_client_id and paypal_secret:
        return PayPalClient(
            client_id=paypal_client_id,
            client_secret=paypal_secret,
            sandbox=paypal_sandbox
        ), "paypal"

    # No payment provider available
    raise RuntimeError("No payment provider available - both Stripe and PayPal are down or unconfigured")
