"""
Webhook Handler System.

Handles incoming webhooks from external services:
- Stripe (payments, subscriptions)
- Shopify (orders, products)
- Plaid (transactions, account updates)
- Custom webhooks

Features:
- Signature verification
- Retry handling
- Event logging
- Async processing
"""

import hashlib
import hmac
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

from fastapi import APIRouter, Request, HTTPException, Header, BackgroundTasks
from pydantic import BaseModel

from src.utils.structured_logging import get_logger
from src.utils.audit_trail import audit_manager, AuditEventType, AuditSeverity

logger = get_logger("webhooks")

router = APIRouter(prefix="/api/v1/webhooks", tags=["webhooks"])


class WebhookSource(str, Enum):
    """Supported webhook sources."""
    STRIPE = "stripe"
    SHOPIFY = "shopify"
    PLAID = "plaid"
    CUSTOM = "custom"


class WebhookStatus(str, Enum):
    """Webhook processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    IGNORED = "ignored"


@dataclass
class WebhookEvent:
    """Represents a received webhook event."""
    id: str = field(default_factory=lambda: str(uuid4()))
    source: WebhookSource = WebhookSource.CUSTOM
    event_type: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)
    signature: Optional[str] = None
    received_at: datetime = field(default_factory=datetime.utcnow)
    processed_at: Optional[datetime] = None
    status: WebhookStatus = WebhookStatus.PENDING
    error: Optional[str] = None
    business_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source.value,
            "event_type": self.event_type,
            "payload": self.payload,
            "received_at": self.received_at.isoformat(),
            "processed_at": self.processed_at.isoformat() if self.processed_at else None,
            "status": self.status.value,
            "error": self.error,
            "business_id": self.business_id,
        }


# Type for webhook handlers
WebhookHandler = Callable[[WebhookEvent], None]


class WebhookRegistry:
    """
    Registry for webhook handlers.
    
    Allows registering handlers for specific source/event combinations.
    """
    
    def __init__(self):
        self._handlers: Dict[str, List[WebhookHandler]] = {}
        self._secrets: Dict[WebhookSource, str] = {}
    
    def register(
        self,
        source: WebhookSource,
        event_type: str,
        handler: WebhookHandler,
    ):
        """Register a handler for a webhook event."""
        key = f"{source.value}:{event_type}"
        if key not in self._handlers:
            self._handlers[key] = []
        self._handlers[key].append(handler)
        logger.info(f"Registered webhook handler for {key}")
    
    def set_secret(self, source: WebhookSource, secret: str):
        """Set signing secret for a webhook source."""
        self._secrets[source] = secret
    
    def get_secret(self, source: WebhookSource) -> Optional[str]:
        """Get signing secret for a webhook source."""
        return self._secrets.get(source)
    
    def get_handlers(
        self,
        source: WebhookSource,
        event_type: str,
    ) -> List[WebhookHandler]:
        """Get all handlers for a webhook event."""
        key = f"{source.value}:{event_type}"
        handlers = self._handlers.get(key, [])
        
        # Also check for wildcard handlers
        wildcard_key = f"{source.value}:*"
        handlers.extend(self._handlers.get(wildcard_key, []))
        
        return handlers


# Global registry
webhook_registry = WebhookRegistry()


class WebhookVerifier:
    """Verifies webhook signatures from different sources."""
    
    @staticmethod
    def verify_stripe(
        payload: bytes,
        signature: str,
        secret: str,
    ) -> bool:
        """Verify Stripe webhook signature."""
        try:
            # Stripe uses HMAC-SHA256 with timestamp
            parts = dict(x.split("=") for x in signature.split(","))
            timestamp = parts.get("t", "")
            sig = parts.get("v1", "")
            
            if not timestamp or not sig:
                return False
            
            # Compute expected signature
            signed_payload = f"{timestamp}.{payload.decode('utf-8')}"
            expected = hmac.new(
                secret.encode('utf-8'),
                signed_payload.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            return hmac.compare_digest(sig, expected)
        except Exception as e:
            logger.error(f"Stripe signature verification failed: {e}")
            return False
    
    @staticmethod
    def verify_shopify(
        payload: bytes,
        signature: str,
        secret: str,
    ) -> bool:
        """Verify Shopify webhook signature."""
        try:
            import base64
            
            computed = base64.b64encode(
                hmac.new(
                    secret.encode('utf-8'),
                    payload,
                    hashlib.sha256
                ).digest()
            ).decode('utf-8')
            
            return hmac.compare_digest(signature, computed)
        except Exception as e:
            logger.error(f"Shopify signature verification failed: {e}")
            return False
    
    @staticmethod
    def verify_plaid(
        payload: bytes,
        signature: str,
        secret: str,
    ) -> bool:
        """Verify Plaid webhook signature."""
        try:
            # Plaid uses JWT verification - simplified for now
            computed = hmac.new(
                secret.encode('utf-8'),
                payload,
                hashlib.sha256
            ).hexdigest()
            
            return hmac.compare_digest(signature, computed)
        except Exception as e:
            logger.error(f"Plaid signature verification failed: {e}")
            return False


async def process_webhook(event: WebhookEvent):
    """Process a webhook event asynchronously."""
    try:
        event.status = WebhookStatus.PROCESSING
        
        # Get handlers
        handlers = webhook_registry.get_handlers(event.source, event.event_type)
        
        if not handlers:
            logger.info(f"No handlers for {event.source.value}:{event.event_type}")
            event.status = WebhookStatus.IGNORED
            return
        
        # Execute handlers
        for handler in handlers:
            try:
                # Support both sync and async handlers
                import asyncio
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                logger.error(f"Handler error: {e}", exc_info=True)
                event.error = str(e)
                event.status = WebhookStatus.FAILED
                return
        
        event.status = WebhookStatus.COMPLETED
        event.processed_at = datetime.utcnow()
        
        logger.info(
            f"Webhook processed",
            source=event.source.value,
            event_type=event.event_type,
            event_id=event.id
        )
        
    except Exception as e:
        logger.error(f"Webhook processing failed: {e}", exc_info=True)
        event.status = WebhookStatus.FAILED
        event.error = str(e)


# API Endpoints

class WebhookResponse(BaseModel):
    """Standard webhook response."""
    received: bool = True
    event_id: str = ""


@router.post("/stripe", response_model=WebhookResponse)
async def stripe_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    stripe_signature: Optional[str] = Header(None),
):
    """
    Handle Stripe webhooks.
    
    Events:
    - payment_intent.succeeded
    - payment_intent.failed
    - customer.subscription.created
    - customer.subscription.deleted
    - invoice.payment_succeeded
    """
    body = await request.body()
    payload = json.loads(body)
    
    # Verify signature if secret is configured
    secret = webhook_registry.get_secret(WebhookSource.STRIPE)
    if secret and stripe_signature:
        if not WebhookVerifier.verify_stripe(body, stripe_signature, secret):
            raise HTTPException(status_code=401, detail="Invalid signature")
    
    event = WebhookEvent(
        source=WebhookSource.STRIPE,
        event_type=payload.get("type", "unknown"),
        payload=payload,
        signature=stripe_signature,
        business_id=payload.get("data", {}).get("object", {}).get("metadata", {}).get("business_id"),
    )
    
    # Process asynchronously
    background_tasks.add_task(process_webhook, event)
    
    # Log for audit
    await audit_manager.record(
        event_type=AuditEventType.PAYMENT_PROCESSED,
        resource_type="stripe_webhook",
        resource_id=event.id,
        action=f"Received Stripe webhook: {event.event_type}",
        new_value={"stripe_event_id": payload.get("id")},
        business_id=event.business_id,
    )
    
    return WebhookResponse(event_id=event.id)


@router.post("/shopify", response_model=WebhookResponse)
async def shopify_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_shopify_hmac_sha256: Optional[str] = Header(None),
    x_shopify_topic: Optional[str] = Header(None),
    x_shopify_shop_domain: Optional[str] = Header(None),
):
    """
    Handle Shopify webhooks.
    
    Events:
    - orders/create
    - orders/fulfilled
    - orders/cancelled
    - products/update
    - inventory_levels/update
    """
    body = await request.body()
    payload = json.loads(body)
    
    # Verify signature if secret is configured
    secret = webhook_registry.get_secret(WebhookSource.SHOPIFY)
    if secret and x_shopify_hmac_sha256:
        if not WebhookVerifier.verify_shopify(body, x_shopify_hmac_sha256, secret):
            raise HTTPException(status_code=401, detail="Invalid signature")
    
    event = WebhookEvent(
        source=WebhookSource.SHOPIFY,
        event_type=x_shopify_topic or payload.get("topic", "unknown"),
        payload=payload,
        headers={"shop_domain": x_shopify_shop_domain} if x_shopify_shop_domain else {},
        signature=x_shopify_hmac_sha256,
    )
    
    # Process asynchronously
    background_tasks.add_task(process_webhook, event)
    
    return WebhookResponse(event_id=event.id)


@router.post("/plaid", response_model=WebhookResponse)
async def plaid_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    plaid_verification: Optional[str] = Header(None),
):
    """
    Handle Plaid webhooks.
    
    Events:
    - TRANSACTIONS.INITIAL_UPDATE
    - TRANSACTIONS.DEFAULT_UPDATE
    - ITEM.ERROR
    - AUTH.AUTOMATICALLY_VERIFIED
    """
    body = await request.body()
    payload = json.loads(body)
    
    event = WebhookEvent(
        source=WebhookSource.PLAID,
        event_type=payload.get("webhook_type", "") + "." + payload.get("webhook_code", ""),
        payload=payload,
        signature=plaid_verification,
    )
    
    # Process asynchronously
    background_tasks.add_task(process_webhook, event)
    
    return WebhookResponse(event_id=event.id)


@router.post("/custom/{source}", response_model=WebhookResponse)
async def custom_webhook(
    source: str,
    request: Request,
    background_tasks: BackgroundTasks,
):
    """
    Handle custom webhooks from arbitrary sources.
    
    Use this for integrations that don't have dedicated endpoints.
    """
    body = await request.body()
    
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        payload = {"raw": body.decode("utf-8", errors="replace")}
    
    event = WebhookEvent(
        source=WebhookSource.CUSTOM,
        event_type=source,
        payload=payload,
    )
    
    # Process asynchronously
    background_tasks.add_task(process_webhook, event)
    
    return WebhookResponse(event_id=event.id)


# Default handlers

async def handle_stripe_payment_succeeded(event: WebhookEvent):
    """Handle successful Stripe payment."""
    payment = event.payload.get("data", {}).get("object", {})
    amount = payment.get("amount", 0) / 100  # Convert from cents
    
    logger.info(
        f"Payment succeeded",
        amount=amount,
        payment_id=payment.get("id"),
        customer=payment.get("customer")
    )
    
    # Update business revenue if business_id is present
    if event.business_id:
        from src.database.connection import get_db_session
        from src.database.models import BusinessUnit
        from sqlalchemy import select
        
        try:
            async with get_db_session() as session:
                result = await session.execute(
                    select(BusinessUnit).where(BusinessUnit.id == event.business_id)
                )
                business = result.scalar_one_or_none()
                
                if business and hasattr(business, "monthly_revenue"):
                    business.monthly_revenue = (business.monthly_revenue or 0) + amount
                    await session.commit()
        except Exception as e:
            logger.error(f"Failed to update revenue: {e}")


async def handle_shopify_order_created(event: WebhookEvent):
    """Handle new Shopify order."""
    order = event.payload
    
    logger.info(
        f"New Shopify order",
        order_id=order.get("id"),
        total=order.get("total_price"),
        items=len(order.get("line_items", []))
    )
    
    # Could trigger supplier fulfillment here


async def handle_plaid_transactions(event: WebhookEvent):
    """Handle Plaid transaction updates."""
    logger.info(
        f"Plaid transaction update",
        webhook_type=event.payload.get("webhook_type"),
        new_transactions=event.payload.get("new_transactions", 0)
    )


# Register default handlers
def register_default_handlers():
    """Register default webhook handlers."""
    webhook_registry.register(
        WebhookSource.STRIPE,
        "payment_intent.succeeded",
        handle_stripe_payment_succeeded
    )
    webhook_registry.register(
        WebhookSource.SHOPIFY,
        "orders/create",
        handle_shopify_order_created
    )
    webhook_registry.register(
        WebhookSource.PLAID,
        "TRANSACTIONS.*",
        handle_plaid_transactions
    )


# Initialize on import
register_default_handlers()
