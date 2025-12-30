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
    if not result.get("success"):
        raise HTTPException(400, result.get("error", "Unknown error"))
    return result.get("data", {})


@router.get("/customers/{business_id}/{customer_id}")
async def get_customer(business_id: str, customer_id: str, agent: FinanceAgent = Depends(get_agent)):
    result = await agent.get_customer(business_id, customer_id)
    if not result.get("success"):
        raise HTTPException(404, result.get("error", "Customer not found"))
    return result.get("data", {})


@router.post("/payments")
async def create_payment(req: PaymentRequest, agent: FinanceAgent = Depends(get_agent)):
    result = await agent.create_payment(
        req.business_id, req.amount, req.currency, req.customer_id, req.description
    )
    if not result.get("success"):
        raise HTTPException(400, result.get("error", "Payment failed"))
    return result.get("data", {})


@router.post("/refunds")
async def process_refund(req: RefundRequest, agent: FinanceAgent = Depends(get_agent)):
    result = await agent.process_refund(
        req.business_id, req.payment_intent_id, req.amount, req.reason
    )
    if not result.get("success"):
        raise HTTPException(400, result.get("error", "Refund failed"))
    return result.get("data", {})


@router.post("/subscriptions")
async def create_subscription(req: SubscriptionRequest, agent: FinanceAgent = Depends(get_agent)):
    result = await agent.create_subscription(
        req.business_id, req.customer_id, req.price_id, req.trial_days
    )
    if not result.get("success"):
        raise HTTPException(400, result.get("error", "Subscription creation failed"))
    return result.get("data", {})


@router.post("/subscriptions/cancel")
async def cancel_subscription(req: CancelSubscriptionRequest, agent: FinanceAgent = Depends(get_agent)):
    result = await agent.cancel_subscription(req.business_id, req.subscription_id, req.immediate)
    if not result.get("success"):
        raise HTTPException(400, result.get("error", "Subscription cancellation failed"))
    return result.get("data", {})


@router.get("/balance/{business_id}")
async def get_balance(business_id: str, agent: FinanceAgent = Depends(get_agent)):
    result = await agent.get_balance(business_id)
    if not result.get("success"):
        raise HTTPException(400, result.get("error", "Failed to get balance"))
    return result.get("data", {})


@router.get("/metrics/{business_id}")
async def get_metrics(business_id: str, agent: FinanceAgent = Depends(get_agent)):
    result = await agent.get_revenue_metrics(business_id)
    if not result.get("success"):
        raise HTTPException(400, result.get("error", "Failed to get metrics"))
    return result.get("data", {})


@router.post("/payouts")
async def request_payout(req: PayoutRequest, agent: FinanceAgent = Depends(get_agent)):
    result = await agent.request_payout(req.business_id, req.amount, req.currency)
    if not result.get("success"):
        raise HTTPException(400, result.get("error", "Payout request failed"))
    return result.get("data", {})


@router.get("/invoices/{business_id}")
async def list_invoices(
    business_id: str,
    customer_id: Optional[str] = None,
    limit: int = 10,
    agent: FinanceAgent = Depends(get_agent)
):
    result = await agent.list_invoices(business_id, customer_id, limit)
    if not result.get("success"):
        raise HTTPException(400, result.get("error", "Failed to list invoices"))
    return result.get("data", {})


@router.post("/webhook/{business_id}")
async def stripe_webhook(
    business_id: str,
    request: Request,
    stripe_signature: str = Header(..., alias="Stripe-Signature"),
    agent: FinanceAgent = Depends(get_agent)
):
    payload = await request.body()
    result = await agent.handle_webhook(business_id, payload, stripe_signature)
    if not result.get("success"):
        raise HTTPException(400, result.get("error", "Webhook processing failed"))
    return {"received": True}
