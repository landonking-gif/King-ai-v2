"""
Finance Agent - Payment processing, subscriptions, and revenue tracking.
Integrates with Stripe for payment processing and financial management.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Optional
from src.agents.base import SubAgent
from src.integrations.stripe_client import (
    StripeClient, StripeCustomer, PaymentIntent, Subscription,
    PaymentStatus, SubscriptionStatus
)
from src.utils.metrics import TASKS_EXECUTED
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


class FinanceAgent(SubAgent):
    """
    The 'CFO' sub-agent for the empire.
    Handles payment processing, subscriptions, and financial tracking.
    """
    name = "finance"
    description = "Manages financial tracking, payment processing, and profit analysis with Stripe integration."
    
    def __init__(self):
        super().__init__()
        self._clients: dict[str, StripeClient] = {}
    
    async def execute(self, task: dict) -> dict:
        """
        Executes a financial task - either traditional analysis or Stripe operations.
        """
        # Check if this is a Stripe operation
        action = task.get("action")
        if action:
            return await self._execute_stripe_action(task)
        
        # Otherwise, fall back to traditional financial analysis
        description = task.get("description", "Financial task")
        input_data = task.get("input_data", {})
        
        prompt = f"""
        ### TASK: FINANCIAL ANALYSIS
        {description}
        
        ### FINANCIAL DATA:
        {input_data}
        
        ### INSTRUCTION:
        Analyze the financial implications and provide recommendations.
        Focus on ROI, cost-cutting, and revenue growth.
        """
        
        try:
            result = await self._ask_llm(prompt)
            TASKS_EXECUTED.labels(agent=self.name, status="success").inc()
            return {
                "success": True, 
                "output": result, 
                "metadata": {"type": "finance_analysis"}
            }
        except Exception as e:
            TASKS_EXECUTED.labels(agent=self.name, status="failed").inc()
            return {"success": False, "error": str(e)}
    
    async def _execute_stripe_action(self, task: dict) -> dict:
        """Execute Stripe-specific actions."""
        action = task.get("action", "")
        business_id = task.get("business_id", "")
        
        try:
            if action == "create_payment":
                return await self.create_payment(
                    business_id, task["amount"], task.get("currency", "usd"),
                    task.get("customer_id"), task.get("description")
                )
            elif action == "refund":
                return await self.process_refund(
                    business_id, task["payment_intent_id"], task.get("amount")
                )
            elif action == "create_subscription":
                return await self.create_subscription(
                    business_id, task["customer_id"], task["price_id"]
                )
            elif action == "cancel_subscription":
                return await self.cancel_subscription(
                    business_id, task["subscription_id"]
                )
            elif action == "metrics":
                return await self.get_revenue_metrics(business_id)
            elif action == "balance":
                return await self.get_balance(business_id)
            
            return {"success": False, "error": f"Unknown action: {action}"}
        except Exception as e:
            TASKS_EXECUTED.labels(agent=self.name, status="failed").inc()
            return {"success": False, "error": str(e)}
    
    # Stripe Initialization
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
    ) -> dict:
        client = self._get_client(business_id)
        if not client:
            return {"success": False, "error": "Stripe not configured"}
        
        try:
            customer = await client.create_customer(email, name, metadata)
            return {
                "success": True,
                "message": "Customer created",
                "data": {"customer_id": customer.id, "email": customer.email}
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def get_customer(self, business_id: str, customer_id: str) -> dict:
        client = self._get_client(business_id)
        if not client:
            return {"success": False, "error": "Stripe not configured"}
        
        customer = await client.get_customer(customer_id)
        if not customer:
            return {"success": False, "error": "Customer not found"}
        
        return {
            "success": True,
            "data": {
                "id": customer.id,
                "email": customer.email,
                "name": customer.name,
                "metadata": customer.metadata,
            }
        }

    # Payment Operations
    async def create_payment(
        self,
        business_id: str,
        amount: float,
        currency: str = "usd",
        customer_id: str = None,
        description: str = None,
        metadata: dict = None,
    ) -> dict:
        client = self._get_client(business_id)
        if not client:
            return {"success": False, "error": "Stripe not configured"}
        
        try:
            amount_cents = int(amount * 100)
            intent = await client.create_payment_intent(
                amount=amount_cents,
                currency=currency,
                customer_id=customer_id,
                description=description,
                metadata=metadata,
            )
            return {
                "success": True,
                "message": "Payment intent created",
                "data": {
                    "payment_intent_id": intent.id,
                    "amount": intent.amount_dollars,
                    "status": intent.status.value,
                }
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def process_refund(
        self,
        business_id: str,
        payment_intent_id: str,
        amount: float = None,
        reason: str = None,
    ) -> dict:
        client = self._get_client(business_id)
        if not client:
            return {"success": False, "error": "Stripe not configured"}
        
        try:
            amount_cents = int(amount * 100) if amount else None
            refund = await client.create_refund(payment_intent_id, amount_cents, reason)
            return {
                "success": True,
                "message": "Refund processed",
                "data": {"refund_id": refund["id"], "amount": refund["amount"] / 100}
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # Subscription Operations
    async def create_subscription(
        self,
        business_id: str,
        customer_id: str,
        price_id: str,
        trial_days: int = None,
        metadata: dict = None,
    ) -> dict:
        client = self._get_client(business_id)
        if not client:
            return {"success": False, "error": "Stripe not configured"}
        
        try:
            sub = await client.create_subscription(
                customer_id, price_id, trial_days, metadata
            )
            return {
                "success": True,
                "message": "Subscription created",
                "data": {
                    "subscription_id": sub.id,
                    "status": sub.status.value,
                    "current_period_end": sub.current_period_end.isoformat(),
                }
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def cancel_subscription(
        self, business_id: str, subscription_id: str, immediate: bool = False
    ) -> dict:
        client = self._get_client(business_id)
        if not client:
            return {"success": False, "error": "Stripe not configured"}
        
        try:
            sub = await client.cancel_subscription(subscription_id, at_period_end=not immediate)
            return {
                "success": True,
                "message": "Subscription canceled",
                "data": {
                    "subscription_id": sub.id,
                    "status": sub.status.value,
                    "cancel_at_period_end": sub.cancel_at_period_end,
                }
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def upgrade_subscription(
        self, business_id: str, subscription_id: str, new_price_id: str
    ) -> dict:
        client = self._get_client(business_id)
        if not client:
            return {"success": False, "error": "Stripe not configured"}
        
        try:
            sub = await client.update_subscription(subscription_id, price_id=new_price_id)
            return {
                "success": True,
                "message": "Subscription upgraded",
                "data": {"subscription_id": sub.id, "new_price": new_price_id}
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # Revenue & Reporting
    async def get_revenue_metrics(self, business_id: str) -> dict:
        client = self._get_client(business_id)
        if not client:
            return {"success": False, "error": "Stripe not configured"}
        
        try:
            balance = await client.get_balance()
            subscriptions = await client.list_subscriptions(status="active", limit=100)
            
            available = sum(b["amount"] for b in balance.get("available", []))
            pending = sum(b["amount"] for b in balance.get("pending", []))
            
            # Calculate MRR from actual subscription data
            # Note: This is a simplified calculation. In production, you'd query
            # subscription prices from Stripe to get accurate MRR values
            mrr = len(subscriptions) * 2999  # TODO: Calculate from actual subscription prices
            
            # TODO: Implement churn_rate calculation based on subscription cancellations
            # TODO: Implement average_order_value from payment intent history
            metrics = RevenueMetrics(
                total_revenue=available / 100,
                monthly_recurring=mrr / 100,
                active_subscriptions=len(subscriptions),
                churn_rate=0.0,  # Placeholder - needs historical data
                average_order_value=0.0,  # Placeholder - needs payment history
                pending_payouts=pending / 100,
            )
            
            return {
                "success": True,
                "data": {
                    "total_revenue": metrics.total_revenue,
                    "monthly_recurring": metrics.monthly_recurring,
                    "active_subscriptions": metrics.active_subscriptions,
                    "pending_payouts": metrics.pending_payouts,
                }
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def get_balance(self, business_id: str) -> dict:
        client = self._get_client(business_id)
        if not client:
            return {"success": False, "error": "Stripe not configured"}
        
        try:
            balance = await client.get_balance()
            available = sum(b["amount"] for b in balance.get("available", [])) / 100
            pending = sum(b["amount"] for b in balance.get("pending", [])) / 100
            
            return {
                "success": True,
                "data": {"available": available, "pending": pending}
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def request_payout(
        self, business_id: str, amount: float, currency: str = "usd"
    ) -> dict:
        client = self._get_client(business_id)
        if not client:
            return {"success": False, "error": "Stripe not configured"}
        
        try:
            payout = await client.create_payout(int(amount * 100), currency)
            return {
                "success": True,
                "message": "Payout initiated",
                "data": {"payout_id": payout["id"], "amount": payout["amount"] / 100}
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def list_invoices(
        self, business_id: str, customer_id: str = None, limit: int = 10
    ) -> dict:
        client = self._get_client(business_id)
        if not client:
            return {"success": False, "error": "Stripe not configured"}
        
        try:
            invoices = await client.list_invoices(customer_id=customer_id, limit=limit)
            return {
                "success": True,
                "data": {
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
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # Webhook Processing
    async def handle_webhook(
        self, business_id: str, payload: bytes, signature: str
    ) -> dict:
        client = self._get_client(business_id)
        if not client:
            return {"success": False, "error": "Stripe not configured"}
        
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
            
            return {
                "success": True,
                "message": f"Processed {event_type}",
                "data": {"event_type": event_type, "event_id": event.get("id")}
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

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
