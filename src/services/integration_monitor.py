"""
Integration Health Monitor.

Monitors the health of all external integrations and alerts on failures.
Registers with the scheduler for periodic health checks.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
import httpx

from src.utils.circuit_breaker import CircuitBreaker, CircuitState
from src.utils.structured_logging import get_logger
from config.settings import settings

logger = get_logger("integration_monitor")


class IntegrationStatus(str, Enum):
    """Status of an integration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class IntegrationType(str, Enum):
    """Types of integrations."""
    PAYMENT = "payment"
    BANKING = "banking"
    ECOMMERCE = "ecommerce"
    LLM = "llm"
    ANALYTICS = "analytics"
    VECTOR_DB = "vector_db"
    NOTIFICATION = "notification"


@dataclass
class IntegrationHealth:
    """Health status of a single integration."""
    name: str
    type: IntegrationType
    status: IntegrationStatus
    latency_ms: Optional[float] = None
    last_check: Optional[datetime] = None
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    error_message: Optional[str] = None
    consecutive_failures: int = 0
    circuit_state: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthAlert:
    """Alert for integration health issues."""
    integration: str
    severity: str  # "warning", "critical"
    message: str
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)


class IntegrationHealthMonitor:
    """
    Monitors all external integrations and provides alerting.
    
    Features:
    - Periodic health checks for all integrations
    - Circuit breaker status monitoring
    - Latency tracking
    - Alert generation for failures
    - Integration with scheduler for autonomous monitoring
    """
    
    def __init__(self):
        self._health: Dict[str, IntegrationHealth] = {}
        self._alerts: List[HealthAlert] = []
        self._alert_callbacks: List[Callable] = []
        self._check_timeout = 10.0  # seconds
        
        # Initialize known integrations
        self._init_integrations()
    
    def _init_integrations(self):
        """Initialize health tracking for known integrations."""
        integrations = [
            ("stripe", IntegrationType.PAYMENT),
            ("paypal", IntegrationType.PAYMENT),
            ("plaid", IntegrationType.BANKING),
            ("shopify", IntegrationType.ECOMMERCE),
            ("ollama", IntegrationType.LLM),
            ("vllm", IntegrationType.LLM),
            ("claude", IntegrationType.LLM),
            ("gemini", IntegrationType.LLM),
            ("pinecone", IntegrationType.VECTOR_DB),
            ("google_analytics", IntegrationType.ANALYTICS),
            ("twilio", IntegrationType.NOTIFICATION),
            ("smtp", IntegrationType.NOTIFICATION),
        ]
        
        for name, itype in integrations:
            self._health[name] = IntegrationHealth(
                name=name,
                type=itype,
                status=IntegrationStatus.UNKNOWN
            )
    
    def register_alert_callback(self, callback: Callable[[HealthAlert], None]):
        """Register a callback to be called when alerts are generated."""
        self._alert_callbacks.append(callback)
    
    async def check_all(self) -> Dict[str, IntegrationHealth]:
        """
        Check health of all integrations.
        
        Returns:
            Dictionary of integration name to health status
        """
        logger.info("Starting integration health check")
        
        # Run all checks concurrently
        checks = [
            self._check_stripe(),
            self._check_paypal(),
            self._check_plaid(),
            self._check_shopify(),
            self._check_ollama(),
            self._check_pinecone(),
            self._check_circuit_breakers(),
        ]
        
        await asyncio.gather(*checks, return_exceptions=True)
        
        # Generate alerts for unhealthy integrations
        await self._generate_alerts()
        
        # Log summary
        healthy = sum(1 for h in self._health.values() if h.status == IntegrationStatus.HEALTHY)
        total = len(self._health)
        logger.info(
            "Integration health check complete",
            healthy=healthy,
            total=total,
            unhealthy=[n for n, h in self._health.items() if h.status == IntegrationStatus.UNHEALTHY]
        )
        
        return self._health.copy()
    
    async def _check_stripe(self):
        """Check Stripe API health."""
        if not settings.stripe_api_key:
            self._health["stripe"].status = IntegrationStatus.UNKNOWN
            return
        
        try:
            start = datetime.utcnow()
            async with httpx.AsyncClient(timeout=self._check_timeout) as client:
                response = await client.get(
                    "https://api.stripe.com/v1/balance",
                    headers={"Authorization": f"Bearer {settings.stripe_api_key}"}
                )
            latency = (datetime.utcnow() - start).total_seconds() * 1000
            
            if response.status_code == 200:
                self._update_health("stripe", IntegrationStatus.HEALTHY, latency)
            else:
                self._update_health(
                    "stripe",
                    IntegrationStatus.DEGRADED,
                    latency,
                    f"HTTP {response.status_code}"
                )
        except Exception as e:
            self._update_health("stripe", IntegrationStatus.UNHEALTHY, error=str(e))
    
    async def _check_paypal(self):
        """Check PayPal API health."""
        if not settings.paypal_client_id:
            self._health["paypal"].status = IntegrationStatus.UNKNOWN
            return
        
        try:
            start = datetime.utcnow()
            # Just check if sandbox/production endpoint is reachable
            async with httpx.AsyncClient(timeout=self._check_timeout) as client:
                response = await client.get("https://api-m.sandbox.paypal.com/v1/")
            latency = (datetime.utcnow() - start).total_seconds() * 1000
            
            # PayPal returns 404 on root but that means it's reachable
            if response.status_code in [200, 404]:
                self._update_health("paypal", IntegrationStatus.HEALTHY, latency)
            else:
                self._update_health("paypal", IntegrationStatus.DEGRADED, latency)
        except Exception as e:
            self._update_health("paypal", IntegrationStatus.UNHEALTHY, error=str(e))
    
    async def _check_plaid(self):
        """Check Plaid API health."""
        if not settings.plaid_client_id:
            self._health["plaid"].status = IntegrationStatus.UNKNOWN
            return
        
        try:
            start = datetime.utcnow()
            async with httpx.AsyncClient(timeout=self._check_timeout) as client:
                # Check Plaid sandbox endpoint
                response = await client.get("https://sandbox.plaid.com/")
            latency = (datetime.utcnow() - start).total_seconds() * 1000
            
            self._update_health("plaid", IntegrationStatus.HEALTHY, latency)
        except Exception as e:
            self._update_health("plaid", IntegrationStatus.UNHEALTHY, error=str(e))
    
    async def _check_shopify(self):
        """Check Shopify API health."""
        shop_url = settings.shopify_shop_url
        access_token = settings.shopify_access_token
        
        if not shop_url or not access_token:
            self._health["shopify"].status = IntegrationStatus.UNKNOWN
            return
        
        try:
            start = datetime.utcnow()
            api_version = settings.shopify_api_version or "2024-10"
            url = f"https://{shop_url}/admin/api/{api_version}/shop.json"
            
            async with httpx.AsyncClient(timeout=self._check_timeout) as client:
                response = await client.get(
                    url,
                    headers={"X-Shopify-Access-Token": access_token}
                )
            latency = (datetime.utcnow() - start).total_seconds() * 1000
            
            if response.status_code == 200:
                self._update_health("shopify", IntegrationStatus.HEALTHY, latency)
            else:
                self._update_health(
                    "shopify",
                    IntegrationStatus.DEGRADED,
                    latency,
                    f"HTTP {response.status_code}"
                )
        except Exception as e:
            self._update_health("shopify", IntegrationStatus.UNHEALTHY, error=str(e))
    
    async def _check_ollama(self):
        """Check Ollama LLM health."""
        if not settings.ollama_url:
            self._health["ollama"].status = IntegrationStatus.UNKNOWN
            return
        
        try:
            start = datetime.utcnow()
            async with httpx.AsyncClient(timeout=self._check_timeout) as client:
                response = await client.get(f"{settings.ollama_url}/api/tags")
            latency = (datetime.utcnow() - start).total_seconds() * 1000
            
            if response.status_code == 200:
                data = response.json()
                model_count = len(data.get("models", []))
                self._update_health(
                    "ollama",
                    IntegrationStatus.HEALTHY,
                    latency,
                    metadata={"models_available": model_count}
                )
            else:
                self._update_health("ollama", IntegrationStatus.DEGRADED, latency)
        except Exception as e:
            self._update_health("ollama", IntegrationStatus.UNHEALTHY, error=str(e))
    
    async def _check_pinecone(self):
        """Check Pinecone vector store health."""
        if not settings.pinecone_api_key:
            self._health["pinecone"].status = IntegrationStatus.UNKNOWN
            return
        
        try:
            start = datetime.utcnow()
            async with httpx.AsyncClient(timeout=self._check_timeout) as client:
                response = await client.get(
                    "https://api.pinecone.io/indexes",
                    headers={"Api-Key": settings.pinecone_api_key}
                )
            latency = (datetime.utcnow() - start).total_seconds() * 1000
            
            if response.status_code == 200:
                self._update_health("pinecone", IntegrationStatus.HEALTHY, latency)
            else:
                self._update_health("pinecone", IntegrationStatus.DEGRADED, latency)
        except Exception as e:
            self._update_health("pinecone", IntegrationStatus.UNHEALTHY, error=str(e))
    
    async def _check_circuit_breakers(self):
        """Update health from circuit breaker states."""
        for name, cb in CircuitBreaker._registry.items():
            if name in self._health:
                self._health[name].circuit_state = cb.state.value
                
                # If circuit is open, mark as unhealthy
                if cb.state == CircuitState.OPEN:
                    self._health[name].status = IntegrationStatus.UNHEALTHY
                    self._health[name].consecutive_failures = cb.stats.consecutive_failures
    
    def _update_health(
        self,
        name: str,
        status: IntegrationStatus,
        latency: float = None,
        error: str = None,
        metadata: Dict[str, Any] = None
    ):
        """Update health status for an integration."""
        health = self._health.get(name)
        if not health:
            return
        
        health.status = status
        health.latency_ms = latency
        health.last_check = datetime.utcnow()
        
        if status == IntegrationStatus.HEALTHY:
            health.last_success = datetime.utcnow()
            health.consecutive_failures = 0
            health.error_message = None
        else:
            health.last_failure = datetime.utcnow()
            health.consecutive_failures += 1
            health.error_message = error
        
        if metadata:
            health.metadata.update(metadata)
    
    async def _generate_alerts(self):
        """Generate alerts for unhealthy integrations."""
        for name, health in self._health.items():
            if health.status == IntegrationStatus.UNHEALTHY:
                # Critical alert for payment providers
                if health.type == IntegrationType.PAYMENT:
                    severity = "critical"
                else:
                    severity = "warning"
                
                alert = HealthAlert(
                    integration=name,
                    severity=severity,
                    message=f"Integration {name} is unhealthy: {health.error_message or 'Unknown error'}",
                    timestamp=datetime.utcnow(),
                    details={
                        "consecutive_failures": health.consecutive_failures,
                        "last_success": health.last_success.isoformat() if health.last_success else None,
                        "circuit_state": health.circuit_state
                    }
                )
                
                self._alerts.append(alert)
                
                # Notify callbacks
                for callback in self._alert_callbacks:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(alert)
                        else:
                            callback(alert)
                    except Exception as e:
                        logger.error(f"Alert callback failed: {e}")
                
                logger.warning(
                    f"Integration alert: {alert.message}",
                    integration=name,
                    severity=severity
                )
    
    def get_status(self) -> Dict[str, Any]:
        """Get overall integration status summary."""
        statuses = {}
        for status in IntegrationStatus:
            statuses[status.value] = sum(
                1 for h in self._health.values()
                if h.status == status
            )
        
        unhealthy_critical = [
            h.name for h in self._health.values()
            if h.status == IntegrationStatus.UNHEALTHY
            and h.type in [IntegrationType.PAYMENT, IntegrationType.BANKING]
        ]
        
        return {
            "overall": "critical" if unhealthy_critical else (
                "degraded" if statuses.get("unhealthy", 0) > 0 else "healthy"
            ),
            "summary": statuses,
            "integrations": {
                name: {
                    "status": h.status.value,
                    "type": h.type.value,
                    "latency_ms": h.latency_ms,
                    "last_check": h.last_check.isoformat() if h.last_check else None,
                    "error": h.error_message,
                    "circuit_state": h.circuit_state
                }
                for name, h in self._health.items()
            },
            "recent_alerts": [
                {
                    "integration": a.integration,
                    "severity": a.severity,
                    "message": a.message,
                    "timestamp": a.timestamp.isoformat()
                }
                for a in self._alerts[-10:]
            ]
        }
    
    def get_payment_status(self) -> Dict[str, Any]:
        """
        Get payment provider status with fallback recommendations.
        
        Returns:
            Status of payment providers and recommended provider
        """
        stripe_health = self._health.get("stripe")
        paypal_health = self._health.get("paypal")
        
        providers = {}
        if stripe_health:
            providers["stripe"] = {
                "status": stripe_health.status.value,
                "available": stripe_health.status == IntegrationStatus.HEALTHY
            }
        if paypal_health:
            providers["paypal"] = {
                "status": paypal_health.status.value,
                "available": paypal_health.status == IntegrationStatus.HEALTHY
            }
        
        # Determine recommended provider
        recommended = None
        if providers.get("stripe", {}).get("available"):
            recommended = "stripe"
        elif providers.get("paypal", {}).get("available"):
            recommended = "paypal"
        
        return {
            "providers": providers,
            "recommended": recommended,
            "all_down": recommended is None
        }


# Global instance
integration_monitor = IntegrationHealthMonitor()


async def register_with_scheduler():
    """Register integration health check with the scheduler."""
    from src.services.scheduler import scheduler, TaskFrequency
    
    scheduler.register_task(
        name="Integration Health Check",
        callback=integration_monitor.check_all,
        frequency=TaskFrequency.EVERY_5_MINUTES,
        enabled=True,
        run_immediately=True,
        metadata={"type": "health_check", "category": "integrations"}
    )
    
    logger.info("Registered integration health monitor with scheduler")
