"""
LLM Cost Tracker.
Tracks token usage and costs across LLM providers with budget alerts.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Coroutine, Dict, List, Optional
from enum import Enum
from collections import defaultdict

from src.utils.structured_logging import get_logger

logger = get_logger("cost_tracker")


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    OLLAMA = "ollama"
    VLLM = "vllm"


@dataclass
class ModelPricing:
    """Pricing for a specific model."""
    model: str
    provider: LLMProvider
    input_cost_per_1k: float  # Cost per 1000 input tokens
    output_cost_per_1k: float  # Cost per 1000 output tokens
    context_window: int = 4096
    
    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for a request."""
        input_cost = (input_tokens / 1000) * self.input_cost_per_1k
        output_cost = (output_tokens / 1000) * self.output_cost_per_1k
        return input_cost + output_cost


# Default pricing (as of early 2025 - should be updated periodically)
DEFAULT_PRICING = {
    # OpenAI
    "gpt-4-turbo": ModelPricing("gpt-4-turbo", LLMProvider.OPENAI, 0.01, 0.03, 128000),
    "gpt-4": ModelPricing("gpt-4", LLMProvider.OPENAI, 0.03, 0.06, 8192),
    "gpt-4o": ModelPricing("gpt-4o", LLMProvider.OPENAI, 0.0025, 0.01, 128000),
    "gpt-4o-mini": ModelPricing("gpt-4o-mini", LLMProvider.OPENAI, 0.00015, 0.0006, 128000),
    "gpt-3.5-turbo": ModelPricing("gpt-3.5-turbo", LLMProvider.OPENAI, 0.0005, 0.0015, 16384),
    
    # Anthropic
    "claude-3-opus": ModelPricing("claude-3-opus", LLMProvider.ANTHROPIC, 0.015, 0.075, 200000),
    "claude-3-sonnet": ModelPricing("claude-3-sonnet", LLMProvider.ANTHROPIC, 0.003, 0.015, 200000),
    "claude-3-haiku": ModelPricing("claude-3-haiku", LLMProvider.ANTHROPIC, 0.00025, 0.00125, 200000),
    "claude-3.5-sonnet": ModelPricing("claude-3.5-sonnet", LLMProvider.ANTHROPIC, 0.003, 0.015, 200000),
    
    # Google
    "gemini-pro": ModelPricing("gemini-pro", LLMProvider.GOOGLE, 0.00025, 0.0005, 32768),
    "gemini-1.5-pro": ModelPricing("gemini-1.5-pro", LLMProvider.GOOGLE, 0.00125, 0.005, 1000000),
    "gemini-1.5-flash": ModelPricing("gemini-1.5-flash", LLMProvider.GOOGLE, 0.000075, 0.0003, 1000000),
    
    # Local models (free)
    "ollama": ModelPricing("ollama", LLMProvider.OLLAMA, 0.0, 0.0, 4096),
    "vllm": ModelPricing("vllm", LLMProvider.VLLM, 0.0, 0.0, 4096),
}


@dataclass
class UsageRecord:
    """Record of a single LLM usage."""
    id: str
    model: str
    provider: LLMProvider
    input_tokens: int
    output_tokens: int
    cost: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    request_type: Optional[str] = None
    business_id: Optional[str] = None
    agent: Optional[str] = None
    latency_ms: float = 0.0
    success: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BudgetConfig:
    """Budget configuration."""
    daily_limit: float = 100.0
    weekly_limit: float = 500.0
    monthly_limit: float = 2000.0
    alert_threshold: float = 0.8  # Alert at 80% of limit
    hard_limit: bool = False  # Block requests over limit


class CostTracker:
    """
    Tracks LLM usage and costs with budget management.
    
    Features:
    - Per-request cost tracking
    - Budget limits and alerts
    - Usage analytics by model/agent/business
    - Cost optimization suggestions
    """
    
    def __init__(
        self,
        pricing: Optional[Dict[str, ModelPricing]] = None,
        budget: Optional[BudgetConfig] = None,
    ):
        self.pricing = pricing or DEFAULT_PRICING.copy()
        self.budget = budget or BudgetConfig()
        self._records: List[UsageRecord] = []
        self._alert_callbacks: List[Callable] = []
        self._lock = asyncio.Lock()
    
    def register_alert_callback(
        self,
        callback: Callable[[str, float, float], Coroutine[Any, Any, None]],
    ) -> None:
        """Register callback for budget alerts."""
        self._alert_callbacks.append(callback)
    
    def add_pricing(self, model: str, pricing: ModelPricing) -> None:
        """Add or update model pricing."""
        self.pricing[model] = pricing
    
    async def track(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        request_type: Optional[str] = None,
        business_id: Optional[str] = None,
        agent: Optional[str] = None,
        latency_ms: float = 0.0,
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> UsageRecord:
        """
        Track a single LLM request.
        
        Args:
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            request_type: Type of request (e.g., "chat", "completion")
            business_id: Associated business ID
            agent: Agent that made the request
            latency_ms: Request latency in milliseconds
            success: Whether the request succeeded
            metadata: Additional metadata
            
        Returns:
            The usage record
        """
        # Get pricing
        pricing = self._get_pricing(model)
        cost = pricing.calculate_cost(input_tokens, output_tokens) if pricing else 0.0
        
        # Create record
        record = UsageRecord(
            id=f"{datetime.utcnow().timestamp()}_{model}",
            model=model,
            provider=pricing.provider if pricing else LLMProvider.OLLAMA,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            request_type=request_type,
            business_id=business_id,
            agent=agent,
            latency_ms=latency_ms,
            success=success,
            metadata=metadata or {},
        )
        
        async with self._lock:
            self._records.append(record)
            
            # Trim old records (keep last 30 days)
            cutoff = datetime.utcnow() - timedelta(days=30)
            self._records = [r for r in self._records if r.timestamp > cutoff]
        
        # Check budget alerts
        await self._check_budget_alerts()
        
        logger.debug(
            f"Tracked LLM usage: {model}",
            cost=cost,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        
        return record
    
    def _get_pricing(self, model: str) -> Optional[ModelPricing]:
        """Get pricing for a model."""
        # Exact match
        if model in self.pricing:
            return self.pricing[model]
        
        # Partial match (for versioned models)
        for name, pricing in self.pricing.items():
            if name in model or model in name:
                return pricing
        
        return None
    
    async def _check_budget_alerts(self) -> None:
        """Check if budget alerts should be triggered."""
        daily_cost = self.get_cost(period="day")
        weekly_cost = self.get_cost(period="week")
        monthly_cost = self.get_cost(period="month")
        
        alerts = []
        
        if daily_cost >= self.budget.daily_limit * self.budget.alert_threshold:
            alerts.append(("daily", daily_cost, self.budget.daily_limit))
        
        if weekly_cost >= self.budget.weekly_limit * self.budget.alert_threshold:
            alerts.append(("weekly", weekly_cost, self.budget.weekly_limit))
        
        if monthly_cost >= self.budget.monthly_limit * self.budget.alert_threshold:
            alerts.append(("monthly", monthly_cost, self.budget.monthly_limit))
        
        for period, cost, limit in alerts:
            logger.warning(
                f"Budget alert: {period} cost ${cost:.2f} approaching limit ${limit:.2f}"
            )
            
            for callback in self._alert_callbacks:
                try:
                    await callback(period, cost, limit)
                except Exception as e:
                    logger.error(f"Alert callback error: {e}")
    
    def get_cost(
        self,
        period: str = "day",
        model: Optional[str] = None,
        provider: Optional[LLMProvider] = None,
        business_id: Optional[str] = None,
        agent: Optional[str] = None,
    ) -> float:
        """
        Get total cost for a period.
        
        Args:
            period: "hour", "day", "week", "month", "all"
            model: Filter by model
            provider: Filter by provider
            business_id: Filter by business
            agent: Filter by agent
            
        Returns:
            Total cost
        """
        records = self._filter_records(period, model, provider, business_id, agent)
        return sum(r.cost for r in records)
    
    def get_tokens(
        self,
        period: str = "day",
        model: Optional[str] = None,
    ) -> Dict[str, int]:
        """Get total tokens for a period."""
        records = self._filter_records(period, model=model)
        
        return {
            "input_tokens": sum(r.input_tokens for r in records),
            "output_tokens": sum(r.output_tokens for r in records),
            "total_tokens": sum(r.input_tokens + r.output_tokens for r in records),
        }
    
    def _filter_records(
        self,
        period: str,
        model: Optional[str] = None,
        provider: Optional[LLMProvider] = None,
        business_id: Optional[str] = None,
        agent: Optional[str] = None,
    ) -> List[UsageRecord]:
        """Filter records by criteria."""
        # Determine time cutoff
        now = datetime.utcnow()
        
        if period == "hour":
            cutoff = now - timedelta(hours=1)
        elif period == "day":
            cutoff = now - timedelta(days=1)
        elif period == "week":
            cutoff = now - timedelta(weeks=1)
        elif period == "month":
            cutoff = now - timedelta(days=30)
        else:  # "all"
            cutoff = datetime.min
        
        records = [r for r in self._records if r.timestamp > cutoff]
        
        if model:
            records = [r for r in records if r.model == model]
        
        if provider:
            records = [r for r in records if r.provider == provider]
        
        if business_id:
            records = [r for r in records if r.business_id == business_id]
        
        if agent:
            records = [r for r in records if r.agent == agent]
        
        return records
    
    def get_stats(self, period: str = "day") -> Dict[str, Any]:
        """Get comprehensive usage statistics."""
        records = self._filter_records(period)
        
        if not records:
            return {
                "period": period,
                "total_requests": 0,
                "total_cost": 0.0,
                "total_tokens": 0,
            }
        
        # Aggregate by model
        by_model = defaultdict(lambda: {"requests": 0, "cost": 0.0, "tokens": 0})
        for r in records:
            by_model[r.model]["requests"] += 1
            by_model[r.model]["cost"] += r.cost
            by_model[r.model]["tokens"] += r.input_tokens + r.output_tokens
        
        # Aggregate by agent
        by_agent = defaultdict(lambda: {"requests": 0, "cost": 0.0})
        for r in records:
            if r.agent:
                by_agent[r.agent]["requests"] += 1
                by_agent[r.agent]["cost"] += r.cost
        
        # Calculate averages
        total_cost = sum(r.cost for r in records)
        total_tokens = sum(r.input_tokens + r.output_tokens for r in records)
        avg_cost = total_cost / len(records) if records else 0
        avg_latency = sum(r.latency_ms for r in records) / len(records) if records else 0
        
        return {
            "period": period,
            "total_requests": len(records),
            "successful_requests": sum(1 for r in records if r.success),
            "total_cost": round(total_cost, 4),
            "total_tokens": total_tokens,
            "total_input_tokens": sum(r.input_tokens for r in records),
            "total_output_tokens": sum(r.output_tokens for r in records),
            "avg_cost_per_request": round(avg_cost, 6),
            "avg_latency_ms": round(avg_latency, 2),
            "by_model": dict(by_model),
            "by_agent": dict(by_agent),
            "budget": {
                "daily_used": self.get_cost("day"),
                "daily_limit": self.budget.daily_limit,
                "daily_remaining": max(0, self.budget.daily_limit - self.get_cost("day")),
                "weekly_used": self.get_cost("week"),
                "weekly_limit": self.budget.weekly_limit,
                "monthly_used": self.get_cost("month"),
                "monthly_limit": self.budget.monthly_limit,
            },
        }
    
    def is_within_budget(self) -> bool:
        """Check if current usage is within budget."""
        if not self.budget.hard_limit:
            return True
        
        return (
            self.get_cost("day") < self.budget.daily_limit and
            self.get_cost("week") < self.budget.weekly_limit and
            self.get_cost("month") < self.budget.monthly_limit
        )
    
    def get_optimization_suggestions(self) -> List[str]:
        """Get suggestions for cost optimization."""
        suggestions = []
        stats = self.get_stats("week")
        
        if not stats.get("by_model"):
            return suggestions
        
        # Check for expensive model usage
        for model, data in stats["by_model"].items():
            pricing = self._get_pricing(model)
            if pricing and pricing.input_cost_per_1k > 0.01:
                if data["requests"] > 100:
                    suggestions.append(
                        f"Consider using a cheaper model for some {model} requests. "
                        f"You've made {data['requests']} requests costing ${data['cost']:.2f}"
                    )
        
        # Check for high output token usage
        total_output = stats.get("total_output_tokens", 0)
        total_input = stats.get("total_input_tokens", 0)
        
        if total_output > total_input * 2:
            suggestions.append(
                "Output tokens significantly exceed input tokens. "
                "Consider using shorter response formats or max_tokens limits."
            )
        
        # Check budget utilization
        budget_stats = stats.get("budget", {})
        if budget_stats.get("daily_used", 0) > budget_stats.get("daily_limit", 1) * 0.9:
            suggestions.append(
                "Daily budget nearly exhausted. Consider batching requests or using caching."
            )
        
        return suggestions
    
    def export_records(
        self,
        period: str = "month",
        format: str = "dict",
    ) -> Any:
        """Export usage records."""
        records = self._filter_records(period)
        
        if format == "dict":
            return [
                {
                    "id": r.id,
                    "model": r.model,
                    "provider": r.provider.value,
                    "input_tokens": r.input_tokens,
                    "output_tokens": r.output_tokens,
                    "cost": r.cost,
                    "timestamp": r.timestamp.isoformat(),
                    "request_type": r.request_type,
                    "business_id": r.business_id,
                    "agent": r.agent,
                    "success": r.success,
                }
                for r in records
            ]
        
        return records


# Global cost tracker instance
cost_tracker = CostTracker()


def get_cost_tracker() -> CostTracker:
    """Get the global cost tracker instance."""
    return cost_tracker


# Decorator for automatic tracking
def track_llm_cost(
    model: str,
    agent: Optional[str] = None,
):
    """
    Decorator to automatically track LLM costs.
    
    Usage:
        @track_llm_cost("gpt-4", agent="research")
        async def generate_response(prompt: str) -> tuple[str, int, int]:
            # Returns (response, input_tokens, output_tokens)
            ...
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            start_time = datetime.utcnow()
            
            try:
                result = await func(*args, **kwargs)
                success = True
            except Exception as e:
                success = False
                raise
            finally:
                # Calculate latency
                latency_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                
                # Extract token counts from result
                if isinstance(result, tuple) and len(result) >= 3:
                    response, input_tokens, output_tokens = result[:3]
                else:
                    input_tokens = 0
                    output_tokens = 0
                
                await cost_tracker.track(
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    agent=agent,
                    latency_ms=latency_ms,
                    success=success,
                )
            
            return result
        
        return wrapper
    return decorator
