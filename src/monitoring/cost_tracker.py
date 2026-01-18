"""
Cost Tracking System.

Monitors LLM token usage, calculates costs, and enforces budgets.
Supports automatic fallback to local models when cloud budgets are exceeded.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path

from src.utils.structured_logging import get_logger

logger = get_logger("cost_tracker")


class ModelProvider(str, Enum):
    """LLM providers."""
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE = "azure"
    GOOGLE = "google"


@dataclass
class ModelPricing:
    """Pricing for a specific model."""
    model_name: str
    provider: ModelProvider
    input_cost_per_1k: float  # Cost per 1K input tokens
    output_cost_per_1k: float  # Cost per 1K output tokens
    is_local: bool = False  # Local models have zero cost
    
    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for a request."""
        if self.is_local:
            return 0.0
        input_cost = (input_tokens / 1000) * self.input_cost_per_1k
        output_cost = (output_tokens / 1000) * self.output_cost_per_1k
        return input_cost + output_cost


# Default pricing for common models
MODEL_PRICING: Dict[str, ModelPricing] = {
    # Ollama (local - free)
    "llama3.2:1b": ModelPricing("llama3.2:1b", ModelProvider.OLLAMA, 0.0, 0.0, True),
    "llama3.2:3b": ModelPricing("llama3.2:3b", ModelProvider.OLLAMA, 0.0, 0.0, True),
    "llama3.1:8b": ModelPricing("llama3.1:8b", ModelProvider.OLLAMA, 0.0, 0.0, True),
    "mistral:7b": ModelPricing("mistral:7b", ModelProvider.OLLAMA, 0.0, 0.0, True),
    "codellama:7b": ModelPricing("codellama:7b", ModelProvider.OLLAMA, 0.0, 0.0, True),
    
    # OpenAI
    "gpt-4o": ModelPricing("gpt-4o", ModelProvider.OPENAI, 0.005, 0.015),
    "gpt-4o-mini": ModelPricing("gpt-4o-mini", ModelProvider.OPENAI, 0.00015, 0.0006),
    "gpt-4-turbo": ModelPricing("gpt-4-turbo", ModelProvider.OPENAI, 0.01, 0.03),
    "gpt-3.5-turbo": ModelPricing("gpt-3.5-turbo", ModelProvider.OPENAI, 0.0005, 0.0015),
    
    # Anthropic
    "claude-3-opus": ModelPricing("claude-3-opus", ModelProvider.ANTHROPIC, 0.015, 0.075),
    "claude-3-sonnet": ModelPricing("claude-3-sonnet", ModelProvider.ANTHROPIC, 0.003, 0.015),
    "claude-3-haiku": ModelPricing("claude-3-haiku", ModelProvider.ANTHROPIC, 0.00025, 0.00125),
    
    # Google
    "gemini-pro": ModelPricing("gemini-pro", ModelProvider.GOOGLE, 0.00025, 0.0005),
    "gemini-flash": ModelPricing("gemini-flash", ModelProvider.GOOGLE, 0.000075, 0.0003),
}


@dataclass
class UsageRecord:
    """Record of a single usage event."""
    id: str
    timestamp: datetime
    user_id: str
    model: str
    input_tokens: int
    output_tokens: int
    cost: float
    task_type: str = ""
    business_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost": self.cost,
            "task_type": self.task_type,
            "business_id": self.business_id,
        }


@dataclass
class BudgetConfig:
    """Budget configuration for a user."""
    daily_limit: float = 10.0
    monthly_limit: float = 100.0
    daily_warning: float = 8.0
    monthly_warning: float = 80.0
    fallback_to_local: bool = True
    local_model: str = "llama3.2:1b"


@dataclass
class BudgetStatus:
    """Current budget status for a user."""
    user_id: str
    daily_spend: float = 0.0
    monthly_spend: float = 0.0
    daily_remaining: float = 0.0
    monthly_remaining: float = 0.0
    daily_warning: bool = False
    monthly_warning: bool = False
    can_use_cloud: bool = True
    reason: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "daily_spend": round(self.daily_spend, 4),
            "monthly_spend": round(self.monthly_spend, 4),
            "daily_remaining": round(self.daily_remaining, 4),
            "monthly_remaining": round(self.monthly_remaining, 4),
            "daily_warning": self.daily_warning,
            "monthly_warning": self.monthly_warning,
            "can_use_cloud": self.can_use_cloud,
            "reason": self.reason,
        }


class CostTracker:
    """
    Tracks token usage and costs with budget enforcement.
    
    Features:
    - Per-user token and cost tracking
    - Daily and monthly budget limits
    - Automatic fallback to local models
    - Usage analytics and reporting
    """
    
    def __init__(
        self,
        redis_client=None,
        storage_path: Optional[Path] = None,
        default_budget: Optional[BudgetConfig] = None
    ):
        """
        Initialize cost tracker.
        
        Args:
            redis_client: Redis for persistence
            storage_path: File-based storage path
            default_budget: Default budget configuration
        """
        self.redis = redis_client
        self.storage_path = storage_path or Path("data/costs")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.default_budget = default_budget or BudgetConfig()
        
        # In-memory caches
        self._daily_spend: Dict[str, float] = {}  # user:date -> spend
        self._monthly_spend: Dict[str, float] = {}  # user:month -> spend
        self._user_budgets: Dict[str, BudgetConfig] = {}
    
    async def track_usage(
        self,
        user_id: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        task_type: str = "",
        business_id: Optional[str] = None
    ) -> UsageRecord:
        """
        Track token usage and calculate cost.
        
        Args:
            user_id: User identifier
            model: Model name
            input_tokens: Input tokens used
            output_tokens: Output tokens generated
            task_type: Type of task
            business_id: Associated business
            
        Returns:
            Usage record
        """
        # Get pricing
        pricing = MODEL_PRICING.get(model)
        if not pricing:
            # Unknown model - assume local/free
            pricing = ModelPricing(model, ModelProvider.OLLAMA, 0.0, 0.0, True)
        
        cost = pricing.calculate_cost(input_tokens, output_tokens)
        
        record = UsageRecord(
            id=f"usage_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
            timestamp=datetime.utcnow(),
            user_id=user_id,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            task_type=task_type,
            business_id=business_id,
        )
        
        # Update spending
        if cost > 0:
            await self._update_spend(user_id, cost)
        
        # Check for budget warnings
        status = await self.get_budget_status(user_id)
        if status.daily_warning or status.monthly_warning:
            await self._send_budget_alert(user_id, status)
        
        # Log usage
        logger.debug(
            "Tracked usage",
            user_id=user_id,
            model=model,
            tokens=input_tokens + output_tokens,
            cost=f"${cost:.4f}",
        )
        
        return record
    
    async def get_budget_status(self, user_id: str) -> BudgetStatus:
        """
        Get current budget status for a user.
        
        Returns:
            Budget status with spending and limits
        """
        budget = self._user_budgets.get(user_id, self.default_budget)
        
        today = datetime.utcnow().strftime("%Y-%m-%d")
        month = datetime.utcnow().strftime("%Y-%m")
        
        daily_spend = await self._get_spend(user_id, "daily", today)
        monthly_spend = await self._get_spend(user_id, "monthly", month)
        
        daily_remaining = budget.daily_limit - daily_spend
        monthly_remaining = budget.monthly_limit - monthly_spend
        
        # Determine if can use cloud
        can_use_cloud = True
        reason = ""
        
        if daily_spend >= budget.daily_limit:
            can_use_cloud = False
            reason = f"Daily limit (${budget.daily_limit:.2f}) exceeded"
        elif monthly_spend >= budget.monthly_limit:
            can_use_cloud = False
            reason = f"Monthly limit (${budget.monthly_limit:.2f}) exceeded"
        
        return BudgetStatus(
            user_id=user_id,
            daily_spend=daily_spend,
            monthly_spend=monthly_spend,
            daily_remaining=max(0, daily_remaining),
            monthly_remaining=max(0, monthly_remaining),
            daily_warning=daily_spend >= budget.daily_warning,
            monthly_warning=monthly_spend >= budget.monthly_warning,
            can_use_cloud=can_use_cloud,
            reason=reason,
        )
    
    async def get_recommended_model(
        self,
        user_id: str,
        preferred_model: str
    ) -> str:
        """
        Get recommended model based on budget.
        
        If budget exceeded, returns local fallback model.
        
        Args:
            user_id: User identifier
            preferred_model: Preferred model to use
            
        Returns:
            Model to actually use
        """
        # Check if preferred model is local (always allowed)
        pricing = MODEL_PRICING.get(preferred_model)
        if pricing and pricing.is_local:
            return preferred_model
        
        # Check budget
        status = await self.get_budget_status(user_id)
        
        if not status.can_use_cloud:
            budget = self._user_budgets.get(user_id, self.default_budget)
            if budget.fallback_to_local:
                logger.warning(
                    "Budget exceeded, falling back to local model",
                    user_id=user_id,
                    preferred=preferred_model,
                    fallback=budget.local_model,
                    reason=status.reason,
                )
                return budget.local_model
        
        return preferred_model
    
    async def set_user_budget(
        self,
        user_id: str,
        budget: BudgetConfig
    ) -> None:
        """Set custom budget for a user."""
        self._user_budgets[user_id] = budget
        
        # Persist to storage
        budgets_file = self.storage_path / "user_budgets.json"
        try:
            if budgets_file.exists():
                with open(budgets_file) as f:
                    all_budgets = json.load(f)
            else:
                all_budgets = {}
            
            all_budgets[user_id] = {
                "daily_limit": budget.daily_limit,
                "monthly_limit": budget.monthly_limit,
                "daily_warning": budget.daily_warning,
                "monthly_warning": budget.monthly_warning,
                "fallback_to_local": budget.fallback_to_local,
                "local_model": budget.local_model,
            }
            
            with open(budgets_file, 'w') as f:
                json.dump(all_budgets, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to persist budget: {e}")
    
    async def get_usage_report(
        self,
        user_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get usage report for a user.
        
        Returns:
            Report with totals by model, day, etc.
        """
        if not start_date:
            start_date = datetime.utcnow() - timedelta(days=30)
        if not end_date:
            end_date = datetime.utcnow()
        
        # Load usage records
        records = await self._load_usage_records(user_id, start_date, end_date)
        
        # Aggregate
        by_model: Dict[str, Dict[str, float]] = {}
        by_day: Dict[str, float] = {}
        by_task: Dict[str, float] = {}
        total_tokens = 0
        total_cost = 0.0
        
        for record in records:
            # By model
            if record.model not in by_model:
                by_model[record.model] = {"tokens": 0, "cost": 0.0}
            by_model[record.model]["tokens"] += record.input_tokens + record.output_tokens
            by_model[record.model]["cost"] += record.cost
            
            # By day
            day = record.timestamp.strftime("%Y-%m-%d")
            by_day[day] = by_day.get(day, 0.0) + record.cost
            
            # By task
            if record.task_type:
                by_task[record.task_type] = by_task.get(record.task_type, 0.0) + record.cost
            
            total_tokens += record.input_tokens + record.output_tokens
            total_cost += record.cost
        
        return {
            "user_id": user_id,
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
            },
            "totals": {
                "tokens": total_tokens,
                "cost": round(total_cost, 4),
            },
            "by_model": by_model,
            "by_day": by_day,
            "by_task": by_task,
        }
    
    # Private methods
    
    async def _update_spend(self, user_id: str, cost: float) -> None:
        """Update spending totals."""
        today = datetime.utcnow().strftime("%Y-%m-%d")
        month = datetime.utcnow().strftime("%Y-%m")
        
        daily_key = f"{user_id}:{today}"
        monthly_key = f"{user_id}:{month}"
        
        if self.redis:
            await self.redis.incrbyfloat(f"cost:daily:{daily_key}", cost)
            await self.redis.incrbyfloat(f"cost:monthly:{monthly_key}", cost)
            await self.redis.expire(f"cost:daily:{daily_key}", 86400 * 7)
            await self.redis.expire(f"cost:monthly:{monthly_key}", 86400 * 60)
        else:
            self._daily_spend[daily_key] = self._daily_spend.get(daily_key, 0.0) + cost
            self._monthly_spend[monthly_key] = self._monthly_spend.get(monthly_key, 0.0) + cost
    
    async def _get_spend(
        self,
        user_id: str,
        period: str,
        key: str
    ) -> float:
        """Get spending for a period."""
        full_key = f"{user_id}:{key}"
        
        if self.redis:
            value = await self.redis.get(f"cost:{period}:{full_key}")
            return float(value) if value else 0.0
        else:
            if period == "daily":
                return self._daily_spend.get(full_key, 0.0)
            else:
                return self._monthly_spend.get(full_key, 0.0)
    
    async def _send_budget_alert(
        self,
        user_id: str,
        status: BudgetStatus
    ) -> None:
        """Send budget warning alert."""
        logger.warning(
            "Budget warning",
            user_id=user_id,
            daily_spend=f"${status.daily_spend:.2f}",
            monthly_spend=f"${status.monthly_spend:.2f}",
            daily_warning=status.daily_warning,
            monthly_warning=status.monthly_warning,
        )
        # TODO: Send notification via notification service
    
    async def _load_usage_records(
        self,
        user_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[UsageRecord]:
        """Load usage records for reporting."""
        # For now, return empty - would load from DB in production
        return []


# Global instance
_cost_tracker: Optional[CostTracker] = None


def get_cost_tracker() -> CostTracker:
    """Get or create the global cost tracker."""
    global _cost_tracker
    if _cost_tracker is None:
        _cost_tracker = CostTracker()
    return _cost_tracker
