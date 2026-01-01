"""
Feature Flag System.
Runtime feature toggles for gradual rollouts.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable, Set
from enum import Enum
import hashlib
import json

from src.utils.structured_logging import get_logger

logger = get_logger("feature_flags")


class FlagType(str, Enum):
    """Types of feature flags."""
    BOOLEAN = "boolean"  # Simple on/off
    PERCENTAGE = "percentage"  # Percentage rollout
    USER_LIST = "user_list"  # Specific users
    VARIANT = "variant"  # A/B variants


class FlagStatus(str, Enum):
    """Status of a feature flag."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ARCHIVED = "archived"


@dataclass
class FlagRule:
    """A targeting rule for a feature flag."""
    id: str
    attribute: str  # user attribute to check
    operator: str  # eq, neq, contains, gt, lt, in
    value: Any
    enabled: bool = True
    
    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Evaluate rule against context."""
        if not self.enabled:
            return False
        
        actual = context.get(self.attribute)
        if actual is None:
            return False
        
        if self.operator == "eq":
            return actual == self.value
        elif self.operator == "neq":
            return actual != self.value
        elif self.operator == "contains":
            return self.value in str(actual)
        elif self.operator == "gt":
            return actual > self.value
        elif self.operator == "lt":
            return actual < self.value
        elif self.operator == "in":
            return actual in self.value
        elif self.operator == "not_in":
            return actual not in self.value
        
        return False


@dataclass
class Variant:
    """A feature variant."""
    id: str
    name: str
    weight: float = 1.0  # Relative weight
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureFlag:
    """A feature flag definition."""
    key: str
    name: str
    description: str = ""
    flag_type: FlagType = FlagType.BOOLEAN
    status: FlagStatus = FlagStatus.INACTIVE
    
    # Boolean flags
    default_value: bool = False
    
    # Percentage rollout
    percentage: float = 0.0
    
    # User targeting
    allowed_users: Set[str] = field(default_factory=set)
    blocked_users: Set[str] = field(default_factory=set)
    
    # Rules
    rules: List[FlagRule] = field(default_factory=list)
    
    # Variants (for A/B testing)
    variants: List[Variant] = field(default_factory=list)
    
    # Metadata
    owner: str = ""
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    # Schedule
    enable_at: Optional[datetime] = None
    disable_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "name": self.name,
            "description": self.description,
            "flag_type": self.flag_type.value,
            "status": self.status.value,
            "default_value": self.default_value,
            "percentage": self.percentage,
            "allowed_users": list(self.allowed_users),
            "owner": self.owner,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class FlagEvaluation:
    """Result of evaluating a feature flag."""
    flag_key: str
    enabled: bool
    variant: Optional[str] = None
    reason: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


class FeatureFlagService:
    """
    Feature Flag Service.
    
    Features:
    - Boolean flags
    - Percentage rollouts
    - User targeting
    - Rule-based targeting
    - A/B variants
    - Scheduled flags
    """
    
    def __init__(self):
        self.flags: Dict[str, FeatureFlag] = {}
        self.evaluation_cache: Dict[str, FlagEvaluation] = {}
        self.cache_ttl_seconds = 60
        self.callbacks: Dict[str, List[Callable]] = {}
        
        self._setup_default_flags()
    
    def _setup_default_flags(self) -> None:
        """Set up default feature flags."""
        # Example flags
        self.create_flag(
            key="new_dashboard",
            name="New Dashboard",
            description="Enable the new dashboard UI",
            flag_type=FlagType.PERCENTAGE,
            percentage=10.0,
            tags=["ui", "beta"],
        )
        
        self.create_flag(
            key="dark_mode",
            name="Dark Mode",
            description="Enable dark mode theme",
            flag_type=FlagType.BOOLEAN,
            default_value=True,
            status=FlagStatus.ACTIVE,
            tags=["ui"],
        )
    
    def create_flag(
        self,
        key: str,
        name: str,
        description: str = "",
        flag_type: FlagType = FlagType.BOOLEAN,
        status: FlagStatus = FlagStatus.INACTIVE,
        default_value: bool = False,
        percentage: float = 0.0,
        allowed_users: List[str] = None,
        owner: str = "",
        tags: List[str] = None,
    ) -> FeatureFlag:
        """
        Create a new feature flag.
        
        Args:
            key: Unique flag identifier
            name: Display name
            description: Flag description
            flag_type: Type of flag
            status: Initial status
            default_value: Default value for boolean flags
            percentage: Rollout percentage
            allowed_users: List of allowed user IDs
            owner: Flag owner
            tags: Flag tags
            
        Returns:
            Created feature flag
        """
        flag = FeatureFlag(
            key=key,
            name=name,
            description=description,
            flag_type=flag_type,
            status=status,
            default_value=default_value,
            percentage=percentage,
            allowed_users=set(allowed_users or []),
            owner=owner,
            tags=tags or [],
        )
        
        self.flags[key] = flag
        logger.info(f"Created feature flag: {key}")
        
        return flag
    
    def is_enabled(
        self,
        key: str,
        context: Dict[str, Any] = None,
        default: bool = False,
    ) -> bool:
        """
        Check if a feature flag is enabled.
        
        Args:
            key: Flag key
            context: Evaluation context (user_id, attributes, etc.)
            default: Default value if flag not found
            
        Returns:
            Whether the flag is enabled
        """
        evaluation = self.evaluate(key, context)
        if evaluation is None:
            return default
        return evaluation.enabled
    
    def evaluate(
        self,
        key: str,
        context: Dict[str, Any] = None,
    ) -> Optional[FlagEvaluation]:
        """
        Evaluate a feature flag.
        
        Args:
            key: Flag key
            context: Evaluation context
            
        Returns:
            Flag evaluation result
        """
        context = context or {}
        flag = self.flags.get(key)
        
        if not flag:
            logger.warning(f"Feature flag not found: {key}")
            return None
        
        # Check cache
        cache_key = self._cache_key(key, context)
        cached = self.evaluation_cache.get(cache_key)
        if cached:
            age = (datetime.utcnow() - cached.timestamp).total_seconds()
            if age < self.cache_ttl_seconds:
                return cached
        
        # Evaluate
        evaluation = self._evaluate_flag(flag, context)
        
        # Cache result
        self.evaluation_cache[cache_key] = evaluation
        
        # Clean old cache entries
        self._clean_cache()
        
        return evaluation
    
    def _evaluate_flag(
        self,
        flag: FeatureFlag,
        context: Dict[str, Any],
    ) -> FlagEvaluation:
        """Evaluate a flag with context."""
        # Check status
        if flag.status == FlagStatus.INACTIVE:
            return FlagEvaluation(
                flag_key=flag.key,
                enabled=False,
                reason="Flag is inactive",
                context=context,
            )
        
        if flag.status == FlagStatus.ARCHIVED:
            return FlagEvaluation(
                flag_key=flag.key,
                enabled=False,
                reason="Flag is archived",
                context=context,
            )
        
        # Check schedule
        now = datetime.utcnow()
        if flag.enable_at and now < flag.enable_at:
            return FlagEvaluation(
                flag_key=flag.key,
                enabled=False,
                reason="Flag not yet enabled (scheduled)",
                context=context,
            )
        
        if flag.disable_at and now > flag.disable_at:
            return FlagEvaluation(
                flag_key=flag.key,
                enabled=False,
                reason="Flag disabled (schedule expired)",
                context=context,
            )
        
        # Get user ID from context
        user_id = context.get("user_id", "")
        
        # Check blocked users
        if user_id and user_id in flag.blocked_users:
            return FlagEvaluation(
                flag_key=flag.key,
                enabled=False,
                reason="User is blocked",
                context=context,
            )
        
        # Check allowed users
        if user_id and user_id in flag.allowed_users:
            return FlagEvaluation(
                flag_key=flag.key,
                enabled=True,
                reason="User is in allowed list",
                context=context,
            )
        
        # Evaluate rules
        for rule in flag.rules:
            if rule.evaluate(context):
                return FlagEvaluation(
                    flag_key=flag.key,
                    enabled=True,
                    reason=f"Rule matched: {rule.id}",
                    context=context,
                )
        
        # Evaluate based on flag type
        if flag.flag_type == FlagType.BOOLEAN:
            return FlagEvaluation(
                flag_key=flag.key,
                enabled=flag.default_value,
                reason="Boolean default value",
                context=context,
            )
        
        elif flag.flag_type == FlagType.PERCENTAGE:
            enabled = self._percentage_rollout(flag.key, user_id, flag.percentage)
            return FlagEvaluation(
                flag_key=flag.key,
                enabled=enabled,
                reason=f"Percentage rollout ({flag.percentage}%)",
                context=context,
            )
        
        elif flag.flag_type == FlagType.USER_LIST:
            enabled = user_id in flag.allowed_users
            return FlagEvaluation(
                flag_key=flag.key,
                enabled=enabled,
                reason="User list check",
                context=context,
            )
        
        elif flag.flag_type == FlagType.VARIANT:
            variant = self._select_variant(flag, user_id)
            return FlagEvaluation(
                flag_key=flag.key,
                enabled=variant is not None,
                variant=variant.id if variant else None,
                reason="Variant selection",
                context=context,
            )
        
        return FlagEvaluation(
            flag_key=flag.key,
            enabled=False,
            reason="No matching conditions",
            context=context,
        )
    
    def _percentage_rollout(
        self,
        flag_key: str,
        user_id: str,
        percentage: float,
    ) -> bool:
        """Determine if user is in percentage rollout."""
        if not user_id:
            return False
        
        if percentage >= 100:
            return True
        
        if percentage <= 0:
            return False
        
        # Consistent hashing for stable assignment
        hash_input = f"{flag_key}:{user_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        bucket = hash_value % 100
        
        return bucket < percentage
    
    def _select_variant(
        self,
        flag: FeatureFlag,
        user_id: str,
    ) -> Optional[Variant]:
        """Select a variant for a user."""
        if not flag.variants:
            return None
        
        if not user_id:
            return flag.variants[0]
        
        # Calculate total weight
        total_weight = sum(v.weight for v in flag.variants)
        if total_weight == 0:
            return flag.variants[0]
        
        # Consistent hashing
        hash_input = f"{flag.key}:variant:{user_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        bucket = (hash_value % 10000) / 100  # 0-100 with 2 decimal precision
        
        # Select variant
        cumulative = 0
        for variant in flag.variants:
            cumulative += (variant.weight / total_weight) * 100
            if bucket < cumulative:
                return variant
        
        return flag.variants[-1]
    
    def get_variant(
        self,
        key: str,
        context: Dict[str, Any] = None,
    ) -> Optional[str]:
        """Get the selected variant for a flag."""
        evaluation = self.evaluate(key, context)
        return evaluation.variant if evaluation else None
    
    def get_variant_payload(
        self,
        key: str,
        context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Get the payload for the selected variant."""
        variant_id = self.get_variant(key, context)
        if not variant_id:
            return {}
        
        flag = self.flags.get(key)
        if not flag:
            return {}
        
        variant = next((v for v in flag.variants if v.id == variant_id), None)
        return variant.payload if variant else {}
    
    def enable_flag(self, key: str) -> bool:
        """Enable a feature flag."""
        flag = self.flags.get(key)
        if not flag:
            return False
        
        flag.status = FlagStatus.ACTIVE
        flag.updated_at = datetime.utcnow()
        self._invalidate_cache(key)
        self._trigger_callbacks(key, "enabled")
        
        logger.info(f"Enabled feature flag: {key}")
        return True
    
    def disable_flag(self, key: str) -> bool:
        """Disable a feature flag."""
        flag = self.flags.get(key)
        if not flag:
            return False
        
        flag.status = FlagStatus.INACTIVE
        flag.updated_at = datetime.utcnow()
        self._invalidate_cache(key)
        self._trigger_callbacks(key, "disabled")
        
        logger.info(f"Disabled feature flag: {key}")
        return True
    
    def set_percentage(self, key: str, percentage: float) -> bool:
        """Set rollout percentage for a flag."""
        flag = self.flags.get(key)
        if not flag:
            return False
        
        flag.percentage = max(0, min(100, percentage))
        flag.updated_at = datetime.utcnow()
        self._invalidate_cache(key)
        
        logger.info(f"Set {key} percentage to {percentage}%")
        return True
    
    def add_user(self, key: str, user_id: str) -> bool:
        """Add a user to allowed list."""
        flag = self.flags.get(key)
        if not flag:
            return False
        
        flag.allowed_users.add(user_id)
        flag.updated_at = datetime.utcnow()
        self._invalidate_cache(key)
        
        return True
    
    def remove_user(self, key: str, user_id: str) -> bool:
        """Remove a user from allowed list."""
        flag = self.flags.get(key)
        if not flag:
            return False
        
        flag.allowed_users.discard(user_id)
        flag.updated_at = datetime.utcnow()
        self._invalidate_cache(key)
        
        return True
    
    def add_rule(
        self,
        key: str,
        rule_id: str,
        attribute: str,
        operator: str,
        value: Any,
    ) -> bool:
        """Add a targeting rule to a flag."""
        flag = self.flags.get(key)
        if not flag:
            return False
        
        rule = FlagRule(
            id=rule_id,
            attribute=attribute,
            operator=operator,
            value=value,
        )
        flag.rules.append(rule)
        flag.updated_at = datetime.utcnow()
        self._invalidate_cache(key)
        
        return True
    
    def on_change(self, key: str, callback: Callable) -> None:
        """Register a callback for flag changes."""
        if key not in self.callbacks:
            self.callbacks[key] = []
        self.callbacks[key].append(callback)
    
    def _trigger_callbacks(self, key: str, event: str) -> None:
        """Trigger callbacks for a flag."""
        for callback in self.callbacks.get(key, []):
            try:
                callback(key, event)
            except Exception as e:
                logger.error(f"Callback error for {key}: {e}")
    
    def _cache_key(self, key: str, context: Dict[str, Any]) -> str:
        """Generate cache key."""
        context_str = json.dumps(context, sort_keys=True, default=str)
        return f"{key}:{hashlib.md5(context_str.encode()).hexdigest()}"
    
    def _invalidate_cache(self, key: str) -> None:
        """Invalidate cache for a flag."""
        to_remove = [k for k in self.evaluation_cache if k.startswith(f"{key}:")]
        for k in to_remove:
            del self.evaluation_cache[k]
    
    def _clean_cache(self) -> None:
        """Clean expired cache entries."""
        now = datetime.utcnow()
        to_remove = [
            k for k, v in self.evaluation_cache.items()
            if (now - v.timestamp).total_seconds() > self.cache_ttl_seconds * 2
        ]
        for k in to_remove:
            del self.evaluation_cache[k]
    
    def get_flag(self, key: str) -> Optional[FeatureFlag]:
        """Get a feature flag."""
        return self.flags.get(key)
    
    def list_flags(
        self,
        status: Optional[FlagStatus] = None,
        tag: Optional[str] = None,
    ) -> List[FeatureFlag]:
        """List feature flags."""
        flags = list(self.flags.values())
        
        if status:
            flags = [f for f in flags if f.status == status]
        
        if tag:
            flags = [f for f in flags if tag in f.tags]
        
        return flags
    
    def export_flags(self) -> Dict[str, Any]:
        """Export all flags as dict."""
        return {key: flag.to_dict() for key, flag in self.flags.items()}


# Global feature flag service
feature_flags = FeatureFlagService()


def get_feature_flags() -> FeatureFlagService:
    """Get the global feature flag service."""
    return feature_flags


# Convenience functions
def is_enabled(key: str, context: Dict[str, Any] = None, default: bool = False) -> bool:
    """Check if a feature flag is enabled."""
    return feature_flags.is_enabled(key, context, default)


def get_variant(key: str, context: Dict[str, Any] = None) -> Optional[str]:
    """Get the variant for a feature flag."""
    return feature_flags.get_variant(key, context)
