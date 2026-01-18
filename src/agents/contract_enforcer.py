"""
Contract Enforcer.

Validates agent actions against role contracts.
Based on mother-harness enforcement patterns.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from uuid import uuid4

from src.utils.structured_logging import get_logger

logger = get_logger("contract_enforcer")


class ActionCategory(str, Enum):
    """Categories of agent actions."""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    EXECUTE = "execute"
    EXTERNAL = "external"
    FINANCIAL = "financial"
    ADMIN = "admin"


class ContractViolationType(str, Enum):
    """Types of contract violations."""
    ACTION_NOT_ALLOWED = "action_not_allowed"
    MISSING_CAPABILITY = "missing_capability"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    INVALID_INPUT = "invalid_input"
    MISSING_ARTIFACT = "missing_artifact"
    APPROVAL_REQUIRED = "approval_required"
    BUDGET_EXCEEDED = "budget_exceeded"
    TIME_RESTRICTION = "time_restriction"


@dataclass
class ContractViolation:
    """A contract violation."""
    id: str = field(default_factory=lambda: f"viol_{uuid4().hex[:8]}")
    violation_type: ContractViolationType = ContractViolationType.ACTION_NOT_ALLOWED
    agent_id: str = ""
    action: str = ""
    description: str = ""
    severity: str = "medium"
    timestamp: datetime = field(default_factory=datetime.utcnow)
    data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "violation_type": self.violation_type.value,
            "agent_id": self.agent_id,
            "action": self.action,
            "description": self.description,
            "severity": self.severity,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
        }


@dataclass
class ActionLimit:
    """Limit on actions."""
    action: str
    max_per_minute: int = 60
    max_per_hour: int = 1000
    max_per_day: int = 10000
    requires_approval: bool = False
    approval_threshold: int = None  # Require approval after this many


@dataclass
class AgentContract:
    """Contract defining what an agent can do."""
    agent_id: str
    name: str = ""
    description: str = ""
    
    # Capabilities
    capabilities: Set[str] = field(default_factory=set)
    
    # Allowed actions (by category or specific action names)
    allowed_categories: Set[ActionCategory] = field(default_factory=set)
    allowed_actions: Set[str] = field(default_factory=set)
    blocked_actions: Set[str] = field(default_factory=set)
    
    # Required inputs/outputs
    required_inputs: List[str] = field(default_factory=list)
    required_outputs: List[str] = field(default_factory=list)
    
    # Rate limits
    action_limits: Dict[str, ActionLimit] = field(default_factory=dict)
    
    # Budget
    max_cost_per_request: float = 10.0
    max_cost_per_hour: float = 100.0
    max_cost_per_day: float = 1000.0
    
    # Time restrictions
    allowed_hours: Optional[tuple] = None  # (start_hour, end_hour)
    allowed_days: Optional[Set[int]] = None  # 0=Monday, 6=Sunday
    
    # Approval requirements
    actions_requiring_approval: Set[str] = field(default_factory=set)
    
    # Retry limits
    max_retries: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "description": self.description,
            "capabilities": list(self.capabilities),
            "allowed_categories": [c.value for c in self.allowed_categories],
            "allowed_actions": list(self.allowed_actions),
            "blocked_actions": list(self.blocked_actions),
            "required_inputs": self.required_inputs,
            "required_outputs": self.required_outputs,
            "max_cost_per_request": self.max_cost_per_request,
            "max_retries": self.max_retries,
        }


@dataclass
class EnforcementResult:
    """Result of enforcement check."""
    allowed: bool
    violations: List[ContractViolation] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    approval_required: bool = False
    approval_reason: Optional[str] = None
    
    @property
    def has_violations(self) -> bool:
        return len(self.violations) > 0


class ContractEnforcer:
    """
    Enforces agent contracts.
    
    Features:
    - Action allowlisting/blocklisting
    - Rate limiting
    - Budget enforcement
    - Time-based restrictions
    - Approval workflows
    - Violation tracking
    """
    
    def __init__(
        self,
        violation_callback: Callable[[ContractViolation], None] = None,
    ):
        """
        Initialize the enforcer.
        
        Args:
            violation_callback: Called when violations occur
        """
        self._contracts: Dict[str, AgentContract] = {}
        self._violation_callback = violation_callback
        
        # Track usage for rate limiting
        self._usage: Dict[str, Dict[str, List[datetime]]] = {}  # agent_id -> action -> timestamps
        
        # Track costs
        self._costs: Dict[str, Dict[str, float]] = {}  # agent_id -> period -> cost
        
        # Violation history
        self._violations: List[ContractViolation] = []
    
    def register_contract(self, contract: AgentContract) -> None:
        """Register an agent contract."""
        self._contracts[contract.agent_id] = contract
        self._usage[contract.agent_id] = {}
        self._costs[contract.agent_id] = {
            "hour": 0.0,
            "day": 0.0,
        }
        logger.info(f"Registered contract for agent: {contract.agent_id}")
    
    def get_contract(self, agent_id: str) -> Optional[AgentContract]:
        """Get an agent's contract."""
        return self._contracts.get(agent_id)
    
    def check_action(
        self,
        agent_id: str,
        action: str,
        category: ActionCategory = None,
        inputs: Dict[str, Any] = None,
        estimated_cost: float = 0.0,
    ) -> EnforcementResult:
        """
        Check if an action is allowed.
        
        Args:
            agent_id: Agent attempting the action
            action: Action name
            category: Action category
            inputs: Action inputs
            estimated_cost: Estimated cost
            
        Returns:
            Enforcement result
        """
        contract = self._contracts.get(agent_id)
        
        # No contract = default allow
        if not contract:
            return EnforcementResult(allowed=True)
        
        violations = []
        warnings = []
        approval_required = False
        approval_reason = None
        
        # Check if action is blocked
        if action in contract.blocked_actions:
            violations.append(ContractViolation(
                violation_type=ContractViolationType.ACTION_NOT_ALLOWED,
                agent_id=agent_id,
                action=action,
                description=f"Action '{action}' is blocked for agent {agent_id}",
                severity="high",
            ))
        
        # Check if action is allowed
        action_allowed = (
            action in contract.allowed_actions or
            (category and category in contract.allowed_categories) or
            (not contract.allowed_actions and not contract.allowed_categories)  # No restrictions
        )
        
        if not action_allowed and action not in contract.blocked_actions:
            violations.append(ContractViolation(
                violation_type=ContractViolationType.ACTION_NOT_ALLOWED,
                agent_id=agent_id,
                action=action,
                description=f"Action '{action}' not in allowed list for agent {agent_id}",
                severity="medium",
            ))
        
        # Check required inputs
        if contract.required_inputs and inputs:
            for required in contract.required_inputs:
                if required not in inputs:
                    violations.append(ContractViolation(
                        violation_type=ContractViolationType.MISSING_ARTIFACT,
                        agent_id=agent_id,
                        action=action,
                        description=f"Missing required input: {required}",
                        severity="medium",
                    ))
        
        # Check rate limits
        rate_violation = self._check_rate_limits(contract, agent_id, action)
        if rate_violation:
            violations.append(rate_violation)
        
        # Check budget
        budget_violation = self._check_budget(contract, agent_id, estimated_cost)
        if budget_violation:
            violations.append(budget_violation)
        
        # Check time restrictions
        time_violation = self._check_time_restrictions(contract, agent_id, action)
        if time_violation:
            violations.append(time_violation)
        
        # Check if approval required
        if action in contract.actions_requiring_approval:
            approval_required = True
            approval_reason = f"Action '{action}' requires approval"
        
        # Check rate-based approval threshold
        if action in contract.action_limits:
            limit = contract.action_limits[action]
            if limit.approval_threshold:
                usage_count = len(self._usage.get(agent_id, {}).get(action, []))
                if usage_count >= limit.approval_threshold:
                    approval_required = True
                    approval_reason = f"Approval threshold ({limit.approval_threshold}) reached for '{action}'"
        
        # Record violations
        for violation in violations:
            self._violations.append(violation)
            if self._violation_callback:
                try:
                    self._violation_callback(violation)
                except Exception as e:
                    logger.warning(f"Violation callback failed: {e}")
        
        return EnforcementResult(
            allowed=len(violations) == 0,
            violations=violations,
            warnings=warnings,
            approval_required=approval_required,
            approval_reason=approval_reason,
        )
    
    def record_action(
        self,
        agent_id: str,
        action: str,
        cost: float = 0.0,
    ) -> None:
        """Record an action execution for rate limiting."""
        if agent_id not in self._usage:
            self._usage[agent_id] = {}
        
        if action not in self._usage[agent_id]:
            self._usage[agent_id][action] = []
        
        self._usage[agent_id][action].append(datetime.utcnow())
        
        # Record cost
        if agent_id in self._costs:
            self._costs[agent_id]["hour"] += cost
            self._costs[agent_id]["day"] += cost
    
    def reset_rate_limits(self, agent_id: str) -> None:
        """Reset rate limits for an agent."""
        if agent_id in self._usage:
            self._usage[agent_id] = {}
    
    def reset_costs(self, agent_id: str, period: str = "all") -> None:
        """Reset cost tracking for an agent."""
        if agent_id in self._costs:
            if period == "all":
                self._costs[agent_id] = {"hour": 0.0, "day": 0.0}
            else:
                self._costs[agent_id][period] = 0.0
    
    def get_violations(
        self,
        agent_id: str = None,
        since: datetime = None,
    ) -> List[ContractViolation]:
        """Get violation history."""
        violations = self._violations
        
        if agent_id:
            violations = [v for v in violations if v.agent_id == agent_id]
        
        if since:
            violations = [v for v in violations if v.timestamp >= since]
        
        return violations
    
    def get_usage_stats(self, agent_id: str) -> Dict[str, Any]:
        """Get usage statistics for an agent."""
        usage = self._usage.get(agent_id, {})
        costs = self._costs.get(agent_id, {})
        
        action_counts = {}
        for action, timestamps in usage.items():
            action_counts[action] = len(timestamps)
        
        return {
            "agent_id": agent_id,
            "action_counts": action_counts,
            "costs": costs,
            "violation_count": len([v for v in self._violations if v.agent_id == agent_id]),
        }
    
    # Private methods
    
    def _check_rate_limits(
        self,
        contract: AgentContract,
        agent_id: str,
        action: str,
    ) -> Optional[ContractViolation]:
        """Check rate limits."""
        if action not in contract.action_limits:
            return None
        
        limit = contract.action_limits[action]
        now = datetime.utcnow()
        usage = self._usage.get(agent_id, {}).get(action, [])
        
        # Clean old entries
        minute_ago = now.replace(second=now.second - 60) if now.second >= 60 else now.replace(minute=now.minute - 1)
        hour_ago = now.replace(hour=now.hour - 1) if now.hour >= 1 else now.replace(hour=23)
        
        # Count recent usage
        minute_count = sum(1 for ts in usage if ts >= minute_ago)
        hour_count = sum(1 for ts in usage if ts >= hour_ago)
        
        if minute_count >= limit.max_per_minute:
            return ContractViolation(
                violation_type=ContractViolationType.RATE_LIMIT_EXCEEDED,
                agent_id=agent_id,
                action=action,
                description=f"Rate limit exceeded: {minute_count}/{limit.max_per_minute} per minute",
                severity="medium",
                data={"limit": limit.max_per_minute, "actual": minute_count, "period": "minute"},
            )
        
        if hour_count >= limit.max_per_hour:
            return ContractViolation(
                violation_type=ContractViolationType.RATE_LIMIT_EXCEEDED,
                agent_id=agent_id,
                action=action,
                description=f"Rate limit exceeded: {hour_count}/{limit.max_per_hour} per hour",
                severity="medium",
                data={"limit": limit.max_per_hour, "actual": hour_count, "period": "hour"},
            )
        
        return None
    
    def _check_budget(
        self,
        contract: AgentContract,
        agent_id: str,
        estimated_cost: float,
    ) -> Optional[ContractViolation]:
        """Check budget constraints."""
        if estimated_cost > contract.max_cost_per_request:
            return ContractViolation(
                violation_type=ContractViolationType.BUDGET_EXCEEDED,
                agent_id=agent_id,
                action="budget_check",
                description=f"Estimated cost ${estimated_cost:.2f} exceeds per-request limit ${contract.max_cost_per_request:.2f}",
                severity="high",
                data={"estimated": estimated_cost, "limit": contract.max_cost_per_request},
            )
        
        costs = self._costs.get(agent_id, {})
        hour_cost = costs.get("hour", 0.0) + estimated_cost
        day_cost = costs.get("day", 0.0) + estimated_cost
        
        if hour_cost > contract.max_cost_per_hour:
            return ContractViolation(
                violation_type=ContractViolationType.BUDGET_EXCEEDED,
                agent_id=agent_id,
                action="budget_check",
                description=f"Hourly cost ${hour_cost:.2f} exceeds limit ${contract.max_cost_per_hour:.2f}",
                severity="high",
                data={"projected": hour_cost, "limit": contract.max_cost_per_hour},
            )
        
        if day_cost > contract.max_cost_per_day:
            return ContractViolation(
                violation_type=ContractViolationType.BUDGET_EXCEEDED,
                agent_id=agent_id,
                action="budget_check",
                description=f"Daily cost ${day_cost:.2f} exceeds limit ${contract.max_cost_per_day:.2f}",
                severity="high",
                data={"projected": day_cost, "limit": contract.max_cost_per_day},
            )
        
        return None
    
    def _check_time_restrictions(
        self,
        contract: AgentContract,
        agent_id: str,
        action: str,
    ) -> Optional[ContractViolation]:
        """Check time-based restrictions."""
        now = datetime.utcnow()
        
        # Check allowed hours
        if contract.allowed_hours:
            start_hour, end_hour = contract.allowed_hours
            if not (start_hour <= now.hour < end_hour):
                return ContractViolation(
                    violation_type=ContractViolationType.TIME_RESTRICTION,
                    agent_id=agent_id,
                    action=action,
                    description=f"Action not allowed outside hours {start_hour}:00-{end_hour}:00",
                    severity="low",
                    data={"current_hour": now.hour, "allowed": contract.allowed_hours},
                )
        
        # Check allowed days
        if contract.allowed_days:
            if now.weekday() not in contract.allowed_days:
                return ContractViolation(
                    violation_type=ContractViolationType.TIME_RESTRICTION,
                    agent_id=agent_id,
                    action=action,
                    description=f"Action not allowed on this day of week",
                    severity="low",
                    data={"current_day": now.weekday(), "allowed_days": list(contract.allowed_days)},
                )
        
        return None


# Default contracts for common agent types
def create_research_agent_contract(agent_id: str) -> AgentContract:
    """Create a standard research agent contract."""
    return AgentContract(
        agent_id=agent_id,
        name="Research Agent",
        description="Can read and analyze but not write or execute",
        capabilities={"research", "analysis", "summarization"},
        allowed_categories={ActionCategory.READ, ActionCategory.EXTERNAL},
        blocked_actions={"delete_file", "execute_code", "modify_database"},
        max_cost_per_request=1.0,
        max_cost_per_hour=50.0,
    )


def create_code_agent_contract(agent_id: str) -> AgentContract:
    """Create a standard code agent contract."""
    return AgentContract(
        agent_id=agent_id,
        name="Code Agent",
        description="Can read and write code but not execute or delete",
        capabilities={"code_generation", "code_review", "refactoring"},
        allowed_categories={ActionCategory.READ, ActionCategory.WRITE},
        allowed_actions={"read_file", "write_file", "create_file"},
        blocked_actions={"execute_code", "delete_file", "shell_command"},
        actions_requiring_approval={"write_file", "create_file"},
        max_cost_per_request=2.0,
        max_cost_per_hour=100.0,
    )


def create_finance_agent_contract(agent_id: str) -> AgentContract:
    """Create a standard finance agent contract."""
    return AgentContract(
        agent_id=agent_id,
        name="Finance Agent",
        description="Can analyze financials but requires approval for transactions",
        capabilities={"financial_analysis", "forecasting", "budgeting"},
        allowed_categories={ActionCategory.READ, ActionCategory.FINANCIAL},
        actions_requiring_approval={"create_invoice", "process_payment", "transfer_funds"},
        max_cost_per_request=0.5,
        max_cost_per_hour=25.0,
    )


# Global enforcer instance
_contract_enforcer: Optional[ContractEnforcer] = None


def get_contract_enforcer() -> ContractEnforcer:
    """Get or create the global contract enforcer."""
    global _contract_enforcer
    if _contract_enforcer is None:
        _contract_enforcer = ContractEnforcer()
    return _contract_enforcer
