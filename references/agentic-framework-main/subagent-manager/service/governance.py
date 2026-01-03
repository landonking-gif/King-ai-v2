"""
Governance Gate for Enterprise Agent Policy Enforcement.

Module: subagent-manager/service/governance.py

Provides policy evaluation, approval workflows, and compliance checking
for enterprise agent operations.
"""

import logging
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from uuid import uuid4

from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class PolicySeverity(str, Enum):
    """Severity level of policy violations."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class PolicyCategory(str, Enum):
    """Categories of governance policies."""

    DATA_ACCESS = "data_access"
    FILE_OPERATIONS = "file_operations"
    NETWORK_ACCESS = "network_access"
    CODE_EXECUTION = "code_execution"
    EXTERNAL_SERVICES = "external_services"
    SENSITIVE_DATA = "sensitive_data"
    COMPLIANCE = "compliance"
    RESOURCE_LIMITS = "resource_limits"


class PolicyViolation(BaseModel):
    """A violation of a governance policy."""

    violation_id: str = Field(default_factory=lambda: str(uuid4())[:12])
    policy_name: str = Field(..., description="Name of violated policy")
    category: PolicyCategory = Field(...)
    severity: PolicySeverity = Field(default=PolicySeverity.ERROR)
    description: str = Field(..., description="Human-readable description")
    affected_step: Optional[str] = Field(None, description="ID of affected plan step")
    affected_tool: Optional[str] = Field(None, description="Tool that triggered violation")
    remediation: Optional[str] = Field(None, description="Suggested remediation")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ApprovalResult(BaseModel):
    """Result of governance evaluation."""

    approval_id: str = Field(default_factory=lambda: str(uuid4())[:12])
    approved: bool = Field(..., description="Whether execution is approved")
    approver: str = Field(default="system", description="Who approved (system/human/policy)")
    requires_human: bool = Field(
        default=False, description="Whether human approval is required"
    )
    violations: List[PolicyViolation] = Field(default_factory=list)
    warnings: List[PolicyViolation] = Field(default_factory=list)
    conditions: List[str] = Field(
        default_factory=list, description="Conditions for approval"
    )
    expires_at: Optional[datetime] = Field(None, description="When approval expires")
    reason: str = Field(default="", description="Reason for decision")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class GovernancePolicy(BaseModel):
    """Definition of a governance policy."""

    name: str = Field(..., description="Unique policy name")
    category: PolicyCategory = Field(...)
    description: str = Field(...)
    enabled: bool = Field(default=True)
    severity: PolicySeverity = Field(default=PolicySeverity.ERROR)
    # Tools that trigger this policy check
    trigger_tools: Set[str] = Field(default_factory=set)
    # Actions that trigger this policy check
    trigger_actions: Set[str] = Field(default_factory=set)
    # Risk levels that trigger this policy
    trigger_risk_levels: Set[str] = Field(default_factory=set)
    # Whether violation blocks execution
    blocking: bool = Field(default=True)
    # Whether human approval can override
    human_overridable: bool = Field(default=True)
    # Custom validation function name
    validator: Optional[str] = Field(None)


class GovernanceGate:
    """
    Governance gate for enterprise policy enforcement.

    Evaluates execution plans against configured policies and determines
    whether approval can be granted automatically or requires human intervention.
    """

    # Default high-risk tools requiring approval
    DEFAULT_HIGH_RISK_TOOLS: Set[str] = {
        "bash_exec",
        "file_write",
        "file_delete",
        "deploy",
        "database_write",
        "send_email",
        "api_call_external",
    }

    # Default sensitive patterns to check
    DEFAULT_SENSITIVE_PATTERNS: List[str] = [
        r"password",
        r"secret",
        r"api[_-]?key",
        r"token",
        r"credential",
        r"private[_-]?key",
        r"\.env",
        r"\.pem",
        r"\.key",
    ]

    def __init__(
        self,
        policies: Optional[List[GovernancePolicy]] = None,
        high_risk_tools: Optional[Set[str]] = None,
        sensitive_patterns: Optional[List[str]] = None,
        custom_validators: Optional[Dict[str, Callable]] = None,
    ):
        """
        Initialize governance gate.

        Args:
            policies: Custom governance policies
            high_risk_tools: Tools requiring approval
            sensitive_patterns: Patterns indicating sensitive data
            custom_validators: Custom policy validators
        """
        self.policies = policies or self._get_default_policies()
        self.high_risk_tools = high_risk_tools or self.DEFAULT_HIGH_RISK_TOOLS
        self.sensitive_patterns = sensitive_patterns or self.DEFAULT_SENSITIVE_PATTERNS
        self.custom_validators = custom_validators or {}

        # Index policies for faster lookup
        self._policy_by_tool: Dict[str, List[GovernancePolicy]] = {}
        self._policy_by_action: Dict[str, List[GovernancePolicy]] = {}
        self._index_policies()

    def _get_default_policies(self) -> List[GovernancePolicy]:
        """Get default governance policies."""
        return [
            GovernancePolicy(
                name="high_risk_tool_execution",
                category=PolicyCategory.CODE_EXECUTION,
                description="High-risk tool execution requires approval",
                trigger_tools=self.DEFAULT_HIGH_RISK_TOOLS,
                severity=PolicySeverity.WARNING,
                blocking=False,
                human_overridable=True,
            ),
            GovernancePolicy(
                name="file_system_write",
                category=PolicyCategory.FILE_OPERATIONS,
                description="File write operations require review",
                trigger_tools={"file_write", "file_edit", "file_delete"},
                severity=PolicySeverity.WARNING,
                blocking=False,
                human_overridable=True,
            ),
            GovernancePolicy(
                name="sensitive_data_access",
                category=PolicyCategory.SENSITIVE_DATA,
                description="Accessing sensitive data patterns",
                trigger_actions={"read_credentials", "access_secrets"},
                severity=PolicySeverity.CRITICAL,
                blocking=True,
                human_overridable=True,
            ),
            GovernancePolicy(
                name="external_network_call",
                category=PolicyCategory.NETWORK_ACCESS,
                description="External network calls require approval",
                trigger_tools={"web_fetch", "api_call"},
                severity=PolicySeverity.INFO,
                blocking=False,
                human_overridable=True,
            ),
            GovernancePolicy(
                name="critical_risk_level",
                category=PolicyCategory.COMPLIANCE,
                description="Critical risk level requires human approval",
                trigger_risk_levels={"critical"},
                severity=PolicySeverity.CRITICAL,
                blocking=True,
                human_overridable=True,
            ),
        ]

    def _index_policies(self) -> None:
        """Index policies by trigger for fast lookup."""
        for policy in self.policies:
            for tool in policy.trigger_tools:
                if tool not in self._policy_by_tool:
                    self._policy_by_tool[tool] = []
                self._policy_by_tool[tool].append(policy)

            for action in policy.trigger_actions:
                if action not in self._policy_by_action:
                    self._policy_by_action[action] = []
                self._policy_by_action[action].append(policy)

    async def evaluate(
        self,
        plan: Any,  # EnterpriseExecutionPlan
        context: Dict[str, Any],
        config: Any,  # EnterpriseAgentConfig
    ) -> ApprovalResult:
        """
        Evaluate a plan against governance policies.

        Args:
            plan: The execution plan to evaluate
            context: Execution context (user_id, iteration, etc.)
            config: Agent configuration

        Returns:
            ApprovalResult with approval status and any violations
        """
        logger.info(f"Evaluating plan {plan.plan_id} against governance policies")

        violations: List[PolicyViolation] = []
        warnings: List[PolicyViolation] = []
        requires_human = False

        # Collect all tools and actions from plan
        tools_used: Set[str] = set()
        actions_used: Set[str] = set()

        for step in plan.steps:
            if step.tool:
                tools_used.add(step.tool)
            actions_used.add(step.action.lower())

        # Check tool-triggered policies
        for tool in tools_used:
            policies = self._policy_by_tool.get(tool, [])
            for policy in policies:
                if not policy.enabled:
                    continue

                violation = self._evaluate_policy(policy, plan, tool, context)
                if violation:
                    if policy.blocking:
                        violations.append(violation)
                        if not policy.human_overridable:
                            requires_human = True
                    else:
                        warnings.append(violation)

        # Check risk level policies
        for policy in self.policies:
            if plan.risk_level in policy.trigger_risk_levels:
                if policy.severity == PolicySeverity.CRITICAL:
                    requires_human = True
                    violations.append(
                        PolicyViolation(
                            policy_name=policy.name,
                            category=policy.category,
                            severity=policy.severity,
                            description=f"Plan has {plan.risk_level} risk level",
                            remediation="Requires human approval for execution",
                        )
                    )

        # Check for sensitive data patterns
        sensitive_violation = self._check_sensitive_patterns(plan)
        if sensitive_violation:
            violations.append(sensitive_violation)
            requires_human = True

        # Determine approval
        approved = self._determine_approval(
            violations=violations,
            warnings=warnings,
            plan=plan,
            config=config,
            context=context,
        )

        # Build result
        approver = "system"
        if requires_human:
            approver = "pending_human"
            approved = False
        elif approved and not violations:
            approver = "auto_policy"

        reason = self._build_approval_reason(approved, violations, warnings)

        result = ApprovalResult(
            approved=approved,
            approver=approver,
            requires_human=requires_human,
            violations=violations,
            warnings=warnings,
            reason=reason,
        )

        logger.info(
            f"Governance evaluation: approved={approved}, "
            f"violations={len(violations)}, warnings={len(warnings)}"
        )

        return result

    def _evaluate_policy(
        self,
        policy: GovernancePolicy,
        plan: Any,
        trigger: str,
        context: Dict[str, Any],
    ) -> Optional[PolicyViolation]:
        """Evaluate a single policy."""
        # Check for custom validator
        if policy.validator and policy.validator in self.custom_validators:
            validator = self.custom_validators[policy.validator]
            return validator(policy, plan, trigger, context)

        # Default evaluation - create violation for trigger match
        return PolicyViolation(
            policy_name=policy.name,
            category=policy.category,
            severity=policy.severity,
            description=f"Policy triggered by: {trigger}",
            affected_tool=trigger if trigger in self.high_risk_tools else None,
            remediation=f"Review {policy.category.value} usage",
        )

    def _check_sensitive_patterns(self, plan: Any) -> Optional[PolicyViolation]:
        """Check plan for sensitive data patterns."""
        import re

        # Check plan content for sensitive patterns
        plan_str = str(plan.model_dump())

        for pattern in self.sensitive_patterns:
            if re.search(pattern, plan_str, re.IGNORECASE):
                return PolicyViolation(
                    policy_name="sensitive_data_detected",
                    category=PolicyCategory.SENSITIVE_DATA,
                    severity=PolicySeverity.CRITICAL,
                    description=f"Sensitive pattern detected: {pattern}",
                    remediation="Review plan for sensitive data handling",
                )

        return None

    def _determine_approval(
        self,
        violations: List[PolicyViolation],
        warnings: List[PolicyViolation],
        plan: Any,
        config: Any,
        context: Dict[str, Any],
    ) -> bool:
        """Determine if plan should be approved."""
        # No critical violations = approved
        critical_violations = [
            v for v in violations if v.severity == PolicySeverity.CRITICAL
        ]

        if critical_violations:
            return False

        # Check config for auto-approval settings
        if hasattr(config, "auto_approve_low_risk") and config.auto_approve_low_risk:
            if plan.risk_level == "low":
                return True

        # Check max risk level for auto-approval
        if hasattr(config, "max_risk_level_auto_approve"):
            risk_order = ["low", "medium", "high", "critical"]
            plan_risk_idx = risk_order.index(plan.risk_level)
            max_auto_idx = risk_order.index(config.max_risk_level_auto_approve)

            if plan_risk_idx <= max_auto_idx:
                return True

        # Default: approve if only warnings
        return len(violations) == 0

    def _build_approval_reason(
        self,
        approved: bool,
        violations: List[PolicyViolation],
        warnings: List[PolicyViolation],
    ) -> str:
        """Build human-readable approval reason."""
        if approved and not violations and not warnings:
            return "Plan approved - no policy violations detected"

        if approved and warnings:
            return f"Plan approved with {len(warnings)} warning(s)"

        if not approved and violations:
            violation_summary = ", ".join(v.policy_name for v in violations[:3])
            return f"Plan denied - {len(violations)} violation(s): {violation_summary}"

        return "Plan requires review"

    def add_policy(self, policy: GovernancePolicy) -> None:
        """Add a custom policy."""
        self.policies.append(policy)
        self._index_policies()

    def remove_policy(self, policy_name: str) -> bool:
        """Remove a policy by name."""
        for i, policy in enumerate(self.policies):
            if policy.name == policy_name:
                self.policies.pop(i)
                self._index_policies()
                return True
        return False

    def register_validator(
        self, name: str, validator: Callable
    ) -> None:
        """Register a custom policy validator."""
        self.custom_validators[name] = validator


class HumanApprovalQueue:
    """
    Queue for managing human approval requests.

    In production, this would integrate with an approval workflow system.
    """

    def __init__(self):
        self._pending: Dict[str, Dict[str, Any]] = {}
        self._completed: Dict[str, ApprovalResult] = {}

    async def request_approval(
        self,
        plan: Any,
        violations: List[PolicyViolation],
        context: Dict[str, Any],
        timeout_seconds: int = 3600,
    ) -> str:
        """
        Request human approval for a plan.

        Returns request_id for tracking.
        """
        request_id = str(uuid4())[:12]

        self._pending[request_id] = {
            "plan": plan,
            "violations": violations,
            "context": context,
            "requested_at": datetime.utcnow(),
            "timeout_seconds": timeout_seconds,
            "status": "pending",
        }

        logger.info(f"Human approval requested: {request_id}")
        return request_id

    async def check_status(self, request_id: str) -> Optional[ApprovalResult]:
        """Check status of an approval request."""
        if request_id in self._completed:
            return self._completed[request_id]

        if request_id in self._pending:
            # Check timeout
            request = self._pending[request_id]
            elapsed = (datetime.utcnow() - request["requested_at"]).total_seconds()

            if elapsed > request["timeout_seconds"]:
                result = ApprovalResult(
                    approved=False,
                    approver="timeout",
                    reason="Approval request timed out",
                )
                self._completed[request_id] = result
                del self._pending[request_id]
                return result

            return None  # Still pending

        return None  # Not found

    async def approve(
        self,
        request_id: str,
        approver_id: str,
        conditions: Optional[List[str]] = None,
    ) -> ApprovalResult:
        """Approve a pending request."""
        if request_id not in self._pending:
            raise ValueError(f"Request {request_id} not found")

        result = ApprovalResult(
            approved=True,
            approver=approver_id,
            conditions=conditions or [],
            reason="Human approval granted",
        )

        self._completed[request_id] = result
        del self._pending[request_id]

        logger.info(f"Approval granted by {approver_id}: {request_id}")
        return result

    async def deny(
        self,
        request_id: str,
        approver_id: str,
        reason: str,
    ) -> ApprovalResult:
        """Deny a pending request."""
        if request_id not in self._pending:
            raise ValueError(f"Request {request_id} not found")

        result = ApprovalResult(
            approved=False,
            approver=approver_id,
            reason=f"Human denied: {reason}",
        )

        self._completed[request_id] = result
        del self._pending[request_id]

        logger.info(f"Approval denied by {approver_id}: {request_id}")
        return result
