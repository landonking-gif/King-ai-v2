"""
Approval Manager - Handle approval workflows.
"""
import uuid
from datetime import datetime, timedelta
from typing import Optional, Callable
from src.approvals.models import (
    ApprovalRequest, ApprovalStatus, ApprovalType, RiskLevel,
    RiskFactor, ApprovalDecision, ApprovalPolicy
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


# Default policies
DEFAULT_POLICIES = [
    ApprovalPolicy(
        id="financial_high",
        name="High-Value Financial",
        action_types=[ApprovalType.FINANCIAL],
        min_risk_level=RiskLevel.MEDIUM,
        auto_approve_below=100.0,
        expiry_hours=24,
    ),
    ApprovalPolicy(
        id="legal_all",
        name="Legal Actions",
        action_types=[ApprovalType.LEGAL],
        min_risk_level=RiskLevel.LOW,
        require_two_approvers=True,
        expiry_hours=48,
    ),
    ApprovalPolicy(
        id="external_comms",
        name="External Communications",
        action_types=[ApprovalType.EXTERNAL],
        min_risk_level=RiskLevel.MEDIUM,
        expiry_hours=12,
    ),
]


class ApprovalManager:
    """Manage approval requests and workflows."""

    def __init__(self):
        self._requests: dict[str, ApprovalRequest] = {}
        self._decisions: list[ApprovalDecision] = []
        self._policies: dict[str, ApprovalPolicy] = {
            p.id: p for p in DEFAULT_POLICIES
        }
        self._hooks: dict[str, list[Callable]] = {
            "request_created": [],
            "request_approved": [],
            "request_rejected": [],
            "request_expired": [],
        }

    def register_hook(self, event: str, callback: Callable):
        """Register a callback for approval events."""
        if event in self._hooks:
            self._hooks[event].append(callback)

    async def create_request(
        self,
        business_id: str,
        action_type: ApprovalType,
        title: str,
        description: str,
        payload: dict,
        risk_level: RiskLevel = RiskLevel.MEDIUM,
        risk_factors: list[RiskFactor] = None,
        source_plan_id: str = None,
        source_task_id: str = None,
    ) -> ApprovalRequest:
        """Create a new approval request."""
        # Find applicable policy
        policy = self._find_policy(action_type, risk_level)
        
        # Check auto-approve
        if policy and policy.auto_approve_below:
            amount = payload.get("amount", 0)
            if amount < policy.auto_approve_below:
                logger.info(f"Auto-approving {title} (amount {amount} < {policy.auto_approve_below})")
                # Create auto-approved request
                request = ApprovalRequest(
                    id=str(uuid.uuid4()),
                    business_id=business_id,
                    action_type=action_type,
                    title=title,
                    description=description,
                    risk_level=risk_level,
                    risk_factors=risk_factors or [],
                    payload=payload,
                    status=ApprovalStatus.APPROVED,
                    reviewed_at=datetime.utcnow(),
                    reviewed_by="system_auto",
                    review_notes="Auto-approved per policy",
                    source_plan_id=source_plan_id,
                    source_task_id=source_task_id,
                )
                self._requests[request.id] = request
                return request

        # Calculate expiry
        expiry_hours = policy.expiry_hours if policy else 24
        expires_at = datetime.utcnow() + timedelta(hours=expiry_hours)

        # Determine required approvers based on policy and risk level
        required_approvers = 1
        if policy and getattr(policy, 'require_two_approvers', False):
            required_approvers = 2
        
        # High-risk and critical always require 2 approvers
        if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            required_approvers = max(required_approvers, 2)

        request = ApprovalRequest(
            id=str(uuid.uuid4()),
            business_id=business_id,
            action_type=action_type,
            title=title,
            description=description,
            risk_level=risk_level,
            risk_factors=risk_factors or [],
            payload=payload,
            expires_at=expires_at,
            required_approvers=required_approvers,
            source_plan_id=source_plan_id,
            source_task_id=source_task_id,
        )

        self._requests[request.id] = request
        logger.info(f"Created approval request: {request.id} - {title}")

        # Trigger hooks
        for hook in self._hooks["request_created"]:
            try:
                await hook(request)
            except Exception as e:
                logger.error(f"Hook error: {e}")

        return request

    async def approve(
        self,
        request_id: str,
        user_id: str,
        notes: str = None,
        modifications: dict = None,
    ) -> Optional[ApprovalRequest]:
        """
        Add an approval vote to a request.
        
        For multi-approver requests, this adds a vote. The request
        is only fully approved when enough votes are received.
        """
        request = self._requests.get(request_id)
        if not request:
            return None
        
        # Check valid status for voting
        if request.status not in [ApprovalStatus.PENDING, ApprovalStatus.PARTIAL]:
            return None

        if request.is_expired:
            request.status = ApprovalStatus.EXPIRED
            return None
        
        # Check if user already voted
        if request.has_voted(user_id):
            logger.warning(f"User {user_id} already voted on request {request_id}")
            return request

        # Handle modifications (single approver case)
        if modifications:
            request.status = ApprovalStatus.MODIFIED
            request.modified_payload = modifications
            request.reviewed_at = datetime.utcnow()
            request.reviewed_by = user_id
            request.review_notes = notes
        else:
            # Add vote using multi-approver logic
            fully_approved = request.add_vote(user_id, "approve", notes)
            
            if fully_approved:
                logger.info(f"Request {request_id} fully approved")
            elif request.status == ApprovalStatus.PARTIAL:
                logger.info(
                    f"Request {request_id} partially approved: "
                    f"{request.approval_count}/{request.required_approvers}"
                )

        # Record decision
        self._decisions.append(ApprovalDecision(
            request_id=request_id,
            decision=request.status,
            decided_by=user_id,
            decided_at=datetime.utcnow(),
            notes=notes,
            modifications=modifications,
        ))

        if request.status == ApprovalStatus.APPROVED:
            logger.info(f"Approved request: {request_id} by {user_id}")
            for hook in self._hooks["request_approved"]:
                try:
                    await hook(request)
                except Exception as e:
                    logger.error(f"Hook error: {e}")

        return request

    async def reject(
        self,
        request_id: str,
        user_id: str,
        notes: str = None,
    ) -> Optional[ApprovalRequest]:
        """
        Reject a request.
        
        Any rejection immediately rejects the entire request,
        regardless of how many approvals were received.
        """
        request = self._requests.get(request_id)
        if not request:
            return None
        
        if request.status not in [ApprovalStatus.PENDING, ApprovalStatus.PARTIAL]:
            return None

        # Use multi-approver vote logic
        request.add_vote(user_id, "reject", notes)

        self._decisions.append(ApprovalDecision(
            request_id=request_id,
            decision=ApprovalStatus.REJECTED,
            decided_by=user_id,
            decided_at=request.reviewed_at,
            notes=notes,
        ))

        logger.info(f"Rejected request: {request_id} by {user_id}")

        for hook in self._hooks["request_rejected"]:
            try:
                await hook(request)
            except Exception as e:
                logger.error(f"Hook error: {e}")

        return request

    async def get_pending(
        self,
        business_id: str = None,
        action_type: ApprovalType = None,
    ) -> list[ApprovalRequest]:
        """Get pending approval requests."""
        pending = []
        for req in self._requests.values():
            if req.status != ApprovalStatus.PENDING:
                continue
            if req.is_expired:
                req.status = ApprovalStatus.EXPIRED
                continue
            if business_id and req.business_id != business_id:
                continue
            if action_type and req.action_type != action_type:
                continue
            pending.append(req)

        # Sort by risk level and creation time
        return sorted(
            pending,
            key=lambda r: (r.risk_level.value, r.created_at),
            reverse=True,
        )

    async def get_request(self, request_id: str) -> Optional[ApprovalRequest]:
        """Get a specific request."""
        return self._requests.get(request_id)

    async def get_history(
        self,
        business_id: str = None,
        limit: int = 50,
    ) -> list[ApprovalRequest]:
        """Get approval history."""
        requests = list(self._requests.values())
        
        if business_id:
            requests = [r for r in requests if r.business_id == business_id]
        
        # Sort by reviewed_at or created_at
        requests.sort(
            key=lambda r: r.reviewed_at or r.created_at,
            reverse=True,
        )
        
        return requests[:limit]

    async def get_stats(self, business_id: str = None) -> dict:
        """Get approval statistics."""
        requests = list(self._requests.values())
        if business_id:
            requests = [r for r in requests if r.business_id == business_id]

        pending = [r for r in requests if r.status == ApprovalStatus.PENDING]
        approved = [r for r in requests if r.status == ApprovalStatus.APPROVED]
        rejected = [r for r in requests if r.status == ApprovalStatus.REJECTED]

        avg_wait = 0
        if approved:
            waits = [
                (r.reviewed_at - r.created_at).total_seconds() / 3600
                for r in approved if r.reviewed_at
            ]
            avg_wait = sum(waits) / len(waits) if waits else 0

        return {
            "total": len(requests),
            "pending": len(pending),
            "approved": len(approved),
            "rejected": len(rejected),
            "approval_rate": len(approved) / len(requests) * 100 if requests else 0,
            "avg_wait_hours": round(avg_wait, 2),
            "high_risk_pending": len([r for r in pending if r.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]]),
        }

    def _find_policy(
        self, action_type: ApprovalType, risk_level: RiskLevel
    ) -> Optional[ApprovalPolicy]:
        """Find applicable policy."""
        for policy in self._policies.values():
            if action_type in policy.action_types:
                return policy
        return None

    async def expire_old_requests(self) -> int:
        """
        Expire requests that have passed their expiry time.
        
        Returns:
            Number of requests that were expired.
        """
        count = 0
        for request in self._requests.values():
            if request.status in (ApprovalStatus.PENDING, ApprovalStatus.PARTIAL) and request.is_expired:
                request.status = ApprovalStatus.EXPIRED
                count += 1
                logger.info(f"Expired approval request: {request.id} - {request.title}")
                
                # Trigger expired hooks
                for hook in self._hooks.get("request_expired", []):
                    try:
                        await hook(request)
                    except Exception as e:
                        logger.error(f"Error in expired hook: {e}")
        
        return count
