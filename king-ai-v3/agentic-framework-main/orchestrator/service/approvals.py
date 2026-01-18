"""
Human Approval Workflows for Orchestrator.

Module: orchestrator/service/approvals.py

Handles human-in-the-loop approval workflows for sensitive operations.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from enum import Enum
from uuid import uuid4
from pydantic import BaseModel, Field
import asyncio
import logging

logger = logging.getLogger(__name__)


class ApprovalStatus(str, Enum):
    """Approval request status."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class ApprovalPriority(str, Enum):
    """Approval request priority."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ApprovalRequest(BaseModel):
    """Model for approval request."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    workflow_id: str
    step_id: str
    operation: str
    description: str
    priority: ApprovalPriority = ApprovalPriority.MEDIUM
    requested_by: str
    requested_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    status: ApprovalStatus = ApprovalStatus.PENDING
    approver_id: Optional[str] = None
    approved_at: Optional[datetime] = None
    rejection_reason: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # Additional context for approval
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context for approver (tool names, parameters, risks)",
    )


class ApprovalResponse(BaseModel):
    """Model for approval response."""

    approval_id: str
    status: ApprovalStatus
    approver_id: Optional[str] = None
    approved_at: Optional[datetime] = None
    rejection_reason: Optional[str] = None


class ApprovalManager:
    """
    Manages human approval workflows.

    Features:
    - Create approval requests
    - Query pending approvals
    - Approve/reject operations
    - Automatic expiry handling
    - Priority-based queuing
    """

    def __init__(self) -> None:
        """Initialize approval manager."""
        self.pending_approvals: Dict[str, ApprovalRequest] = {}
        self.approval_history: List[ApprovalRequest] = []
        self._cleanup_task: Optional[asyncio.Task[None]] = None

    async def start(self) -> None:
        """Start background cleanup task."""
        self._cleanup_task = asyncio.create_task(self._cleanup_expired_approvals())
        logger.info("Approval manager started")

    async def stop(self) -> None:
        """Stop background tasks."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        logger.info("Approval manager stopped")

    async def create_approval_request(
        self,
        workflow_id: str,
        step_id: str,
        operation: str,
        description: str,
        requested_by: str,
        priority: ApprovalPriority = ApprovalPriority.MEDIUM,
        context: Optional[Dict[str, Any]] = None,
        expires_in_seconds: int = 3600,  # 1 hour default
    ) -> ApprovalRequest:
        """
        Create a new approval request.

        Args:
            workflow_id: ID of workflow requesting approval
            step_id: ID of step requiring approval
            operation: Operation being requested
            description: Human-readable description
            requested_by: ID of entity requesting approval
            priority: Priority level
            context: Additional context for approver
            expires_in_seconds: Expiration time in seconds

        Returns:
            Created approval request
        """
        request = ApprovalRequest(
            workflow_id=workflow_id,
            step_id=step_id,
            operation=operation,
            description=description,
            requested_by=requested_by,
            priority=priority,
            expires_at=datetime.utcnow() + timedelta(seconds=expires_in_seconds),
            context=context or {},
        )

        self.pending_approvals[request.id] = request

        logger.info(
            f"Created approval request {request.id} for workflow {workflow_id}, "
            f"operation: {operation}, priority: {priority}"
        )

        return request

    async def get_approval_request(self, approval_id: str) -> Optional[ApprovalRequest]:
        """
        Get approval request by ID.

        Args:
            approval_id: Approval request ID

        Returns:
            Approval request or None if not found
        """
        # Check pending
        if approval_id in self.pending_approvals:
            return self.pending_approvals[approval_id]

        # Check history
        for request in self.approval_history:
            if request.id == approval_id:
                return request

        return None

    async def list_pending_approvals(
        self,
        workflow_id: Optional[str] = None,
        priority: Optional[ApprovalPriority] = None,
    ) -> List[ApprovalRequest]:
        """
        List pending approval requests.

        Args:
            workflow_id: Filter by workflow ID
            priority: Filter by priority

        Returns:
            List of pending approval requests
        """
        requests = list(self.pending_approvals.values())

        # Apply filters
        if workflow_id:
            requests = [r for r in requests if r.workflow_id == workflow_id]

        if priority:
            requests = [r for r in requests if r.priority == priority]

        # Sort by priority (critical first) then by requested time
        priority_order = {
            ApprovalPriority.CRITICAL: 0,
            ApprovalPriority.HIGH: 1,
            ApprovalPriority.MEDIUM: 2,
            ApprovalPriority.LOW: 3,
        }

        requests.sort(key=lambda r: (priority_order[r.priority], r.requested_at))

        return requests

    async def approve_request(
        self, approval_id: str, approver_id: str
    ) -> ApprovalResponse:
        """
        Approve a request.

        Args:
            approval_id: Approval request ID
            approver_id: ID of approver

        Returns:
            Approval response

        Raises:
            ValueError: If approval not found or already processed
        """
        request = self.pending_approvals.get(approval_id)

        if not request:
            raise ValueError(f"Approval request {approval_id} not found")

        if request.status != ApprovalStatus.PENDING:
            raise ValueError(f"Approval request {approval_id} already {request.status}")

        # Check expiry
        if request.expires_at and datetime.utcnow() > request.expires_at:
            request.status = ApprovalStatus.EXPIRED
            self._move_to_history(approval_id)
            raise ValueError(f"Approval request {approval_id} has expired")

        # Approve
        request.status = ApprovalStatus.APPROVED
        request.approver_id = approver_id
        request.approved_at = datetime.utcnow()

        self._move_to_history(approval_id)

        logger.info(
            f"Approval request {approval_id} approved by {approver_id} "
            f"for workflow {request.workflow_id}"
        )

        return ApprovalResponse(
            approval_id=request.id,
            status=request.status,
            approver_id=request.approver_id,
            approved_at=request.approved_at,
        )

    async def reject_request(
        self, approval_id: str, approver_id: str, reason: str
    ) -> ApprovalResponse:
        """
        Reject a request.

        Args:
            approval_id: Approval request ID
            approver_id: ID of approver
            reason: Rejection reason

        Returns:
            Approval response

        Raises:
            ValueError: If approval not found or already processed
        """
        request = self.pending_approvals.get(approval_id)

        if not request:
            raise ValueError(f"Approval request {approval_id} not found")

        if request.status != ApprovalStatus.PENDING:
            raise ValueError(f"Approval request {approval_id} already {request.status}")

        # Reject
        request.status = ApprovalStatus.REJECTED
        request.approver_id = approver_id
        request.approved_at = datetime.utcnow()
        request.rejection_reason = reason

        self._move_to_history(approval_id)

        logger.info(
            f"Approval request {approval_id} rejected by {approver_id} "
            f"for workflow {request.workflow_id}: {reason}"
        )

        return ApprovalResponse(
            approval_id=request.id,
            status=request.status,
            approver_id=request.approver_id,
            approved_at=request.approved_at,
            rejection_reason=request.rejection_reason,
        )

    async def cancel_request(self, approval_id: str) -> ApprovalResponse:
        """
        Cancel a pending request.

        Args:
            approval_id: Approval request ID

        Returns:
            Approval response

        Raises:
            ValueError: If approval not found or already processed
        """
        request = self.pending_approvals.get(approval_id)

        if not request:
            raise ValueError(f"Approval request {approval_id} not found")

        if request.status != ApprovalStatus.PENDING:
            raise ValueError(f"Approval request {approval_id} already {request.status}")

        request.status = ApprovalStatus.CANCELLED
        self._move_to_history(approval_id)

        logger.info(f"Approval request {approval_id} cancelled")

        return ApprovalResponse(
            approval_id=request.id,
            status=request.status,
        )

    async def wait_for_approval(
        self, approval_id: str, timeout_seconds: int = 3600
    ) -> ApprovalResponse:
        """
        Wait for approval decision (blocking).

        Args:
            approval_id: Approval request ID
            timeout_seconds: Maximum wait time

        Returns:
            Approval response

        Raises:
            TimeoutError: If approval not decided within timeout
            ValueError: If approval not found
        """
        start_time = datetime.utcnow()
        timeout = timedelta(seconds=timeout_seconds)

        while True:
            request = await self.get_approval_request(approval_id)

            if not request:
                raise ValueError(f"Approval request {approval_id} not found")

            if request.status in [
                ApprovalStatus.APPROVED,
                ApprovalStatus.REJECTED,
                ApprovalStatus.CANCELLED,
                ApprovalStatus.EXPIRED,
            ]:
                return ApprovalResponse(
                    approval_id=request.id,
                    status=request.status,
                    approver_id=request.approver_id,
                    approved_at=request.approved_at,
                    rejection_reason=request.rejection_reason,
                )

            # Check timeout
            if datetime.utcnow() - start_time > timeout:
                # Mark as expired
                if request.status == ApprovalStatus.PENDING:
                    request.status = ApprovalStatus.EXPIRED
                    self._move_to_history(approval_id)

                raise TimeoutError(
                    f"Approval request {approval_id} timed out after {timeout_seconds}s"
                )

            # Wait before polling again
            await asyncio.sleep(1.0)

    def _move_to_history(self, approval_id: str) -> None:
        """Move approval from pending to history."""
        if approval_id in self.pending_approvals:
            request = self.pending_approvals.pop(approval_id)
            self.approval_history.append(request)

            # Limit history size
            if len(self.approval_history) > 10000:
                self.approval_history = self.approval_history[-5000:]

    async def _cleanup_expired_approvals(self) -> None:
        """Background task to clean up expired approvals."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute

                now = datetime.utcnow()
                expired_ids = []

                for approval_id, request in self.pending_approvals.items():
                    if request.expires_at and now > request.expires_at:
                        request.status = ApprovalStatus.EXPIRED
                        expired_ids.append(approval_id)

                for approval_id in expired_ids:
                    self._move_to_history(approval_id)
                    logger.info(f"Approval request {approval_id} expired")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in approval cleanup task: {str(e)}")


# Global approval manager instance
approval_manager = ApprovalManager()
