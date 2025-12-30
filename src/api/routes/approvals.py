"""
Approval API Routes.
"""
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from src.approvals.models import ApprovalType, RiskLevel
from src.approvals.manager import ApprovalManager
from src.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()

_manager: Optional[ApprovalManager] = None


def get_manager() -> ApprovalManager:
    global _manager
    if _manager is None:
        _manager = ApprovalManager()
    return _manager


class ApproveRequest(BaseModel):
    user_id: str
    notes: Optional[str] = None
    modifications: Optional[dict] = None


class RejectRequest(BaseModel):
    user_id: str
    notes: Optional[str] = None


@router.get("/pending")
async def get_pending(
    business_id: Optional[str] = None,
    action_type: Optional[str] = None,
    manager: ApprovalManager = Depends(get_manager),
):
    """Get pending approval requests."""
    atype = None
    if action_type:
        try:
            atype = ApprovalType(action_type)
        except ValueError:
            pass

    pending = await manager.get_pending(business_id, atype)
    
    return {
        "count": len(pending),
        "requests": [
            {
                "id": r.id,
                "business_id": r.business_id,
                "type": r.action_type.value,
                "title": r.title,
                "description": r.description,
                "risk_level": r.risk_level.value,
                "risk_factors": [
                    {"category": f.category, "description": f.description, "severity": f.severity.value}
                    for f in r.risk_factors
                ],
                "payload": r.payload,
                "created_at": r.created_at.isoformat(),
                "expires_at": r.expires_at.isoformat() if r.expires_at else None,
                "waiting_hours": round(r.waiting_hours, 2),
            }
            for r in pending
        ],
    }


@router.get("/stats")
async def get_stats(
    business_id: Optional[str] = None,
    manager: ApprovalManager = Depends(get_manager),
):
    """Get approval statistics."""
    return await manager.get_stats(business_id)


@router.get("/{request_id}")
async def get_request(
    request_id: str,
    manager: ApprovalManager = Depends(get_manager),
):
    """Get a specific approval request."""
    request = await manager.get_request(request_id)
    if not request:
        raise HTTPException(404, "Request not found")
    
    return {
        "id": request.id,
        "business_id": request.business_id,
        "type": request.action_type.value,
        "title": request.title,
        "description": request.description,
        "risk_level": request.risk_level.value,
        "risk_factors": [
            {
                "category": f.category,
                "description": f.description,
                "severity": f.severity.value,
                "mitigation": f.mitigation,
            }
            for f in request.risk_factors
        ],
        "payload": request.payload,
        "status": request.status.value,
        "created_at": request.created_at.isoformat(),
        "expires_at": request.expires_at.isoformat() if request.expires_at else None,
        "reviewed_at": request.reviewed_at.isoformat() if request.reviewed_at else None,
        "reviewed_by": request.reviewed_by,
        "review_notes": request.review_notes,
        "modified_payload": request.modified_payload,
    }


@router.post("/{request_id}/approve")
async def approve_request(
    request_id: str,
    req: ApproveRequest,
    manager: ApprovalManager = Depends(get_manager),
):
    """Approve a request."""
    request = await manager.approve(
        request_id,
        req.user_id,
        req.notes,
        req.modifications,
    )
    
    if not request:
        raise HTTPException(400, "Cannot approve request")
    
    return {
        "status": request.status.value,
        "message": "Request approved",
    }


@router.post("/{request_id}/reject")
async def reject_request(
    request_id: str,
    req: RejectRequest,
    manager: ApprovalManager = Depends(get_manager),
):
    """Reject a request."""
    request = await manager.reject(request_id, req.user_id, req.notes)
    
    if not request:
        raise HTTPException(400, "Cannot reject request")
    
    return {
        "status": request.status.value,
        "message": "Request rejected",
    }


@router.get("/history/{business_id}")
async def get_history(
    business_id: str,
    limit: int = 50,
    manager: ApprovalManager = Depends(get_manager),
):
    """Get approval history for a business."""
    history = await manager.get_history(business_id, limit)
    
    return {
        "count": len(history),
        "requests": [
            {
                "id": r.id,
                "type": r.action_type.value,
                "title": r.title,
                "status": r.status.value,
                "risk_level": r.risk_level.value,
                "created_at": r.created_at.isoformat(),
                "reviewed_at": r.reviewed_at.isoformat() if r.reviewed_at else None,
                "reviewed_by": r.reviewed_by,
            }
            for r in history
        ],
    }
