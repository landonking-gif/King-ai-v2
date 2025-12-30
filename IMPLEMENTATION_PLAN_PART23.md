# Implementation Plan Part 23: Dashboard Approval Workflows

| Field | Value |
|-------|-------|
| Module | Human-in-the-Loop Approval System |
| Priority | High |
| Estimated Effort | 5-6 hours |
| Dependencies | Part 5 (ReAct Planner), Part 22 (Dashboard Components) |

---

## 1. Scope

This module implements the approval workflow system:

- **Approval Queue** - Pending actions requiring human review
- **Risk Assessment Display** - Show why approval is needed
- **Approval/Rejection Flow** - Accept, reject, or modify actions
- **Audit Trail** - Track all approval decisions
- **Notifications** - Alert users of pending approvals

---

## 2. Tasks

### Task 23.1: Approval Models (Backend)

**File: `src/approvals/models.py`**

```python
"""
Approval System Models.
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Any


class ApprovalStatus(Enum):
    """Status of an approval request."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    MODIFIED = "modified"


class RiskLevel(Enum):
    """Risk level requiring approval."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ApprovalType(Enum):
    """Types of actions requiring approval."""
    FINANCIAL = "financial"
    LEGAL = "legal"
    OPERATIONAL = "operational"
    STRATEGIC = "strategic"
    TECHNICAL = "technical"
    EXTERNAL = "external"


@dataclass
class RiskFactor:
    """A single risk factor."""
    category: str
    description: str
    severity: RiskLevel
    mitigation: Optional[str] = None


@dataclass
class ApprovalRequest:
    """A request for human approval."""
    id: str
    business_id: str
    action_type: ApprovalType
    title: str
    description: str
    risk_level: RiskLevel
    risk_factors: list[RiskFactor] = field(default_factory=list)
    payload: dict = field(default_factory=dict)  # Action details
    status: ApprovalStatus = ApprovalStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    reviewed_at: Optional[datetime] = None
    reviewed_by: Optional[str] = None
    review_notes: Optional[str] = None
    modified_payload: Optional[dict] = None
    source_plan_id: Optional[str] = None
    source_task_id: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        if self.expires_at and self.status == ApprovalStatus.PENDING:
            return datetime.utcnow() > self.expires_at
        return False

    @property
    def waiting_hours(self) -> float:
        if self.status == ApprovalStatus.PENDING:
            return (datetime.utcnow() - self.created_at).total_seconds() / 3600
        return 0


@dataclass
class ApprovalDecision:
    """Record of an approval decision."""
    request_id: str
    decision: ApprovalStatus
    decided_by: str
    decided_at: datetime
    notes: Optional[str] = None
    modifications: Optional[dict] = None


@dataclass
class ApprovalPolicy:
    """Policy defining when approval is required."""
    id: str
    name: str
    action_types: list[ApprovalType]
    min_risk_level: RiskLevel
    auto_approve_below: Optional[float] = None  # Auto-approve if $ below
    require_two_approvers: bool = False
    expiry_hours: int = 24
    notify_on_create: bool = True
    escalate_after_hours: int = 4
```

---

### Task 23.2: Approval Manager (Backend)

**File: `src/approvals/manager.py`**

```python
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
        """Approve a request."""
        request = self._requests.get(request_id)
        if not request or request.status != ApprovalStatus.PENDING:
            return None

        if request.is_expired:
            request.status = ApprovalStatus.EXPIRED
            return None

        if modifications:
            request.status = ApprovalStatus.MODIFIED
            request.modified_payload = modifications
        else:
            request.status = ApprovalStatus.APPROVED

        request.reviewed_at = datetime.utcnow()
        request.reviewed_by = user_id
        request.review_notes = notes

        # Record decision
        self._decisions.append(ApprovalDecision(
            request_id=request_id,
            decision=request.status,
            decided_by=user_id,
            decided_at=request.reviewed_at,
            notes=notes,
            modifications=modifications,
        ))

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
        """Reject a request."""
        request = self._requests.get(request_id)
        if not request or request.status != ApprovalStatus.PENDING:
            return None

        request.status = ApprovalStatus.REJECTED
        request.reviewed_at = datetime.utcnow()
        request.reviewed_by = user_id
        request.review_notes = notes

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
```

---

### Task 23.3: Approval API Routes

**File: `src/api/routes/approvals.py`**

```python
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
router = APIRouter(prefix="/approvals", tags=["approvals"])

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
```

---

### Task 23.4: React Approval Components

**File: `dashboard/src/components/Approvals/ApprovalQueue.jsx`**

```jsx
import { useState, useEffect } from 'react';
import { ApprovalCard } from './ApprovalCard';
import './Approvals.css';

export function ApprovalQueue({ businessId, onApprove, onReject }) {
  const [requests, setRequests] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState('all');
  
  useEffect(() => {
    fetchPending();
  }, [businessId]);
  
  const fetchPending = async () => {
    setLoading(true);
    try {
      const url = businessId
        ? `/api/approvals/pending?business_id=${businessId}`
        : '/api/approvals/pending';
      const res = await fetch(url);
      const data = await res.json();
      setRequests(data.requests || []);
    } catch (err) {
      console.error('Failed to fetch approvals:', err);
    }
    setLoading(false);
  };
  
  const filteredRequests = requests.filter(r => {
    if (filter === 'all') return true;
    return r.risk_level === filter;
  });
  
  const handleAction = async (id, action, data) => {
    if (action === 'approve') {
      await onApprove?.(id, data);
    } else {
      await onReject?.(id, data);
    }
    fetchPending();
  };
  
  if (loading) {
    return <div className="loading">Loading approvals...</div>;
  }
  
  return (
    <div className="approval-queue">
      <div className="queue-header">
        <h2>Pending Approvals</h2>
        <span className="count-badge">{requests.length}</span>
        
        <div className="filter-tabs">
          {['all', 'critical', 'high', 'medium', 'low'].map(level => (
            <button
              key={level}
              className={`filter-tab ${filter === level ? 'active' : ''}`}
              onClick={() => setFilter(level)}
            >
              {level}
            </button>
          ))}
        </div>
      </div>
      
      <div className="queue-list">
        {filteredRequests.length === 0 ? (
          <div className="empty-queue">
            <span className="empty-icon">✓</span>
            <p>No pending approvals</p>
          </div>
        ) : (
          filteredRequests.map(req => (
            <ApprovalCard
              key={req.id}
              request={req}
              onAction={handleAction}
            />
          ))
        )}
      </div>
    </div>
  );
}
```

**File: `dashboard/src/components/Approvals/ApprovalCard.jsx`**

```jsx
import { useState } from 'react';
import { RiskBadge } from './RiskBadge';
import './Approvals.css';

export function ApprovalCard({ request, onAction }) {
  const [expanded, setExpanded] = useState(false);
  const [notes, setNotes] = useState('');
  const [processing, setProcessing] = useState(false);
  
  const handleApprove = async () => {
    setProcessing(true);
    await onAction(request.id, 'approve', { notes });
    setProcessing(false);
  };
  
  const handleReject = async () => {
    setProcessing(true);
    await onAction(request.id, 'reject', { notes });
    setProcessing(false);
  };
  
  const typeIcons = {
    financial: '💰',
    legal: '⚖️',
    operational: '⚙️',
    strategic: '🎯',
    technical: '🔧',
    external: '🌐',
  };
  
  return (
    <div className={`approval-card risk-${request.risk_level}`}>
      <div className="card-main" onClick={() => setExpanded(!expanded)}>
        <div className="card-icon">{typeIcons[request.type] || '📋'}</div>
        
        <div className="card-content">
          <h3 className="card-title">{request.title}</h3>
          <p className="card-desc">{request.description}</p>
          
          <div className="card-meta">
            <span className="meta-type">{request.type}</span>
            <span className="meta-time">
              ⏱ {request.waiting_hours.toFixed(1)}h waiting
            </span>
          </div>
        </div>
        
        <div className="card-risk">
          <RiskBadge level={request.risk_level} />
        </div>
        
        <span className="expand-icon">{expanded ? '▲' : '▼'}</span>
      </div>
      
      {expanded && (
        <div className="card-details">
          {request.risk_factors?.length > 0 && (
            <div className="risk-factors">
              <h4>Risk Factors</h4>
              {request.risk_factors.map((factor, i) => (
                <div key={i} className={`risk-factor severity-${factor.severity}`}>
                  <span className="factor-category">{factor.category}</span>
                  <p>{factor.description}</p>
                </div>
              ))}
            </div>
          )}
          
          <div className="payload-preview">
            <h4>Action Details</h4>
            <pre>{JSON.stringify(request.payload, null, 2)}</pre>
          </div>
          
          <div className="action-form">
            <textarea
              placeholder="Add notes (optional)..."
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              rows={2}
            />
            
            <div className="action-buttons">
              <button
                className="btn btn-reject"
                onClick={handleReject}
                disabled={processing}
              >
                Reject
              </button>
              <button
                className="btn btn-approve"
                onClick={handleApprove}
                disabled={processing}
              >
                Approve
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
```

**File: `dashboard/src/components/Approvals/RiskBadge.jsx`**

```jsx
import './Approvals.css';

const riskConfig = {
  critical: { color: '#c0392b', icon: '🔴', label: 'Critical' },
  high: { color: '#e74c3c', icon: '🟠', label: 'High' },
  medium: { color: '#f39c12', icon: '🟡', label: 'Medium' },
  low: { color: '#27ae60', icon: '🟢', label: 'Low' },
};

export function RiskBadge({ level, showLabel = true }) {
  const config = riskConfig[level] || riskConfig.medium;
  
  return (
    <span
      className="risk-badge"
      style={{ '--risk-color': config.color }}
    >
      <span className="risk-icon">{config.icon}</span>
      {showLabel && <span className="risk-label">{config.label}</span>}
    </span>
  );
}
```

**File: `dashboard/src/components/Approvals/ApprovalHistory.jsx`**

```jsx
import { useState, useEffect } from 'react';
import { DataTable } from '../Common/DataTable';
import { RiskBadge } from './RiskBadge';
import { StatusIndicator } from '../Common/StatusIndicator';
import './Approvals.css';

export function ApprovalHistory({ businessId }) {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  
  useEffect(() => {
    fetchHistory();
  }, [businessId]);
  
  const fetchHistory = async () => {
    setLoading(true);
    try {
      const res = await fetch(`/api/approvals/history/${businessId}`);
      const data = await res.json();
      setHistory(data.requests || []);
    } catch (err) {
      console.error('Failed to fetch history:', err);
    }
    setLoading(false);
  };
  
  const columns = [
    { key: 'title', label: 'Action' },
    { key: 'type', label: 'Type' },
    {
      key: 'risk_level',
      label: 'Risk',
      render: (val) => <RiskBadge level={val} showLabel={false} />,
    },
    {
      key: 'status',
      label: 'Status',
      render: (val) => <StatusIndicator status={val} />,
    },
    {
      key: 'created_at',
      label: 'Created',
      render: (val) => new Date(val).toLocaleDateString(),
    },
    { key: 'reviewed_by', label: 'Reviewer' },
  ];
  
  if (loading) {
    return <div className="loading">Loading history...</div>;
  }
  
  return (
    <div className="approval-history">
      <h2>Approval History</h2>
      <DataTable columns={columns} data={history} />
    </div>
  );
}
```

---

### Task 23.5: Approval Styles

**File: `dashboard/src/components/Approvals/Approvals.css`**

```css
/* Approval Queue */
.approval-queue {
  padding: 20px;
}

.queue-header {
  display: flex;
  align-items: center;
  gap: 16px;
  margin-bottom: 20px;
}

.queue-header h2 {
  margin: 0;
}

.count-badge {
  background: #e74c3c;
  color: white;
  padding: 4px 12px;
  border-radius: 12px;
  font-weight: 600;
}

.filter-tabs {
  display: flex;
  gap: 8px;
  margin-left: auto;
}

.filter-tab {
  padding: 6px 12px;
  border: 1px solid #ddd;
  background: white;
  border-radius: 4px;
  cursor: pointer;
  text-transform: capitalize;
}

.filter-tab.active {
  background: #3498db;
  color: white;
  border-color: #3498db;
}

/* Empty Queue */
.empty-queue {
  text-align: center;
  padding: 60px 20px;
  color: #95a5a6;
}

.empty-icon {
  font-size: 48px;
  display: block;
  margin-bottom: 16px;
}

/* Approval Card */
.approval-card {
  background: white;
  border-radius: 8px;
  margin-bottom: 12px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  border-left: 4px solid #95a5a6;
}

.approval-card.risk-critical { border-left-color: #c0392b; }
.approval-card.risk-high { border-left-color: #e74c3c; }
.approval-card.risk-medium { border-left-color: #f39c12; }
.approval-card.risk-low { border-left-color: #27ae60; }

.card-main {
  display: flex;
  align-items: center;
  padding: 16px;
  cursor: pointer;
}

.card-main:hover {
  background: #f9f9f9;
}

.card-icon {
  font-size: 24px;
  margin-right: 16px;
}

.card-content {
  flex: 1;
}

.card-title {
  margin: 0 0 4px 0;
  font-size: 16px;
}

.card-desc {
  margin: 0;
  color: #666;
  font-size: 14px;
}

.card-meta {
  display: flex;
  gap: 16px;
  margin-top: 8px;
  font-size: 12px;
  color: #999;
}

.expand-icon {
  color: #999;
  margin-left: 16px;
}

/* Card Details */
.card-details {
  padding: 16px;
  border-top: 1px solid #eee;
  background: #fafafa;
}

.risk-factors h4,
.payload-preview h4 {
  margin: 0 0 12px 0;
  font-size: 14px;
  color: #666;
}

.risk-factor {
  padding: 8px 12px;
  margin-bottom: 8px;
  border-radius: 4px;
  background: white;
  border-left: 3px solid #95a5a6;
}

.risk-factor.severity-critical { border-left-color: #c0392b; }
.risk-factor.severity-high { border-left-color: #e74c3c; }
.risk-factor.severity-medium { border-left-color: #f39c12; }

.factor-category {
  font-weight: 600;
  font-size: 12px;
  text-transform: uppercase;
  color: #666;
}

.payload-preview pre {
  background: white;
  padding: 12px;
  border-radius: 4px;
  overflow: auto;
  max-height: 200px;
  font-size: 12px;
}

/* Action Form */
.action-form {
  margin-top: 16px;
}

.action-form textarea {
  width: 100%;
  padding: 12px;
  border: 1px solid #ddd;
  border-radius: 4px;
  resize: vertical;
  margin-bottom: 12px;
}

.action-buttons {
  display: flex;
  justify-content: flex-end;
  gap: 12px;
}

.btn {
  padding: 10px 24px;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-weight: 600;
}

.btn-approve {
  background: #27ae60;
  color: white;
}

.btn-reject {
  background: #e74c3c;
  color: white;
}

.btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

/* Risk Badge */
.risk-badge {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 4px 8px;
  border-radius: 4px;
  background: color-mix(in srgb, var(--risk-color) 15%, transparent);
  font-size: 12px;
}

.risk-label {
  color: var(--risk-color);
  font-weight: 600;
}
```

---

### Task 23.6: Tests

**File: `tests/test_approvals.py`**

```python
"""Tests for Approval System."""
import pytest
from src.approvals.models import ApprovalStatus, ApprovalType, RiskLevel, RiskFactor
from src.approvals.manager import ApprovalManager


@pytest.fixture
def manager():
    return ApprovalManager()


class TestApprovalManager:
    @pytest.mark.asyncio
    async def test_create_request(self, manager):
        request = await manager.create_request(
            business_id="biz_1",
            action_type=ApprovalType.FINANCIAL,
            title="Large Purchase",
            description="Purchase inventory",
            payload={"amount": 5000},
            risk_level=RiskLevel.HIGH,
        )
        
        assert request.id is not None
        assert request.status == ApprovalStatus.PENDING

    @pytest.mark.asyncio
    async def test_auto_approve(self, manager):
        request = await manager.create_request(
            business_id="biz_1",
            action_type=ApprovalType.FINANCIAL,
            title="Small Purchase",
            description="Minor expense",
            payload={"amount": 50},  # Below auto-approve threshold
            risk_level=RiskLevel.LOW,
        )
        
        assert request.status == ApprovalStatus.APPROVED

    @pytest.mark.asyncio
    async def test_approve_request(self, manager):
        request = await manager.create_request(
            business_id="biz_1",
            action_type=ApprovalType.OPERATIONAL,
            title="Test Action",
            description="Test",
            payload={},
            risk_level=RiskLevel.MEDIUM,
        )
        
        approved = await manager.approve(request.id, "user_1", "Looks good")
        
        assert approved.status == ApprovalStatus.APPROVED
        assert approved.reviewed_by == "user_1"

    @pytest.mark.asyncio
    async def test_reject_request(self, manager):
        request = await manager.create_request(
            business_id="biz_1",
            action_type=ApprovalType.EXTERNAL,
            title="Test Action",
            description="Test",
            payload={},
        )
        
        rejected = await manager.reject(request.id, "user_1", "Too risky")
        
        assert rejected.status == ApprovalStatus.REJECTED

    @pytest.mark.asyncio
    async def test_get_pending(self, manager):
        await manager.create_request(
            business_id="biz_1",
            action_type=ApprovalType.FINANCIAL,
            title="Action 1",
            description="",
            payload={},
        )
        await manager.create_request(
            business_id="biz_1",
            action_type=ApprovalType.LEGAL,
            title="Action 2",
            description="",
            payload={},
        )
        
        pending = await manager.get_pending("biz_1")
        
        assert len(pending) == 2

    @pytest.mark.asyncio
    async def test_get_stats(self, manager):
        await manager.create_request(
            business_id="biz_1",
            action_type=ApprovalType.OPERATIONAL,
            title="Test",
            description="",
            payload={},
        )
        
        stats = await manager.get_stats("biz_1")
        
        assert stats["pending"] == 1
        assert stats["total"] == 1
```

---

## 3. Acceptance Criteria

| Criteria | Validation |
|----------|------------|
| Requests created | With risk level and factors |
| Auto-approve works | Low-value items auto-approved |
| Approve/reject | Updates status correctly |
| Queue displays | Shows pending with filters |
| Risk displayed | Color-coded risk badges |
| History tracked | Audit trail available |

---

## 4. File Summary

| File | Purpose |
|------|---------|
| `src/approvals/models.py` | Data models for approvals |
| `src/approvals/manager.py` | Approval workflow logic |
| `src/api/routes/approvals.py` | REST API endpoints |
| `components/Approvals/ApprovalQueue.jsx` | Pending approvals list |
| `components/Approvals/ApprovalCard.jsx` | Individual approval UI |
| `components/Approvals/RiskBadge.jsx` | Risk level indicator |
| `components/Approvals/ApprovalHistory.jsx` | History table |
| `components/Approvals/Approvals.css` | Component styles |
| `tests/test_approvals.py` | Unit tests |
