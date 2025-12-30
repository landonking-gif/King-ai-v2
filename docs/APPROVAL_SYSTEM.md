# Approval System Documentation

## Overview

The Approval System provides a comprehensive human-in-the-loop workflow for managing actions that require manual review before execution. It supports risk-based routing, auto-approval policies, and complete audit trails.

## Features

- **Risk-Based Routing**: Automatically route requests based on risk level (Low, Medium, High, Critical)
- **Auto-Approval**: Configure policies to automatically approve low-risk, low-value actions
- **Multiple Action Types**: Support for Financial, Legal, Operational, Strategic, Technical, and External actions
- **Audit Trail**: Complete history of all approval decisions
- **Real-time Filtering**: Filter pending approvals by risk level
- **Statistics**: Track approval rates, wait times, and high-risk items

## Backend Usage

### Creating an Approval Request

```python
from src.approvals.manager import ApprovalManager
from src.approvals.models import ApprovalType, RiskLevel, RiskFactor

manager = ApprovalManager()

# Create a request requiring approval
request = await manager.create_request(
    business_id="business_123",
    action_type=ApprovalType.FINANCIAL,
    title="Server Infrastructure Upgrade",
    description="Purchase new cloud server instances",
    payload={"amount": 5000, "provider": "AWS"},
    risk_level=RiskLevel.HIGH,
    risk_factors=[
        RiskFactor(
            category="Financial",
            description="Large expenditure requiring budget approval",
            severity=RiskLevel.HIGH
        )
    ]
)
```

### Approving a Request

```python
approved_request = await manager.approve(
    request_id=request.id,
    user_id="admin@example.com",
    notes="Approved after budget review"
)
```

### Rejecting a Request

```python
rejected_request = await manager.reject(
    request_id=request.id,
    user_id="admin@example.com",
    notes="Budget constraints - please resubmit next quarter"
)
```

### Getting Pending Requests

```python
# Get all pending requests
pending = await manager.get_pending()

# Filter by business
pending = await manager.get_pending(business_id="business_123")

# Filter by action type
pending = await manager.get_pending(action_type=ApprovalType.FINANCIAL)
```

### Viewing Statistics

```python
stats = await manager.get_stats("business_123")
print(f"Approval rate: {stats['approval_rate']:.1f}%")
print(f"Average wait time: {stats['avg_wait_hours']:.1f} hours")
```

## API Endpoints

### GET /api/approvals/pending

Get pending approval requests.

**Query Parameters:**
- `business_id` (optional): Filter by business ID
- `action_type` (optional): Filter by action type

**Response:**
```json
{
  "count": 2,
  "requests": [
    {
      "id": "req_123",
      "business_id": "business_123",
      "type": "financial",
      "title": "Server Upgrade",
      "description": "Purchase new servers",
      "risk_level": "high",
      "risk_factors": [...],
      "payload": {...},
      "created_at": "2025-12-30T20:00:00Z",
      "expires_at": "2025-12-31T20:00:00Z",
      "waiting_hours": 2.5
    }
  ]
}
```

### POST /api/approvals/{request_id}/approve

Approve a request.

**Body:**
```json
{
  "user_id": "admin@example.com",
  "notes": "Approved after review",
  "modifications": {
    "amount": 4500
  }
}
```

### POST /api/approvals/{request_id}/reject

Reject a request.

**Body:**
```json
{
  "user_id": "admin@example.com",
  "notes": "Budget constraints"
}
```

### GET /api/approvals/stats

Get approval statistics.

**Query Parameters:**
- `business_id` (optional): Filter by business ID

### GET /api/approvals/history/{business_id}

Get approval history for a business.

**Query Parameters:**
- `limit` (optional, default=50): Maximum number of records

## Frontend Usage

### Using ApprovalQueue Component

```jsx
import { ApprovalQueue } from './components/Approvals';

function MyComponent() {
  const handleApprove = async (id, data) => {
    await fetch(`/api/approvals/${id}/approve`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        user_id: 'admin@example.com',
        notes: data.notes
      })
    });
  };

  const handleReject = async (id, data) => {
    await fetch(`/api/approvals/${id}/reject`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        user_id: 'admin@example.com',
        notes: data.notes
      })
    });
  };

  return (
    <ApprovalQueue
      businessId="business_123"
      onApprove={handleApprove}
      onReject={handleReject}
    />
  );
}
```

### Using ApprovalHistory Component

```jsx
import { ApprovalHistory } from './components/Approvals';

function HistoryPage() {
  return <ApprovalHistory businessId="business_123" />;
}
```

## Auto-Approval Policies

The system includes default policies that automatically approve certain requests:

### Financial Policy
- Action Type: FINANCIAL
- Auto-approve: Amounts below $100
- Expiry: 24 hours

### Legal Policy
- Action Type: LEGAL
- Auto-approve: None (all require approval)
- Expiry: 48 hours
- Requires: Two approvers

### External Communications Policy
- Action Type: EXTERNAL
- Expiry: 12 hours

## Risk Levels

- **Low**: Minor impact, routine operations
- **Medium**: Moderate impact, requires attention
- **High**: Significant impact, careful review needed
- **Critical**: Major impact, highest priority review

## Status Values

- **pending**: Awaiting human review
- **approved**: Approved for execution
- **rejected**: Denied
- **expired**: Request expired before review
- **modified**: Approved with modifications

## Testing

Run the test suite:
```bash
python -m pytest tests/test_approvals.py -v
```

Run the demo script:
```bash
PYTHONPATH=/home/runner/work/King-ai-v2/King-ai-v2 python scripts/demo_approvals.py
```

## Integration with ReAct Planner

To integrate with the ReAct Planner, register hooks for approval events:

```python
async def on_approval_granted(request):
    # Resume the paused task
    print(f"Resuming task after approval: {request.source_task_id}")

manager.register_hook("request_approved", on_approval_granted)
```

## Best Practices

1. **Set Appropriate Risk Levels**: Use risk levels consistently to ensure proper routing
2. **Provide Context**: Include detailed descriptions and risk factors to help reviewers
3. **Monitor Wait Times**: Track average wait times and adjust policies if needed
4. **Regular Audits**: Review approval history periodically to identify patterns
5. **Configure Auto-Approval**: Set reasonable thresholds to reduce manual review burden
6. **Document Decisions**: Always provide notes when approving or rejecting requests
