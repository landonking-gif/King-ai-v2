#!/usr/bin/env python3
"""
Example script demonstrating the Approval System.
This script shows how to create approval requests, approve/reject them, and view statistics.
"""

import asyncio
from src.approvals.manager import ApprovalManager
from src.approvals.models import ApprovalType, RiskLevel, RiskFactor


async def demo_approval_system():
    """Demonstrate the approval system functionality."""
    print("=" * 60)
    print("Approval System Demo")
    print("=" * 60)
    
    # Create manager
    manager = ApprovalManager()
    
    # Create some approval requests
    print("\n1. Creating approval requests...")
    
    # Small financial request (will be auto-approved)
    request1 = await manager.create_request(
        business_id="business_123",
        action_type=ApprovalType.FINANCIAL,
        title="Office Supplies Purchase",
        description="Purchase printer paper and pens",
        payload={"amount": 45.99, "vendor": "Office Depot"},
        risk_level=RiskLevel.LOW,
    )
    print(f"   Request 1: {request1.title} - Status: {request1.status.value}")
    
    # Large financial request (requires approval)
    request2 = await manager.create_request(
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
            ),
            RiskFactor(
                category="Operational",
                description="Critical infrastructure change",
                severity=RiskLevel.MEDIUM
            )
        ]
    )
    print(f"   Request 2: {request2.title} - Status: {request2.status.value}")
    
    # Legal request (requires approval)
    request3 = await manager.create_request(
        business_id="business_123",
        action_type=ApprovalType.LEGAL,
        title="Partnership Agreement",
        description="Review and sign partnership agreement with vendor",
        payload={"partner": "Acme Corp", "contract_value": 10000},
        risk_level=RiskLevel.CRITICAL,
        risk_factors=[
            RiskFactor(
                category="Legal",
                description="Binding legal agreement",
                severity=RiskLevel.CRITICAL
            )
        ]
    )
    print(f"   Request 3: {request3.title} - Status: {request3.status.value}")
    
    # External communication request
    request4 = await manager.create_request(
        business_id="business_123",
        action_type=ApprovalType.EXTERNAL,
        title="Press Release",
        description="Publish press release about new product",
        payload={"content": "Announcing our new product...", "channels": ["website", "social_media"]},
        risk_level=RiskLevel.MEDIUM,
    )
    print(f"   Request 4: {request4.title} - Status: {request4.status.value}")
    
    # View pending requests
    print("\n2. Viewing pending requests...")
    pending = await manager.get_pending("business_123")
    print(f"   Found {len(pending)} pending requests:")
    for req in pending:
        print(f"   - {req.title} (Risk: {req.risk_level.value}, Waiting: {req.waiting_hours:.1f}h)")
    
    # Approve a request
    print("\n3. Approving server infrastructure upgrade...")
    approved_request = await manager.approve(
        request2.id,
        user_id="admin@example.com",
        notes="Approved after budget review. Proceed with AWS setup."
    )
    print(f"   Status: {approved_request.status.value}")
    print(f"   Reviewed by: {approved_request.reviewed_by}")
    
    # Reject a request
    print("\n4. Rejecting partnership agreement...")
    rejected_request = await manager.reject(
        request3.id,
        user_id="legal@example.com",
        notes="Terms need renegotiation. Please revise Section 3."
    )
    print(f"   Status: {rejected_request.status.value}")
    print(f"   Reviewed by: {rejected_request.reviewed_by}")
    
    # Approve with modifications
    print("\n5. Approving press release with modifications...")
    modified_request = await manager.approve(
        request4.id,
        user_id="marketing@example.com",
        notes="Approved with content changes",
        modifications={
            "content": "Updated press release content...",
            "channels": ["website"]  # Removed social media
        }
    )
    print(f"   Status: {modified_request.status.value}")
    print(f"   Modified payload: {modified_request.modified_payload}")
    
    # View statistics
    print("\n6. Approval statistics:")
    stats = await manager.get_stats("business_123")
    print(f"   Total requests: {stats['total']}")
    print(f"   Pending: {stats['pending']}")
    print(f"   Approved: {stats['approved']}")
    print(f"   Rejected: {stats['rejected']}")
    print(f"   Approval rate: {stats['approval_rate']:.1f}%")
    print(f"   High-risk pending: {stats['high_risk_pending']}")
    
    # View history
    print("\n7. Recent approval history:")
    history = await manager.get_history("business_123", limit=5)
    for req in history:
        status_str = f"{req.status.value}"
        if req.reviewed_by:
            status_str += f" by {req.reviewed_by}"
        print(f"   - {req.title}: {status_str}")
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(demo_approval_system())
