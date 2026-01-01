"""
Test Data Factories.
Factory classes for generating realistic test data.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import random
import string


def random_string(length: int = 10) -> str:
    """Generate a random string."""
    return "".join(random.choices(string.ascii_lowercase, k=length))


def random_email() -> str:
    """Generate a random email."""
    return f"{random_string(8)}@{random_string(5)}.com"


def random_uuid() -> str:
    """Generate a random UUID."""
    return str(uuid.uuid4())


class Factory:
    """Base factory class."""
    
    _counter = 0
    
    @classmethod
    def _next_id(cls) -> int:
        cls._counter += 1
        return cls._counter
    
    @classmethod
    def reset(cls) -> None:
        """Reset the factory counter."""
        cls._counter = 0


class BusinessFactory(Factory):
    """Factory for creating test Business instances."""
    
    BUSINESS_TYPES = ["dropshipping", "saas", "agency", "ecommerce", "consulting"]
    STATUSES = ["active", "planning", "paused", "archived"]
    
    @classmethod
    def create(
        cls,
        id: Optional[str] = None,
        name: Optional[str] = None,
        business_type: Optional[str] = None,
        status: Optional[str] = None,
        revenue: Optional[float] = None,
        expenses: Optional[float] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Create a test business."""
        counter = cls._next_id()
        
        return {
            "id": id or random_uuid(),
            "name": name or f"Test Business {counter}",
            "business_type": business_type or random.choice(cls.BUSINESS_TYPES),
            "status": status or "active",
            "description": kwargs.get("description", f"Test business description {counter}"),
            "revenue": revenue if revenue is not None else random.uniform(1000, 100000),
            "expenses": expenses if expenses is not None else random.uniform(500, 50000),
            "profit_margin": kwargs.get("profit_margin", random.uniform(0.1, 0.4)),
            "customer_count": kwargs.get("customer_count", random.randint(10, 1000)),
            "created_at": kwargs.get("created_at", datetime.utcnow()),
            "updated_at": kwargs.get("updated_at", datetime.utcnow()),
            "metadata": kwargs.get("metadata", {}),
            **{k: v for k, v in kwargs.items() if k not in [
                "description", "profit_margin", "customer_count", 
                "created_at", "updated_at", "metadata"
            ]},
        }
    
    @classmethod
    def create_batch(cls, count: int, **kwargs) -> List[Dict[str, Any]]:
        """Create multiple test businesses."""
        return [cls.create(**kwargs) for _ in range(count)]
    
    @classmethod
    def create_profitable(cls, **kwargs) -> Dict[str, Any]:
        """Create a profitable business."""
        revenue = random.uniform(50000, 200000)
        return cls.create(
            revenue=revenue,
            expenses=revenue * 0.6,  # 40% profit margin
            **kwargs,
        )
    
    @classmethod
    def create_struggling(cls, **kwargs) -> Dict[str, Any]:
        """Create a struggling business."""
        revenue = random.uniform(10000, 50000)
        return cls.create(
            revenue=revenue,
            expenses=revenue * 1.2,  # 20% loss
            status="paused",
            **kwargs,
        )


class UserFactory(Factory):
    """Factory for creating test User instances."""
    
    ROLES = ["owner", "admin", "approver", "viewer"]
    
    @classmethod
    def create(
        cls,
        id: Optional[str] = None,
        email: Optional[str] = None,
        name: Optional[str] = None,
        role: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Create a test user."""
        counter = cls._next_id()
        
        return {
            "id": id or random_uuid(),
            "email": email or random_email(),
            "name": name or f"Test User {counter}",
            "role": role or "approver",
            "is_active": kwargs.get("is_active", True),
            "created_at": kwargs.get("created_at", datetime.utcnow()),
            "last_login": kwargs.get("last_login"),
            "preferences": kwargs.get("preferences", {}),
        }
    
    @classmethod
    def create_admin(cls, **kwargs) -> Dict[str, Any]:
        """Create an admin user."""
        return cls.create(role="admin", **kwargs)
    
    @classmethod
    def create_owner(cls, **kwargs) -> Dict[str, Any]:
        """Create an owner user."""
        return cls.create(role="owner", **kwargs)


class ApprovalRequestFactory(Factory):
    """Factory for creating test ApprovalRequest instances."""
    
    ACTION_TYPES = ["financial", "legal", "external", "system", "strategic"]
    RISK_LEVELS = ["low", "medium", "high", "critical"]
    STATUSES = ["pending", "approved", "rejected", "expired", "partial"]
    
    @classmethod
    def create(
        cls,
        id: Optional[str] = None,
        business_id: Optional[str] = None,
        action_type: Optional[str] = None,
        risk_level: Optional[str] = None,
        status: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Create a test approval request."""
        counter = cls._next_id()
        
        return {
            "id": id or random_uuid(),
            "business_id": business_id or random_uuid(),
            "action_type": action_type or random.choice(cls.ACTION_TYPES),
            "title": kwargs.get("title", f"Test Approval {counter}"),
            "description": kwargs.get("description", f"Test approval description {counter}"),
            "risk_level": risk_level or random.choice(cls.RISK_LEVELS),
            "risk_factors": kwargs.get("risk_factors", []),
            "payload": kwargs.get("payload", {"amount": random.uniform(100, 10000)}),
            "status": status or "pending",
            "created_at": kwargs.get("created_at", datetime.utcnow()),
            "expires_at": kwargs.get("expires_at", datetime.utcnow() + timedelta(hours=24)),
            "reviewed_at": kwargs.get("reviewed_at"),
            "reviewed_by": kwargs.get("reviewed_by"),
            "review_notes": kwargs.get("review_notes"),
        }
    
    @classmethod
    def create_high_risk(cls, **kwargs) -> Dict[str, Any]:
        """Create a high-risk approval request."""
        return cls.create(
            risk_level="high",
            risk_factors=[
                {"name": "large_amount", "severity": 0.8},
                {"name": "new_vendor", "severity": 0.6},
            ],
            **kwargs,
        )
    
    @classmethod
    def create_expired(cls, **kwargs) -> Dict[str, Any]:
        """Create an expired approval request."""
        return cls.create(
            status="expired",
            created_at=datetime.utcnow() - timedelta(days=2),
            expires_at=datetime.utcnow() - timedelta(hours=1),
            **kwargs,
        )
    
    @classmethod
    def create_approved(cls, **kwargs) -> Dict[str, Any]:
        """Create an approved request."""
        return cls.create(
            status="approved",
            reviewed_at=datetime.utcnow(),
            reviewed_by=kwargs.pop("reviewed_by", random_uuid()),
            **kwargs,
        )


class TransactionFactory(Factory):
    """Factory for creating test Transaction instances."""
    
    TYPES = ["sale", "refund", "payout", "fee", "adjustment"]
    STATUSES = ["pending", "completed", "failed", "cancelled"]
    
    @classmethod
    def create(
        cls,
        id: Optional[str] = None,
        business_id: Optional[str] = None,
        amount: Optional[float] = None,
        transaction_type: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Create a test transaction."""
        counter = cls._next_id()
        
        return {
            "id": id or random_uuid(),
            "business_id": business_id or random_uuid(),
            "amount": amount if amount is not None else random.uniform(10, 1000),
            "currency": kwargs.get("currency", "USD"),
            "type": transaction_type or random.choice(cls.TYPES),
            "status": kwargs.get("status", "completed"),
            "description": kwargs.get("description", f"Test transaction {counter}"),
            "created_at": kwargs.get("created_at", datetime.utcnow()),
            "metadata": kwargs.get("metadata", {}),
        }
    
    @classmethod
    def create_sale(cls, amount: float, **kwargs) -> Dict[str, Any]:
        """Create a sale transaction."""
        return cls.create(amount=amount, transaction_type="sale", **kwargs)
    
    @classmethod
    def create_refund(cls, amount: float, **kwargs) -> Dict[str, Any]:
        """Create a refund transaction."""
        return cls.create(amount=-abs(amount), transaction_type="refund", **kwargs)


class PlanFactory(Factory):
    """Factory for creating test Plan instances."""
    
    PLAN_TYPES = ["optimization", "expansion", "maintenance", "recovery"]
    STATUSES = ["draft", "active", "completed", "failed", "cancelled"]
    
    @classmethod
    def create(
        cls,
        id: Optional[str] = None,
        business_id: Optional[str] = None,
        plan_type: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Create a test plan."""
        counter = cls._next_id()
        
        return {
            "id": id or random_uuid(),
            "business_id": business_id or random_uuid(),
            "name": kwargs.get("name", f"Test Plan {counter}"),
            "description": kwargs.get("description", f"Test plan description {counter}"),
            "type": plan_type or random.choice(cls.PLAN_TYPES),
            "status": kwargs.get("status", "draft"),
            "priority": kwargs.get("priority", random.randint(1, 10)),
            "tasks": kwargs.get("tasks", []),
            "created_at": kwargs.get("created_at", datetime.utcnow()),
            "started_at": kwargs.get("started_at"),
            "completed_at": kwargs.get("completed_at"),
        }
    
    @classmethod
    def create_with_tasks(cls, task_count: int = 3, **kwargs) -> Dict[str, Any]:
        """Create a plan with tasks."""
        plan_id = random_uuid()
        tasks = [
            TaskFactory.create(plan_id=plan_id)
            for _ in range(task_count)
        ]
        return cls.create(id=plan_id, tasks=tasks, **kwargs)


class TaskFactory(Factory):
    """Factory for creating test Task instances."""
    
    STATUSES = ["pending", "in_progress", "completed", "failed", "blocked"]
    TYPES = ["research", "action", "decision", "communication", "integration"]
    
    @classmethod
    def create(
        cls,
        id: Optional[str] = None,
        plan_id: Optional[str] = None,
        task_type: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Create a test task."""
        counter = cls._next_id()
        
        return {
            "id": id or random_uuid(),
            "plan_id": plan_id or random_uuid(),
            "name": kwargs.get("name", f"Test Task {counter}"),
            "description": kwargs.get("description", f"Test task description {counter}"),
            "type": task_type or random.choice(cls.TYPES),
            "status": kwargs.get("status", "pending"),
            "assigned_agent": kwargs.get("assigned_agent"),
            "priority": kwargs.get("priority", random.randint(1, 10)),
            "dependencies": kwargs.get("dependencies", []),
            "created_at": kwargs.get("created_at", datetime.utcnow()),
            "started_at": kwargs.get("started_at"),
            "completed_at": kwargs.get("completed_at"),
            "result": kwargs.get("result"),
        }


class EvolutionProposalFactory(Factory):
    """Factory for creating test EvolutionProposal instances."""
    
    TYPES = ["capability", "optimization", "refactor", "feature"]
    STATUSES = ["draft", "testing", "approved", "deployed", "rejected"]
    
    @classmethod
    def create(
        cls,
        id: Optional[str] = None,
        proposal_type: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Create a test evolution proposal."""
        counter = cls._next_id()
        
        return {
            "id": id or random_uuid(),
            "title": kwargs.get("title", f"Test Evolution {counter}"),
            "description": kwargs.get("description", f"Test evolution description {counter}"),
            "type": proposal_type or random.choice(cls.TYPES),
            "status": kwargs.get("status", "draft"),
            "confidence_score": kwargs.get("confidence_score", random.uniform(0.5, 1.0)),
            "changes": kwargs.get("changes", []),
            "test_results": kwargs.get("test_results", {}),
            "created_at": kwargs.get("created_at", datetime.utcnow()),
            "deployed_at": kwargs.get("deployed_at"),
        }


class KPIFactory(Factory):
    """Factory for creating test KPI instances."""
    
    KPI_NAMES = [
        "revenue", "profit_margin", "customer_count", "churn_rate",
        "conversion_rate", "average_order_value", "customer_acquisition_cost"
    ]
    
    @classmethod
    def create(
        cls,
        business_id: Optional[str] = None,
        name: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Create a test KPI."""
        kpi_name = name or random.choice(cls.KPI_NAMES)
        
        return {
            "business_id": business_id or random_uuid(),
            "name": kpi_name,
            "value": kwargs.get("value", random.uniform(0, 100)),
            "target": kwargs.get("target", random.uniform(50, 150)),
            "unit": kwargs.get("unit", "USD" if "revenue" in kpi_name.lower() else "%"),
            "trend": kwargs.get("trend", random.choice(["up", "down", "stable"])),
            "period": kwargs.get("period", "monthly"),
            "recorded_at": kwargs.get("recorded_at", datetime.utcnow()),
        }
    
    @classmethod
    def create_healthy(cls, **kwargs) -> Dict[str, Any]:
        """Create a KPI exceeding target."""
        target = random.uniform(50, 100)
        return cls.create(
            value=target * 1.2,  # 20% above target
            target=target,
            trend="up",
            **kwargs,
        )
    
    @classmethod
    def create_warning(cls, **kwargs) -> Dict[str, Any]:
        """Create a KPI below target."""
        target = random.uniform(50, 100)
        return cls.create(
            value=target * 0.8,  # 20% below target
            target=target,
            trend="down",
            **kwargs,
        )
