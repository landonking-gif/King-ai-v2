"""
P&L Tracking Models and Services for Master Control Panel
"""

from datetime import datetime
from decimal import Decimal
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from enum import Enum

class TransactionType(str, Enum):
    REVENUE = "revenue"
    EXPENSE = "expense"

class ExpenseCategory(str, Enum):
    LLM_COSTS = "llm_costs"
    INFRASTRUCTURE = "infrastructure"
    TOOLS = "tools"
    PERSONNEL = "personnel"
    OTHER = "other"

class RevenueCategory(str, Enum):
    WORKFLOW_EXECUTION = "workflow_execution"
    API_CALLS = "api_calls"
    SUBSCRIPTIONS = "subscriptions"
    OTHER = "other"

class BusinessUnit(BaseModel):
    id: str
    name: str
    description: str
    created_at: datetime
    updated_at: Optional[datetime] = None

class FinancialTransaction(BaseModel):
    id: str
    business_unit_id: str
    workflow_id: Optional[str] = None
    transaction_type: TransactionType
    category: str
    amount: Decimal
    description: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None

class PLSummary(BaseModel):
    period: str  # "daily", "weekly", "monthly"
    total_revenue: Decimal
    total_expenses: Decimal
    net_profit: Decimal
    margin_percent: float
    transaction_count: int
    avg_transaction_value: Decimal

class CostBreakdown(BaseModel):
    category: str
    amount: Decimal
    percentage: float
    transactions: int

class RevenueTrend(BaseModel):
    date: str
    revenue: Decimal
    expenses: Decimal
    profit: Decimal
    margin_percent: float

class BudgetAlert(BaseModel):
    id: str
    business_unit_id: str
    category: str
    threshold: Decimal
    current: Decimal
    percentage: float
    triggered: bool
    created_at: datetime

class ROICalculation(BaseModel):
    workflow_id: str
    total_investment: Decimal
    total_revenue: Decimal
    roi_percent: float
    payback_period_days: Optional[int] = None
