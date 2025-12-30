"""
Unit tests for Business Logic.
Tests business creation, lifecycle transitions, and portfolio aggregation.
"""

import pytest
from src.business.lifecycle import LifecycleEngine
from src.database.models import BusinessStatus
from src.business.portfolio import PortfolioManager
from unittest.mock import MagicMock

# --- Lifecycle Engine Tests ---

def test_lifecycle_transitions():
    engine = LifecycleEngine()
    
    # Valid transition
    next_step = engine.get_next_status(BusinessStatus.DISCOVERY)
    assert next_step == BusinessStatus.VALIDATION
    
    # Late stage transition
    next_step = engine.get_next_status(BusinessStatus.OPTIMIZATION)
    assert next_step == BusinessStatus.REPLICATION
    
    # End state should return None
    next_step = engine.get_next_status(BusinessStatus.REPLICATION)
    assert next_step is None

def test_is_failed():
    engine = LifecycleEngine()
    assert engine.is_failed(BusinessStatus.SUNSET) == True
    assert engine.is_failed(BusinessStatus.OPERATION) == False

# --- Portfolio Manager Tests ---

def test_portfolio_count_stages():
    mgr = PortfolioManager()
    
    # Create mock BusinessUnit objects
    unit1 = MagicMock(status=BusinessStatus.OPERATION)
    unit2 = MagicMock(status=BusinessStatus.OPERATION)
    unit3 = MagicMock(status=BusinessStatus.DISCOVERY)
    
    # Check counting logic
    stats = mgr._count_stages([unit1, unit2, unit3])
    
    assert stats["operation"] == 2
    assert stats["discovery"] == 1
    assert "setup" not in stats
