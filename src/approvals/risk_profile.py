"""
Risk Profile Management - Loads and manages risk configurations from YAML.
"""

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Literal
from enum import Enum

from src.utils.structured_logging import get_logger

logger = get_logger("risk_profile")


class RiskAction(Enum):
    """Types of actions that can be controlled by risk profiles."""
    LEGAL_ACTIONS = "legal_actions"
    FINANCIAL_TRANSACTIONS = "financial_transactions"
    EXTERNAL_API_CALLS = "external_api_calls"
    CODE_MODIFICATIONS = "code_modifications"
    RESEARCH = "research"
    ANALYSIS = "analysis"
    PLANNING = "planning"
    CONTENT_CREATION = "content_creation"
    SMALL_PURCHASES = "small_purchases"
    ALL = "all"


@dataclass
class RiskProfile:
    """
    Risk profile configuration loaded from YAML.
    
    Defines the autonomy level and approval requirements for a specific risk tolerance.
    """
    name: str
    max_spend_without_approval: float
    require_approval_for: List[str] = field(default_factory=list)
    autonomous_actions_allowed: List[str] = field(default_factory=list)
    
    # Additional settings
    max_daily_transactions: int = 100
    max_single_transaction: float = 10000.0
    require_two_factor_for_amount: float = 1000.0
    escalation_contacts: List[str] = field(default_factory=list)
    
    def requires_approval(self, action: str, amount: float = 0.0) -> bool:
        """
        Check if an action requires human approval.
        
        Args:
            action: The action type (e.g., 'legal_actions', 'code_modifications')
            amount: The monetary amount involved (if applicable)
            
        Returns:
            True if approval is required, False otherwise
        """
        # Check if action is in require_approval_for list
        if action in self.require_approval_for:
            return True
        
        # Check if amount exceeds threshold
        if amount > 0 and amount > self.max_spend_without_approval:
            return True
        
        return False
    
    def is_action_allowed(self, action: str) -> bool:
        """
        Check if an action is allowed autonomously.
        
        Args:
            action: The action type
            
        Returns:
            True if action can be performed autonomously
        """
        if "all" in self.autonomous_actions_allowed:
            return True
        
        return action in self.autonomous_actions_allowed
    
    def get_approval_level(self, amount: float) -> str:
        """
        Get the required approval level based on amount.
        
        Args:
            amount: The monetary amount
            
        Returns:
            'auto', 'single', or 'multi' approval level
        """
        if amount <= self.max_spend_without_approval:
            return "auto"
        elif amount <= self.require_two_factor_for_amount:
            return "single"
        else:
            return "multi"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "max_spend_without_approval": self.max_spend_without_approval,
            "require_approval_for": self.require_approval_for,
            "autonomous_actions_allowed": self.autonomous_actions_allowed,
            "max_daily_transactions": self.max_daily_transactions,
            "max_single_transaction": self.max_single_transaction,
            "require_two_factor_for_amount": self.require_two_factor_for_amount,
            "escalation_contacts": self.escalation_contacts,
        }


class RiskProfileManager:
    """
    Manages risk profiles loaded from YAML configuration.
    """
    
    _instance: Optional["RiskProfileManager"] = None
    _profiles: Dict[str, RiskProfile] = {}
    _active_profile: Optional[RiskProfile] = None
    
    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_profiles()
        return cls._instance
    
    def _load_profiles(self):
        """Load risk profiles from YAML file."""
        config_path = Path("config/risk_profiles.yaml")
        
        if not config_path.exists():
            logger.warning("Risk profiles YAML not found, using defaults")
            self._create_default_profiles()
            return
        
        try:
            with open(config_path, 'r') as f:
                data = yaml.safe_load(f)
            
            for profile_name, profile_data in data.items():
                self._profiles[profile_name] = RiskProfile(
                    name=profile_name,
                    max_spend_without_approval=profile_data.get("max_spend_without_approval", 100),
                    require_approval_for=profile_data.get("require_approval_for", []),
                    autonomous_actions_allowed=profile_data.get("autonomous_actions_allowed", []),
                    max_daily_transactions=profile_data.get("max_daily_transactions", 100),
                    max_single_transaction=profile_data.get("max_single_transaction", 10000),
                    require_two_factor_for_amount=profile_data.get("require_two_factor_for_amount", 1000),
                    escalation_contacts=profile_data.get("escalation_contacts", []),
                )
            
            logger.info(f"Loaded {len(self._profiles)} risk profiles")
            
        except Exception as e:
            logger.error(f"Failed to load risk profiles: {e}")
            self._create_default_profiles()
    
    def _create_default_profiles(self):
        """Create default risk profiles."""
        self._profiles = {
            "conservative": RiskProfile(
                name="conservative",
                max_spend_without_approval=50,
                require_approval_for=["legal_actions", "financial_transactions", "external_api_calls", "code_modifications"],
                autonomous_actions_allowed=["research", "analysis", "planning"],
            ),
            "moderate": RiskProfile(
                name="moderate",
                max_spend_without_approval=500,
                require_approval_for=["legal_actions", "code_modifications"],
                autonomous_actions_allowed=["research", "analysis", "planning", "content_creation", "small_purchases"],
            ),
            "aggressive": RiskProfile(
                name="aggressive",
                max_spend_without_approval=5000,
                require_approval_for=["legal_actions"],
                autonomous_actions_allowed=["all"],
            ),
        }
    
    def get_profile(self, name: str) -> Optional[RiskProfile]:
        """
        Get a risk profile by name.
        
        Args:
            name: Profile name ('conservative', 'moderate', 'aggressive')
            
        Returns:
            RiskProfile or None if not found
        """
        return self._profiles.get(name)
    
    def get_active_profile(self) -> RiskProfile:
        """
        Get the currently active risk profile.
        
        Returns:
            The active RiskProfile (defaults to 'moderate')
        """
        if self._active_profile is None:
            from config.settings import settings
            profile_name = getattr(settings, 'risk_profile', 'moderate')
            self._active_profile = self._profiles.get(profile_name, self._profiles.get("moderate"))
        
        return self._active_profile
    
    def set_active_profile(self, name: str) -> bool:
        """
        Set the active risk profile.
        
        Args:
            name: Profile name to activate
            
        Returns:
            True if profile was set, False if not found
        """
        profile = self._profiles.get(name)
        if profile:
            self._active_profile = profile
            logger.info(f"Activated risk profile: {name}")
            return True
        return False
    
    def list_profiles(self) -> List[str]:
        """Get list of available profile names."""
        return list(self._profiles.keys())
    
    def check_action(self, action: str, amount: float = 0.0) -> Dict[str, Any]:
        """
        Check if an action is allowed under current risk profile.
        
        Args:
            action: Action type to check
            amount: Monetary amount involved
            
        Returns:
            Dict with 'allowed', 'requires_approval', 'approval_level' keys
        """
        profile = self.get_active_profile()
        
        requires_approval = profile.requires_approval(action, amount)
        is_allowed = profile.is_action_allowed(action)
        approval_level = profile.get_approval_level(amount) if amount > 0 else "auto"
        
        return {
            "allowed": is_allowed or not requires_approval,
            "requires_approval": requires_approval,
            "approval_level": approval_level,
            "max_auto_approve": profile.max_spend_without_approval,
            "profile": profile.name,
        }
    
    def reload(self):
        """Reload profiles from YAML file."""
        self._profiles.clear()
        self._active_profile = None
        self._load_profiles()


# Singleton accessor
def get_risk_manager() -> RiskProfileManager:
    """Get the risk profile manager singleton."""
    return RiskProfileManager()


# Convenience function
def check_risk(action: str, amount: float = 0.0) -> Dict[str, Any]:
    """
    Quick check if an action is allowed under current risk profile.
    
    Args:
        action: Action type
        amount: Monetary amount
        
    Returns:
        Risk check result
    """
    return get_risk_manager().check_action(action, amount)
