"""
Lifecycle Engine - Manages the progression of business units.
Implements the standard King AI transition logic.
"""

from src.database.models import BusinessStatus

class LifecycleEngine:
    """
    Deterministic state machine for business progression.
    """
    
    # Define valid transitions
    TRANSITIONS = {
        BusinessStatus.DISCOVERY: BusinessStatus.VALIDATION,
        BusinessStatus.VALIDATION: BusinessStatus.SETUP,
        BusinessStatus.SETUP: BusinessStatus.OPERATION,
        BusinessStatus.OPERATION: BusinessStatus.OPTIMIZATION,
        BusinessStatus.OPTIMIZATION: BusinessStatus.REPLICATION
    }

    def get_next_status(self, current_status: BusinessStatus) -> BusinessStatus | None:
        """
        Returns the next logical stage for a business unit.
        """
        return self.TRANSITIONS.get(current_status)

    def is_failed(self, status: BusinessStatus) -> bool:
        """Checks if the business has been sunset."""
        return status == BusinessStatus.SUNSET
