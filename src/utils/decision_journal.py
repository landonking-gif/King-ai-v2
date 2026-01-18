"""
Decision Journal.

Tracks important decisions with context, options, reasoning, and outcomes.
Based on mother-harness PKM patterns.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4
import json

from src.utils.structured_logging import get_logger

logger = get_logger("decision_journal")


class DecisionStatus(str, Enum):
    """Status of a decision."""
    PROPOSED = "proposed"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    IMPLEMENTED = "implemented"
    ABANDONED = "abandoned"
    REVISIT = "revisit"


class DecisionCategory(str, Enum):
    """Categories of decisions."""
    TECHNICAL = "technical"
    BUSINESS = "business"
    FINANCIAL = "financial"
    LEGAL = "legal"
    STRATEGIC = "strategic"
    OPERATIONAL = "operational"
    PRODUCT = "product"
    TEAM = "team"


class OutcomeType(str, Enum):
    """Outcome of a decision."""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    NEUTRAL = "neutral"
    PARTIAL_FAILURE = "partial_failure"
    FAILURE = "failure"
    UNKNOWN = "unknown"


@dataclass
class Option:
    """An option considered for a decision."""
    id: str = field(default_factory=lambda: f"opt_{uuid4().hex[:6]}")
    name: str = ""
    description: str = ""
    pros: List[str] = field(default_factory=list)
    cons: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    estimated_cost: Optional[float] = None
    estimated_time_days: Optional[int] = None
    selected: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "pros": self.pros,
            "cons": self.cons,
            "risks": self.risks,
            "estimated_cost": self.estimated_cost,
            "estimated_time_days": self.estimated_time_days,
            "selected": self.selected,
        }


@dataclass
class DecisionOutcome:
    """Outcome of a decision after implementation."""
    outcome_type: OutcomeType = OutcomeType.UNKNOWN
    recorded_at: datetime = field(default_factory=datetime.utcnow)
    description: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    lessons_learned: List[str] = field(default_factory=list)
    what_went_well: List[str] = field(default_factory=list)
    what_went_wrong: List[str] = field(default_factory=list)
    would_do_differently: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "outcome_type": self.outcome_type.value,
            "recorded_at": self.recorded_at.isoformat(),
            "description": self.description,
            "metrics": self.metrics,
            "lessons_learned": self.lessons_learned,
            "what_went_well": self.what_went_well,
            "what_went_wrong": self.what_went_wrong,
            "would_do_differently": self.would_do_differently,
        }


@dataclass
class Decision:
    """A tracked decision."""
    id: str = field(default_factory=lambda: f"dec_{uuid4().hex[:10]}")
    title: str = ""
    description: str = ""
    
    # Classification
    category: DecisionCategory = DecisionCategory.BUSINESS
    status: DecisionStatus = DecisionStatus.PROPOSED
    priority: str = "medium"  # low, medium, high, critical
    
    # Context
    business_id: Optional[str] = None
    agent_id: Optional[str] = None
    workflow_id: Optional[str] = None
    
    # Problem statement
    problem_statement: str = ""
    constraints: List[str] = field(default_factory=list)
    stakeholders: List[str] = field(default_factory=list)
    
    # Options
    options: List[Option] = field(default_factory=list)
    
    # Decision
    selected_option_id: Optional[str] = None
    reasoning: str = ""
    deciding_factors: List[str] = field(default_factory=list)
    trade_offs: List[str] = field(default_factory=list)
    
    # Approval
    decided_by: Optional[str] = None
    decided_at: Optional[datetime] = None
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    
    # Implementation
    implementation_notes: str = ""
    implemented_at: Optional[datetime] = None
    
    # Outcome
    outcome: Optional[DecisionOutcome] = None
    
    # Review
    review_date: Optional[datetime] = None
    review_notes: List[str] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    tags: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)  # Links to related artifacts/docs
    
    @property
    def selected_option(self) -> Optional[Option]:
        """Get the selected option."""
        if self.selected_option_id:
            for opt in self.options:
                if opt.id == self.selected_option_id:
                    return opt
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "category": self.category.value,
            "status": self.status.value,
            "priority": self.priority,
            "business_id": self.business_id,
            "agent_id": self.agent_id,
            "problem_statement": self.problem_statement,
            "constraints": self.constraints,
            "stakeholders": self.stakeholders,
            "options": [opt.to_dict() for opt in self.options],
            "selected_option_id": self.selected_option_id,
            "reasoning": self.reasoning,
            "deciding_factors": self.deciding_factors,
            "trade_offs": self.trade_offs,
            "decided_by": self.decided_by,
            "decided_at": self.decided_at.isoformat() if self.decided_at else None,
            "approved_by": self.approved_by,
            "approved_at": self.approved_at.isoformat() if self.approved_at else None,
            "implementation_notes": self.implementation_notes,
            "implemented_at": self.implemented_at.isoformat() if self.implemented_at else None,
            "outcome": self.outcome.to_dict() if self.outcome else None,
            "review_date": self.review_date.isoformat() if self.review_date else None,
            "review_notes": self.review_notes,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "tags": self.tags,
            "references": self.references,
        }
    
    def to_markdown(self) -> str:
        """Convert to markdown for display."""
        md = [
            f"# Decision: {self.title}",
            f"",
            f"**ID:** {self.id}",
            f"**Status:** {self.status.value}",
            f"**Category:** {self.category.value}",
            f"**Priority:** {self.priority}",
            f"**Created:** {self.created_at.strftime('%Y-%m-%d')}",
            f"",
            f"## Problem Statement",
            f"{self.problem_statement or self.description}",
            f"",
        ]
        
        if self.constraints:
            md.append("## Constraints")
            for c in self.constraints:
                md.append(f"- {c}")
            md.append("")
        
        if self.options:
            md.append("## Options Considered")
            for opt in self.options:
                selected = " ✓" if opt.selected else ""
                md.append(f"### {opt.name}{selected}")
                md.append(opt.description)
                md.append("")
                if opt.pros:
                    md.append("**Pros:**")
                    for p in opt.pros:
                        md.append(f"- {p}")
                if opt.cons:
                    md.append("**Cons:**")
                    for c in opt.cons:
                        md.append(f"- {c}")
                md.append("")
        
        if self.reasoning:
            md.append("## Decision Reasoning")
            md.append(self.reasoning)
            md.append("")
        
        if self.outcome:
            md.append("## Outcome")
            md.append(f"**Result:** {self.outcome.outcome_type.value}")
            md.append(self.outcome.description)
            if self.outcome.lessons_learned:
                md.append("")
                md.append("### Lessons Learned")
                for lesson in self.outcome.lessons_learned:
                    md.append(f"- {lesson}")
        
        return "\n".join(md)


class DecisionJournal:
    """
    Journal for tracking decisions.
    
    Features:
    - Create and track decisions
    - Record options with pros/cons
    - Track outcomes and lessons learned
    - Schedule reviews
    - Query and search
    """
    
    def __init__(
        self,
        storage_path: Optional[str] = None,
    ):
        """
        Initialize the journal.
        
        Args:
            storage_path: Path to store decisions (in-memory if None)
        """
        self._decisions: Dict[str, Decision] = {}
        self._storage_path = storage_path
        
        # Index for quick lookups
        self._by_business: Dict[str, Set[str]] = {}
        self._by_status: Dict[DecisionStatus, Set[str]] = {
            status: set() for status in DecisionStatus
        }
        self._by_category: Dict[DecisionCategory, Set[str]] = {
            cat: set() for cat in DecisionCategory
        }
        self._pending_review: Set[str] = set()
    
    def create_decision(
        self,
        title: str,
        description: str = "",
        category: DecisionCategory = DecisionCategory.BUSINESS,
        business_id: str = None,
        **kwargs,
    ) -> Decision:
        """
        Create a new decision.
        
        Args:
            title: Decision title
            description: Description
            category: Decision category
            business_id: Associated business
            **kwargs: Additional fields
            
        Returns:
            Created decision
        """
        decision = Decision(
            title=title,
            description=description,
            category=category,
            business_id=business_id,
            **kwargs,
        )
        
        self._decisions[decision.id] = decision
        self._index_decision(decision)
        
        logger.info(f"Created decision: {decision.id} - {title}")
        return decision
    
    def get(self, decision_id: str) -> Optional[Decision]:
        """Get a decision by ID."""
        return self._decisions.get(decision_id)
    
    def update(self, decision: Decision) -> None:
        """Update a decision."""
        decision.updated_at = datetime.utcnow()
        self._decisions[decision.id] = decision
        self._index_decision(decision)
    
    def add_option(
        self,
        decision_id: str,
        name: str,
        description: str = "",
        pros: List[str] = None,
        cons: List[str] = None,
        risks: List[str] = None,
        **kwargs,
    ) -> Option:
        """Add an option to a decision."""
        decision = self._decisions.get(decision_id)
        if not decision:
            raise ValueError(f"Decision not found: {decision_id}")
        
        option = Option(
            name=name,
            description=description,
            pros=pros or [],
            cons=cons or [],
            risks=risks or [],
            **kwargs,
        )
        
        decision.options.append(option)
        decision.updated_at = datetime.utcnow()
        
        return option
    
    def select_option(
        self,
        decision_id: str,
        option_id: str,
        reasoning: str,
        deciding_factors: List[str] = None,
        decided_by: str = None,
    ) -> Decision:
        """Select an option for a decision."""
        decision = self._decisions.get(decision_id)
        if not decision:
            raise ValueError(f"Decision not found: {decision_id}")
        
        # Mark all options as not selected
        for opt in decision.options:
            opt.selected = opt.id == option_id
        
        decision.selected_option_id = option_id
        decision.reasoning = reasoning
        decision.deciding_factors = deciding_factors or []
        decision.decided_by = decided_by
        decision.decided_at = datetime.utcnow()
        decision.status = DecisionStatus.APPROVED
        decision.updated_at = datetime.utcnow()
        
        return decision
    
    def record_outcome(
        self,
        decision_id: str,
        outcome_type: OutcomeType,
        description: str = "",
        lessons_learned: List[str] = None,
        **kwargs,
    ) -> Decision:
        """Record the outcome of a decision."""
        decision = self._decisions.get(decision_id)
        if not decision:
            raise ValueError(f"Decision not found: {decision_id}")
        
        decision.outcome = DecisionOutcome(
            outcome_type=outcome_type,
            description=description,
            lessons_learned=lessons_learned or [],
            **kwargs,
        )
        
        decision.status = DecisionStatus.IMPLEMENTED
        decision.implemented_at = datetime.utcnow()
        decision.updated_at = datetime.utcnow()
        
        logger.info(f"Recorded outcome for decision {decision_id}: {outcome_type.value}")
        return decision
    
    def schedule_review(
        self,
        decision_id: str,
        review_date: datetime,
    ) -> None:
        """Schedule a decision for review."""
        decision = self._decisions.get(decision_id)
        if not decision:
            raise ValueError(f"Decision not found: {decision_id}")
        
        decision.review_date = review_date
        self._pending_review.add(decision_id)
    
    def get_due_for_review(self) -> List[Decision]:
        """Get decisions due for review."""
        now = datetime.utcnow()
        due = []
        
        for decision_id in self._pending_review:
            decision = self._decisions.get(decision_id)
            if decision and decision.review_date and decision.review_date <= now:
                due.append(decision)
        
        return due
    
    def add_review_note(
        self,
        decision_id: str,
        note: str,
    ) -> None:
        """Add a review note."""
        decision = self._decisions.get(decision_id)
        if not decision:
            raise ValueError(f"Decision not found: {decision_id}")
        
        decision.review_notes.append(f"[{datetime.utcnow().isoformat()}] {note}")
        decision.updated_at = datetime.utcnow()
    
    def query(
        self,
        business_id: str = None,
        status: DecisionStatus = None,
        category: DecisionCategory = None,
        tags: Set[str] = None,
        since: datetime = None,
        limit: int = 100,
    ) -> List[Decision]:
        """
        Query decisions.
        
        Args:
            business_id: Filter by business
            status: Filter by status
            category: Filter by category
            tags: Filter by tags (any match)
            since: Filter by creation date
            limit: Maximum results
            
        Returns:
            Matching decisions
        """
        # Start with appropriate index
        if business_id:
            decision_ids = self._by_business.get(business_id, set())
        elif status:
            decision_ids = self._by_status.get(status, set())
        elif category:
            decision_ids = self._by_category.get(category, set())
        else:
            decision_ids = set(self._decisions.keys())
        
        decisions = []
        for decision_id in decision_ids:
            decision = self._decisions.get(decision_id)
            if not decision:
                continue
            
            # Apply filters
            if status and decision.status != status:
                continue
            if category and decision.category != category:
                continue
            if business_id and decision.business_id != business_id:
                continue
            if tags and not tags.intersection(set(decision.tags)):
                continue
            if since and decision.created_at < since:
                continue
            
            decisions.append(decision)
        
        # Sort by creation date (newest first)
        decisions.sort(key=lambda d: d.created_at, reverse=True)
        
        return decisions[:limit]
    
    def get_lessons_learned(
        self,
        category: DecisionCategory = None,
        outcome_type: OutcomeType = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Extract lessons learned from past decisions.
        
        Args:
            category: Filter by category
            outcome_type: Filter by outcome
            limit: Maximum lessons
            
        Returns:
            List of lessons with context
        """
        lessons = []
        
        for decision in self._decisions.values():
            if not decision.outcome or not decision.outcome.lessons_learned:
                continue
            
            if category and decision.category != category:
                continue
            if outcome_type and decision.outcome.outcome_type != outcome_type:
                continue
            
            for lesson in decision.outcome.lessons_learned:
                lessons.append({
                    "lesson": lesson,
                    "decision_id": decision.id,
                    "decision_title": decision.title,
                    "category": decision.category.value,
                    "outcome": decision.outcome.outcome_type.value,
                    "date": decision.implemented_at.isoformat() if decision.implemented_at else None,
                })
        
        return lessons[:limit]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get journal statistics."""
        outcomes = {}
        for decision in self._decisions.values():
            if decision.outcome:
                outcome = decision.outcome.outcome_type.value
                outcomes[outcome] = outcomes.get(outcome, 0) + 1
        
        return {
            "total_decisions": len(self._decisions),
            "by_status": {
                status.value: len(ids) 
                for status, ids in self._by_status.items()
            },
            "by_category": {
                cat.value: len(ids) 
                for cat, ids in self._by_category.items()
            },
            "by_outcome": outcomes,
            "pending_review": len(self._pending_review),
        }
    
    def export_to_json(self) -> str:
        """Export all decisions to JSON."""
        return json.dumps(
            [d.to_dict() for d in self._decisions.values()],
            indent=2,
            default=str,
        )
    
    # Private methods
    
    def _index_decision(self, decision: Decision) -> None:
        """Index a decision for quick lookups."""
        decision_id = decision.id
        
        # Remove from old indices
        for ids in self._by_business.values():
            ids.discard(decision_id)
        for ids in self._by_status.values():
            ids.discard(decision_id)
        for ids in self._by_category.values():
            ids.discard(decision_id)
        
        # Add to new indices
        if decision.business_id:
            if decision.business_id not in self._by_business:
                self._by_business[decision.business_id] = set()
            self._by_business[decision.business_id].add(decision_id)
        
        self._by_status[decision.status].add(decision_id)
        self._by_category[decision.category].add(decision_id)
        
        # Track review
        if decision.review_date:
            self._pending_review.add(decision_id)


# Global journal instance
_decision_journal: Optional[DecisionJournal] = None


def get_decision_journal() -> DecisionJournal:
    """Get or create the global decision journal."""
    global _decision_journal
    if _decision_journal is None:
        _decision_journal = DecisionJournal()
    return _decision_journal
