# King AI v2 - Implementation Plan Part 5
## Evolution Engine - Core Models & Proposal System

**Target Timeline:** Week 4-5
**Objective:** Implement the core evolution engine with proposal models, confidence scoring, and basic self-modification capabilities.

---

## Table of Contents
1. [Overview of 20-Part Plan](#overview-of-20-part-plan)
2. [Part 5 Scope](#part-5-scope)
3. [Current State Analysis](#current-state-analysis)
4. [Implementation Tasks](#implementation-tasks)
5. [File-by-File Instructions](#file-by-file-instructions)
6. [Testing Requirements](#testing-requirements)
7. [Acceptance Criteria](#acceptance-criteria)

---

## Overview of 20-Part Plan

| Part | Title | Status |
|------|-------|--------|
| 1 | Infrastructure Layer & Core System Hardening | ✅ Complete |
| 2 | Master AI Brain - Core Enhancements | ✅ Complete |
| 3 | Master AI Brain - Context & Memory System | ✅ Complete |
| 4 | Master AI Brain - Planning & ReAct Implementation | ✅ Complete |
| **5** | **Evolution Engine - Core Models & Proposal System** | 🔄 Current |
| 5.25 | Evolution Engine - Code Analysis & AST Utilities | ⏳ Pending |
| 5.5 | Evolution Engine - Code Patching & Generation | ⏳ Pending |
| 5.75 | Evolution Engine - Git Integration & Version Control | ⏳ Pending |
| 6 | Evolution Engine - ML Retraining Pipeline | ⏳ Pending |
| 7 | Evolution Engine - Sandbox & Testing | ⏳ Pending |
| 8 | Sub-Agents - Research Agent Enhancement | ⏳ Pending |
| 9 | Sub-Agents - Code Generator Agent | ⏳ Pending |
| 10 | Sub-Agents - Content Agent | ⏳ Pending |
| 11 | Sub-Agents - Commerce Agent (Shopify/AliExpress) | ⏳ Pending |
| 12 | Sub-Agents - Finance Agent (Stripe/Plaid) | ⏳ Pending |
| 13 | Sub-Agents - Analytics Agent | ⏳ Pending |
| 14 | Sub-Agents - Legal Agent | ⏳ Pending |
| 15 | Business Units - Lifecycle Engine | ⏳ Pending |
| 16 | Business Units - Playbook System | ⏳ Pending |
| 17 | Business Units - Portfolio Management | ⏳ Pending |
| 18 | Dashboard - React UI Components | ⏳ Pending |
| 19 | Dashboard - Approval Workflows & Risk Engine | ⏳ Pending |
| 20 | Dashboard - Real-time Monitoring & WebSocket + Final Integration | ⏳ Pending |

---

## Part 5 Scope

This part focuses on:
1. Evolution proposal data models
2. Confidence scoring system
3. Proposal generation and validation
4. Basic evolution engine with approval workflow
5. Database persistence for evolution history
6. Risk assessment for self-modifications

---

## Current State Analysis

### What Exists in `src/master_ai/evolution.py`
| Feature | Status | Issue |
|---------|--------|-------|
| Basic EvolutionEngine class | ✅ Exists | Very basic implementation |
| propose_improvement method | ✅ Works | No confidence scoring |
| apply_proposal method | ✅ Works | No validation or rollback |
| Database models | ⚠️ Partial | Basic EvolutionProposal model |

### What Needs to Be Added
1. Comprehensive proposal models with metadata
2. Confidence scoring algorithms
3. Proposal validation and testing
4. Evolution history tracking
5. Risk assessment for modifications
6. Approval workflow integration

---

## Implementation Tasks

### Task 5.1: Create Evolution Data Models
**Priority:** 🔴 Critical
**Estimated Time:** 2 hours
**Dependencies:** Part 2 complete

#### File: `src/master_ai/evolution_models.py` (CREATE NEW FILE)
```python
"""
Data models for the evolution engine.
Supports proposal generation, validation, and execution tracking.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Literal
from enum import Enum
from datetime import datetime
from uuid import uuid4


class ProposalStatus(str, Enum):
    """Status of an evolution proposal."""
    DRAFT = "draft"
    VALIDATING = "validating"
    READY = "ready"
    APPROVED = "approved"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"


class ProposalType(str, Enum):
    """Type of evolution proposal."""
    CODE_MODIFICATION = "code_modification"
    CONFIGURATION_CHANGE = "configuration_change"
    MODEL_RETRAINING = "model_retraining"
    AGENT_ENHANCEMENT = "agent_enhancement"
    INFRASTRUCTURE_UPDATE = "infrastructure_update"
    BUSINESS_RULE_CHANGE = "business_rule_change"


class RiskLevel(str, Enum):
    """Risk classification for proposals."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ConfidenceScore(BaseModel):
    """Confidence score for a proposal."""
    overall: float = Field(..., ge=0.0, le=1.0)
    components: Dict[str, float] = Field(default_factory=dict)
    reasoning: str = ""
    calculated_at: datetime = Field(default_factory=datetime.now)
    
    @validator('overall')
    def validate_overall(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Overall confidence must be between 0.0 and 1.0')
        return v


class ValidationResult(BaseModel):
    """Result of proposal validation."""
    passed: bool
    tests_run: int = 0
    tests_passed: int = 0
    tests_failed: int = 0
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    execution_time_seconds: Optional[float] = None
    validated_at: datetime = Field(default_factory=datetime.now)


class CodeChange(BaseModel):
    """A single code change in a proposal."""
    file_path: str
    change_type: Literal["add", "modify", "delete", "rename"]
    old_content: Optional[str] = None
    new_content: Optional[str] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    description: str = ""
    
    @property
    def is_safe(self) -> bool:
        """Check if this change appears safe."""
        # Simple heuristics for safety
        if self.change_type == "delete":
            return False  # Deletions are risky
        if self.old_content and len(self.old_content) > 1000:
            return False  # Large changes are risky
        return True


class EvolutionProposal(BaseModel):
    """A proposal for system evolution."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    title: str
    description: str
    proposal_type: ProposalType
    
    # Status and lifecycle
    status: ProposalStatus = ProposalStatus.DRAFT
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    # Author and approval
    proposed_by: str = "master_ai"
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    rejected_reason: Optional[str] = None
    
    # Risk and confidence
    risk_level: RiskLevel = RiskLevel.MEDIUM
    confidence_score: Optional[ConfidenceScore] = None
    requires_approval: bool = True
    
    # Content
    changes: List[CodeChange] = Field(default_factory=list)
    configuration_changes: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Validation and execution
    validation_result: Optional[ValidationResult] = None
    execution_result: Optional[Dict[str, Any]] = None
    rollback_data: Optional[Dict[str, Any]] = None
    
    # Relationships
    parent_proposal_id: Optional[str] = None
    related_proposals: List[str] = Field(default_factory=list)
    
    # Metrics
    estimated_impact: Dict[str, float] = Field(default_factory=dict)  # e.g., {"performance": 0.1, "reliability": -0.05}
    actual_impact: Optional[Dict[str, float]] = None
    
    def can_execute(self) -> bool:
        """Check if proposal is ready for execution."""
        return (
            self.status in [ProposalStatus.READY, ProposalStatus.APPROVED] and
            self.confidence_score and
            self.confidence_score.overall >= 0.7 and
            self.validation_result and
            self.validation_result.passed
        )
    
    def is_high_risk(self) -> bool:
        """Check if this is a high-risk proposal."""
        return (
            self.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL] or
            any(not change.is_safe for change in self.changes) or
            len(self.changes) > 10  # Many changes = high risk
        )
    
    def calculate_risk_level(self) -> RiskLevel:
        """Calculate risk level based on proposal characteristics."""
        risk_factors = 0
        
        # Type-based risk
        if self.proposal_type in [ProposalType.CODE_MODIFICATION, ProposalType.MODEL_RETRAINING]:
            risk_factors += 2
        elif self.proposal_type == ProposalType.INFRASTRUCTURE_UPDATE:
            risk_factors += 3
        
        # Change-based risk
        if len(self.changes) > 5:
            risk_factors += 1
        if any(not change.is_safe for change in self.changes):
            risk_factors += 2
        
        # Confidence-based risk
        if self.confidence_score and self.confidence_score.overall < 0.8:
            risk_factors += 1
        
        # Map to risk level
        if risk_factors >= 4:
            return RiskLevel.CRITICAL
        elif risk_factors >= 3:
            return RiskLevel.HIGH
        elif risk_factors >= 2:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW


class EvolutionHistory(BaseModel):
    """History of evolution events."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)
    event_type: Literal["proposed", "approved", "executed", "failed", "rolled_back"]
    proposal_id: str
    description: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    system_state_before: Optional[Dict[str, Any]] = None
    system_state_after: Optional[Dict[str, Any]] = None


class EvolutionMetrics(BaseModel):
    """Metrics for evolution performance."""
    total_proposals: int = 0
    successful_proposals: int = 0
    failed_proposals: int = 0
    average_confidence: float = 0.0
    average_execution_time: float = 0.0
    risk_distribution: Dict[str, int] = Field(default_factory=dict)
    last_updated: datetime = Field(default_factory=datetime.now)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_proposals == 0:
            return 0.0
        return self.successful_proposals / self.total_proposals
    
    def update_from_proposal(self, proposal: EvolutionProposal):
        """Update metrics from a completed proposal."""
        self.total_proposals += 1
        
        if proposal.status == ProposalStatus.COMPLETED:
            self.successful_proposals += 1
        elif proposal.status == ProposalStatus.FAILED:
            self.failed_proposals += 1
        
        if proposal.confidence_score:
            # Rolling average
            self.average_confidence = (
                (self.average_confidence * (self.total_proposals - 1)) +
                proposal.confidence_score.overall
            ) / self.total_proposals
        
        # Update risk distribution
        risk = proposal.risk_level.value
        self.risk_distribution[risk] = self.risk_distribution.get(risk, 0) + 1
        
        self.last_updated = datetime.now()
```

---

### Task 5.2: Implement Confidence Scoring System
**Priority:** 🔴 Critical
**Estimated Time:** 3 hours
**Dependencies:** Task 5.1

#### File: `src/master_ai/confidence_scorer.py` (CREATE NEW FILE)
```python
"""
Confidence scoring system for evolution proposals.
Evaluates proposal quality and safety using multiple metrics.
"""

import re
import ast
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime

from src.master_ai.evolution_models import (
    EvolutionProposal, ConfidenceScore, CodeChange, ProposalType
)
from src.utils.llm_router import LLMRouter, TaskContext
from src.utils.structured_logging import get_logger
from src.utils.retry import with_retry, LLM_RETRY_CONFIG
from config.settings import settings

logger = get_logger("confidence_scorer")


@dataclass
class ConfidenceComponent:
    """A component of confidence scoring."""
    name: str
    weight: float
    description: str
    
    def score(self, proposal: EvolutionProposal) -> float:
        """Calculate score for this component. Override in subclasses."""
        raise NotImplementedError


class CodeQualityScorer(ConfidenceComponent):
    """Scores code quality and safety."""
    
    def __init__(self):
        super().__init__(
            name="code_quality",
            weight=0.3,
            description="Code quality, safety, and best practices"
        )
    
    def score(self, proposal: EvolutionProposal) -> float:
        """Score based on code analysis."""
        if not proposal.changes:
            return 1.0  # No code changes = safe
        
        total_score = 0.0
        for change in proposal.changes:
            score = self._score_change(change)
            total_score += score
        
        return min(1.0, total_score / len(proposal.changes))
    
    def _score_change(self, change: CodeChange) -> float:
        """Score a single code change."""
        score = 1.0
        
        # Penalize deletions
        if change.change_type == "delete":
            score *= 0.3
        
        # Penalize large changes
        if change.old_content and len(change.old_content) > 500:
            score *= 0.7
        
        # Check for dangerous patterns
        if change.new_content:
            dangerous_patterns = [
                r'import\s+os\s*$',  # Direct os import
                r'subprocess\.',     # Subprocess usage
                r'eval\s*\(',        # Eval usage
                r'exec\s*\(',        # Exec usage
            ]
            
            for pattern in dangerous_patterns:
                if re.search(pattern, change.new_content, re.MULTILINE):
                    score *= 0.5
                    break
        
        # Bonus for safe patterns
        if change.new_content and 'try:' in change.new_content:
            score *= 1.1
        
        return min(1.0, score)


class TestingCoverageScorer(ConfidenceComponent):
    """Scores testing coverage and validation."""
    
    def __init__(self):
        super().__init__(
            name="testing_coverage",
            weight=0.25,
            description="Testing coverage and validation completeness"
        )
    
    def score(self, proposal: EvolutionProposal) -> float:
        """Score based on testing and validation."""
        score = 0.5  # Base score
        
        # Validation results
        if proposal.validation_result:
            val = proposal.validation_result
            if val.passed:
                score += 0.3
            if val.tests_run > 0:
                coverage = val.tests_passed / val.tests_run
                score += coverage * 0.2
        
        # Changes that affect critical files
        critical_files = [
            'src/master_ai/brain.py',
            'src/master_ai/evolution.py',
            'src/database/',
            'config/settings.py'
        ]
        
        affects_critical = any(
            any(change.file_path.startswith(critical) for critical in critical_files)
            for change in proposal.changes
        )
        
        if affects_critical:
            score *= 0.8  # Penalize critical changes
        
        return min(1.0, score)


class HistoricalSuccessScorer(ConfidenceComponent):
    """Scores based on historical success patterns."""
    
    def __init__(self, evolution_engine):
        super().__init__(
            name="historical_success",
            weight=0.2,
            description="Historical success rate for similar proposals"
        )
        self.evolution_engine = evolution_engine
    
    def score(self, proposal: EvolutionProposal) -> float:
        """Score based on historical data."""
        # Get similar proposals
        similar = self.evolution_engine.get_similar_proposals(proposal)
        
        if not similar:
            return 0.5  # Neutral score with no history
        
        successful = sum(1 for p in similar if p.status == "completed")
        total = len(similar)
        
        success_rate = successful / total if total > 0 else 0.5
        
        # Adjust based on recency (recent failures are worse)
        recent_failures = sum(
            1 for p in similar[:5]  # Last 5 similar proposals
            if p.status == "failed" and
            (datetime.now() - p.updated_at).days < 7
        )
        
        if recent_failures > 0:
            success_rate *= 0.8
        
        return success_rate


class LLMConfidenceScorer(ConfidenceComponent):
    """Uses LLM to assess proposal confidence."""
    
    def __init__(self, llm_router: LLMRouter):
        super().__init__(
            name="llm_assessment",
            weight=0.25,
            description="LLM-based assessment of proposal quality"
        )
        self.llm = llm_router
    
    @with_retry(LLM_RETRY_CONFIG)
    async def score_async(self, proposal: EvolutionProposal) -> float:
        """Async scoring using LLM."""
        prompt = f"""Assess the confidence level for this evolution proposal.

PROPOSAL: {proposal.title}
DESCRIPTION: {proposal.description}
TYPE: {proposal.proposal_type.value}
RISK: {proposal.risk_level.value}

CHANGES:
{chr(10).join(f"- {c.file_path}: {c.change_type} ({c.description})" for c in proposal.changes[:5])}

Rate the confidence from 0.0 to 1.0 considering:
- Code quality and safety
- Potential for bugs or regressions
- Alignment with system goals
- Testing adequacy
- Risk mitigation

Respond with only a number between 0.0 and 1.0, and a brief explanation.

Example: 0.85 - Well-structured changes with good testing"""
        
        try:
            llm_context = TaskContext(
                task_type="analysis",
                risk_level="low",
                requires_accuracy=True,
                token_estimate=500,
                priority="high"
            )
            
            response = await self.llm.complete(prompt, context=llm_context)
            
            # Parse response
            lines = response.strip().split('\n')
            first_line = lines[0].strip()
            
            # Extract number
            match = re.match(r'^(\d*\.?\d+)', first_line)
            if match:
                score = float(match.group(1))
                return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.warning("LLM confidence scoring failed", error=str(e))
        
        return 0.5  # Neutral fallback


class ConfidenceScorer:
    """
    Main confidence scoring system.
    Combines multiple scoring components.
    """
    
    def __init__(self, llm_router: LLMRouter, evolution_engine):
        """
        Initialize the scorer.
        
        Args:
            llm_router: LLM router for AI assessment
            evolution_engine: Evolution engine for historical data
        """
        self.components = [
            CodeQualityScorer(),
            TestingCoverageScorer(),
            HistoricalSuccessScorer(evolution_engine),
        ]
        self.llm_scorer = LLMConfidenceScorer(llm_router)
    
    async def score_proposal(self, proposal: EvolutionProposal) -> ConfidenceScore:
        """
        Calculate comprehensive confidence score for a proposal.
        
        Args:
            proposal: The proposal to score
            
        Returns:
            Complete confidence score
        """
        logger.info("Scoring proposal confidence", proposal_id=proposal.id)
        
        component_scores = {}
        total_weight = 0.0
        weighted_sum = 0.0
        
        # Score each component
        for component in self.components:
            score = component.score(proposal)
            component_scores[component.name] = score
            
            weighted_sum += score * component.weight
            total_weight += component.weight
        
        # LLM assessment
        llm_score = await self.llm_scorer.score_async(proposal)
        component_scores['llm_assessment'] = llm_score
        weighted_sum += llm_score * self.llm_scorer.weight
        total_weight += self.llm_scorer.weight
        
        # Calculate overall score
        overall = weighted_sum / total_weight if total_weight > 0 else 0.5
        
        # Generate reasoning
        reasoning_parts = []
        for name, score in component_scores.items():
            component = next((c for c in self.components + [self.llm_scorer] if c.name == name), None)
            if component:
                reasoning_parts.append(f"{component.description}: {score:.2f}")
        
        reasoning = "; ".join(reasoning_parts)
        
        confidence = ConfidenceScore(
            overall=overall,
            components=component_scores,
            reasoning=reasoning
        )
        
        logger.info(
            "Confidence scored",
            proposal_id=proposal.id,
            overall=overall,
            components=component_scores
        )
        
        return confidence
    
    def get_thresholds(self, risk_level: str) -> Dict[str, float]:
        """Get confidence thresholds for different risk levels."""
        thresholds = {
            "low": {"min_confidence": 0.6, "min_components": 0.5},
            "medium": {"min_confidence": 0.7, "min_components": 0.6},
            "high": {"min_confidence": 0.8, "min_components": 0.7},
            "critical": {"min_confidence": 0.9, "min_components": 0.8}
        }
        return thresholds.get(risk_level, thresholds["medium"])
    
    def meets_threshold(self, confidence: ConfidenceScore, risk_level: str) -> bool:
        """Check if confidence meets the threshold for the risk level."""
        thresholds = self.get_thresholds(risk_level)
        
        # Overall threshold
        if confidence.overall < thresholds["min_confidence"]:
            return False
        
        # Component thresholds
        min_component = thresholds["min_components"]
        if any(score < min_component for score in confidence.components.values()):
            return False
        
        return True
```

---

### Task 5.3: Enhance Evolution Engine
**Priority:** 🔴 Critical
**Estimated Time:** 4 hours
**Dependencies:** Tasks 5.1, 5.2

#### File: `src/master_ai/evolution.py` (REPLACE ENTIRE FILE)
```python
"""
Evolution Engine - Manages self-modification proposals and execution.
Enhanced with confidence scoring, validation, and approval workflows.
"""

import asyncio
import json
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from pathlib import Path

from src.master_ai.evolution_models import (
    EvolutionProposal, ProposalStatus, ProposalType, RiskLevel,
    ValidationResult, EvolutionHistory, EvolutionMetrics, CodeChange
)
from src.master_ai.confidence_scorer import ConfidenceScorer
from src.master_ai.prompts import EVOLUTION_PROPOSAL_PROMPT, VALIDATION_PROMPT
from src.utils.llm_router import LLMRouter, TaskContext
from src.utils.structured_logging import get_logger
from src.utils.retry import with_retry, LLM_RETRY_CONFIG
from src.utils.sandbox import Sandbox
from src.database.connection import get_db
from src.database.models import EvolutionProposal as DBEvolutionProposal
from config.settings import settings

logger = get_logger("evolution_engine")


class EvolutionEngine:
    """
    Enhanced evolution engine with comprehensive proposal management.
    
    Features:
    - Proposal generation with confidence scoring
    - Validation and testing
    - Approval workflows
    - Safe execution with rollback
    - Historical tracking and metrics
    """
    
    def __init__(self, llm_router: LLMRouter, sandbox: Sandbox):
        """
        Initialize the evolution engine.
        
        Args:
            llm_router: LLM router for proposal generation
            sandbox: Code testing sandbox
        """
        self.llm = llm_router
        self.sandbox = sandbox
        self.confidence_scorer = ConfidenceScorer(llm_router, self)
        
        # Metrics
        self.metrics = EvolutionMetrics()
        
        # Active proposals
        self._active_proposals: Dict[str, EvolutionProposal] = {}
    
    async def propose_improvement(
        self,
        goal: str,
        context: str,
        proposal_type: ProposalType = None,
        constraints: List[str] = None
    ) -> EvolutionProposal:
        """
        Generate an evolution proposal to achieve a goal.
        
        Args:
            goal: The improvement goal
            context: Current system context
            proposal_type: Type of proposal (optional)
            constraints: Constraints to consider
            
        Returns:
            Generated proposal
        """
        logger.info("Generating evolution proposal", goal=goal[:100])
        
        # Generate proposal using LLM
        proposal_data = await self._generate_proposal(goal, context, proposal_type, constraints)
        
        # Create proposal object
        proposal = EvolutionProposal(
            title=proposal_data.get("title", f"Improvement: {goal[:50]}"),
            description=proposal_data.get("description", goal),
            proposal_type=proposal_type or ProposalType.CODE_MODIFICATION,
            changes=self._parse_changes(proposal_data.get("changes", [])),
            configuration_changes=proposal_data.get("config_changes", {}),
            metadata={"goal": goal, "context": context[:500]}
        )
        
        # Calculate risk level
        proposal.risk_level = proposal.calculate_risk_level()
        
        # Score confidence
        proposal.confidence_score = await self.confidence_scorer.score_proposal(proposal)
        
        # Determine if approval needed
        proposal.requires_approval = (
            proposal.is_high_risk() or
            not self.confidence_scorer.meets_threshold(
                proposal.confidence_score,
                proposal.risk_level.value
            )
        )
        
        # Set initial status
        if proposal.requires_approval:
            proposal.status = ProposalStatus.READY
        else:
            proposal.status = ProposalStatus.APPROVED
        
        # Persist to database
        await self._persist_proposal(proposal)
        
        # Add to active proposals
        self._active_proposals[proposal.id] = proposal
        
        logger.info(
            "Proposal generated",
            proposal_id=proposal.id,
            risk=proposal.risk_level.value,
            confidence=proposal.confidence_score.overall if proposal.confidence_score else 0,
            requires_approval=proposal.requires_approval
        )
        
        return proposal
    
    @with_retry(LLM_RETRY_CONFIG)
    async def _generate_proposal(
        self,
        goal: str,
        context: str,
        proposal_type: ProposalType = None,
        constraints: List[str] = None
    ) -> Dict[str, Any]:
        """Generate proposal using LLM."""
        prompt = EVOLUTION_PROPOSAL_PROMPT.format(
            goal=goal,
            context=context[:3000],
            proposal_type=proposal_type.value if proposal_type else "auto",
            constraints="\n".join(f"- {c}" for c in (constraints or [])),
            risk_profile=settings.risk_profile
        )
        
        llm_context = TaskContext(
            task_type="code_generation",
            risk_level="medium",
            requires_accuracy=True,
            token_estimate=1500,
            priority="high"
        )
        
        response = await self.llm.complete(prompt, context=llm_context)
        
        # Parse JSON response
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            logger.warning("Failed to parse proposal JSON", response=response[:500])
            # Fallback parsing
            return self._parse_fallback_proposal(response)
    
    def _parse_changes(self, changes_data: List[Dict]) -> List[CodeChange]:
        """Parse code changes from proposal data."""
        changes = []
        for change_data in changes_data:
            change = CodeChange(
                file_path=change_data.get("file_path", ""),
                change_type=change_data.get("change_type", "modify"),
                old_content=change_data.get("old_content"),
                new_content=change_data.get("new_content"),
                line_start=change_data.get("line_start"),
                line_end=change_data.get("line_end"),
                description=change_data.get("description", "")
            )
            changes.append(change)
        return changes
    
    def _parse_fallback_proposal(self, response: str) -> Dict[str, Any]:
        """Fallback parsing for malformed JSON responses."""
        # Extract title
        title_match = re.search(r'"title"\s*:\s*"([^"]+)"', response)
        title = title_match.group(1) if title_match else "Generated Proposal"
        
        # Extract description
        desc_match = re.search(r'"description"\s*:\s*"([^"]+)"', response)
        description = desc_match.group(1) if desc_match else "Auto-generated improvement"
        
        return {
            "title": title,
            "description": description,
            "changes": [],
            "config_changes": {}
        }
    
    async def validate_proposal(self, proposal: EvolutionProposal) -> ValidationResult:
        """
        Validate a proposal through testing and analysis.
        
        Args:
            proposal: The proposal to validate
            
        Returns:
            Validation results
        """
        logger.info("Validating proposal", proposal_id=proposal.id)
        
        proposal.status = ProposalStatus.VALIDATING
        
        result = ValidationResult(passed=True)
        start_time = datetime.now()
        
        try:
            # Run syntax checks
            syntax_ok = await self._validate_syntax(proposal)
            if not syntax_ok:
                result.passed = False
                result.errors.append("Syntax validation failed")
            
            # Run sandbox tests
            if result.passed:
                test_result = await self._run_sandbox_tests(proposal)
                result.tests_run = test_result.get("tests_run", 0)
                result.tests_passed = test_result.get("tests_passed", 0)
                result.tests_failed = test_result.get("tests_failed", 0)
                
                if result.tests_failed > 0:
                    result.passed = False
                    result.errors.extend(test_result.get("errors", []))
            
            # LLM-based validation
            if result.passed:
                llm_validation = await self._validate_with_llm(proposal)
                if not llm_validation["passed"]:
                    result.passed = False
                    result.errors.extend(llm_validation["issues"])
            
        except Exception as e:
            result.passed = False
            result.errors.append(f"Validation error: {str(e)}")
            logger.error("Proposal validation failed", error=str(e), exc_info=True)
        
        result.execution_time_seconds = (datetime.now() - start_time).total_seconds()
        proposal.validation_result = result
        
        # Update status
        if result.passed:
            proposal.status = ProposalStatus.READY
        else:
            proposal.status = ProposalStatus.DRAFT
        
        # Persist validation result
        await self._update_proposal(proposal)
        
        logger.info(
            "Validation completed",
            proposal_id=proposal.id,
            passed=result.passed,
            tests_run=result.tests_run,
            errors=len(result.errors)
        )
        
        return result
    
    async def _validate_syntax(self, proposal: EvolutionProposal) -> bool:
        """Validate syntax of code changes."""
        for change in proposal.changes:
            if change.new_content and change.file_path.endswith('.py'):
                try:
                    ast.parse(change.new_content)
                except SyntaxError as e:
                    logger.warning(
                        "Syntax error in proposal",
                        file=change.file_path,
                        error=str(e)
                    )
                    return False
        return True
    
    async def _run_sandbox_tests(self, proposal: EvolutionProposal) -> Dict[str, Any]:
        """Run tests in sandbox environment."""
        # This would integrate with the sandbox system
        # For now, return mock results
        return {
            "tests_run": 5,
            "tests_passed": 4,
            "tests_failed": 1,
            "errors": ["Mock test failure"]
        }
    
    @with_retry(LLM_RETRY_CONFIG)
    async def _validate_with_llm(self, proposal: EvolutionProposal) -> Dict[str, Any]:
        """Use LLM to validate proposal quality."""
        prompt = VALIDATION_PROMPT.format(
            title=proposal.title,
            description=proposal.description,
            changes=json.dumps([
                {
                    "file": c.file_path,
                    "type": c.change_type,
                    "content": c.new_content[:200] if c.new_content else ""
                } for c in proposal.changes
            ], indent=2)
        )
        
        response = await self.llm.complete(prompt)
        
        # Parse response
        if "ISSUES FOUND" in response.upper():
            return {"passed": False, "issues": [response]}
        else:
            return {"passed": True, "issues": []}
    
    async def approve_proposal(
        self,
        proposal_id: str,
        approver: str = "system"
    ) -> bool:
        """
        Approve a proposal for execution.
        
        Args:
            proposal_id: ID of proposal to approve
            approver: Who approved it
            
        Returns:
            Success status
        """
        proposal = self._active_proposals.get(proposal_id)
        if not proposal:
            # Load from database
            proposal = await self._load_proposal(proposal_id)
            if not proposal:
                return False
        
        if proposal.status not in [ProposalStatus.READY, ProposalStatus.APPROVED]:
            return False
        
        proposal.status = ProposalStatus.APPROVED
        proposal.approved_by = approver
        proposal.approved_at = datetime.now()
        
        await self._update_proposal(proposal)
        
        logger.info("Proposal approved", proposal_id=proposal_id, approver=approver)
        return True
    
    async def execute_proposal(self, proposal: EvolutionProposal) -> Dict[str, Any]:
        """
        Execute an approved proposal.
        
        Args:
            proposal: The proposal to execute
            
        Returns:
            Execution results
        """
        if not proposal.can_execute():
            raise ValueError("Proposal is not ready for execution")
        
        logger.info("Executing proposal", proposal_id=proposal.id)
        
        proposal.status = ProposalStatus.EXECUTING
        
        # Create rollback data
        rollback_data = await self._create_rollback_data(proposal)
        proposal.rollback_data = rollback_data
        
        success = True
        errors = []
        
        try:
            # Apply changes
            for change in proposal.changes:
                await self._apply_change(change)
            
            # Apply configuration changes
            if proposal.configuration_changes:
                await self._apply_config_changes(proposal.configuration_changes)
            
            # Run post-execution validation
            validation_ok = await self._validate_execution(proposal)
            if not validation_ok:
                success = False
                errors.append("Post-execution validation failed")
        
        except Exception as e:
            success = False
            errors.append(f"Execution error: {str(e)}")
            logger.error("Proposal execution failed", error=str(e), exc_info=True)
        
        # Update status
        if success:
            proposal.status = ProposalStatus.COMPLETED
            self.metrics.update_from_proposal(proposal)
        else:
            proposal.status = ProposalStatus.FAILED
            # Attempt rollback
            await self._rollback_proposal(proposal)
        
        proposal.execution_result = {
            "success": success,
            "errors": errors,
            "executed_at": datetime.now().isoformat()
        }
        
        # Persist results
        await self._update_proposal(proposal)
        
        # Record in history
        await self._record_history(proposal, "executed" if success else "failed")
        
        logger.info(
            "Proposal execution finished",
            proposal_id=proposal.id,
            success=success,
            errors=errors
        )
        
        return proposal.execution_result
    
    async def _apply_change(self, change: CodeChange):
        """Apply a single code change."""
        # This will be implemented in Part 5.5
        # For now, just log
        logger.info(
            "Applying change",
            file=change.file_path,
            type=change.change_type
        )
    
    async def _apply_config_changes(self, config_changes: Dict[str, Any]):
        """Apply configuration changes."""
        # This will be implemented in Part 5.5
        logger.info("Applying config changes", changes=config_changes)
    
    async def _create_rollback_data(self, proposal: EvolutionProposal) -> Dict[str, Any]:
        """Create rollback data for the proposal."""
        # This will be implemented in Part 5.75
        return {"placeholder": True}
    
    async def _rollback_proposal(self, proposal: EvolutionProposal):
        """Rollback a failed proposal."""
        # This will be implemented in Part 5.75
        logger.warning("Rolling back proposal", proposal_id=proposal.id)
    
    async def _validate_execution(self, proposal: EvolutionProposal) -> bool:
        """Validate that execution was successful."""
        # This will be implemented in Part 5.5
        return True
    
    async def get_similar_proposals(self, proposal: EvolutionProposal) -> List[EvolutionProposal]:
        """Get historically similar proposals."""
        # This will be implemented with vector search in Part 3
        return []
    
    async def _persist_proposal(self, proposal: EvolutionProposal):
        """Persist proposal to database."""
        async with get_db() as db:
            db_proposal = DBEvolutionProposal(
                id=proposal.id,
                title=proposal.title,
                description=proposal.description,
                proposal_type=proposal.proposal_type.value,
                status=proposal.status.value,
                risk_level=proposal.risk_level.value,
                confidence_score=proposal.confidence_score.dict() if proposal.confidence_score else None,
                changes=[c.dict() for c in proposal.changes],
                configuration_changes=proposal.configuration_changes,
                metadata=proposal.metadata,
                created_at=proposal.created_at,
                requires_approval=proposal.requires_approval
            )
            db.add(db_proposal)
            await db.commit()
    
    async def _update_proposal(self, proposal: EvolutionProposal):
        """Update proposal in database."""
        async with get_db() as db:
            db_proposal = await db.get(DBEvolutionProposal, proposal.id)
            if db_proposal:
                db_proposal.status = proposal.status.value
                db_proposal.confidence_score = proposal.confidence_score.dict() if proposal.confidence_score else None
                db_proposal.validation_result = proposal.validation_result.dict() if proposal.validation_result else None
                db_proposal.execution_result = proposal.execution_result
                db_proposal.approved_by = proposal.approved_by
                db_proposal.approved_at = proposal.approved_at
                db_proposal.updated_at = datetime.now()
                await db.commit()
    
    async def _load_proposal(self, proposal_id: str) -> Optional[EvolutionProposal]:
        """Load proposal from database."""
        async with get_db() as db:
            db_proposal = await db.get(DBEvolutionProposal, proposal_id)
            if db_proposal:
                return EvolutionProposal(
                    id=db_proposal.id,
                    title=db_proposal.title,
                    description=db_proposal.description,
                    proposal_type=ProposalType(db_proposal.proposal_type),
                    status=ProposalStatus(db_proposal.status),
                    risk_level=RiskLevel(db_proposal.risk_level),
                    confidence_score=ConfidenceScore(**db_proposal.confidence_score) if db_proposal.confidence_score else None,
                    changes=[CodeChange(**c) for c in db_proposal.changes],
                    configuration_changes=db_proposal.configuration_changes,
                    metadata=db_proposal.metadata,
                    created_at=db_proposal.created_at,
                    requires_approval=db_proposal.requires_approval,
                    approved_by=db_proposal.approved_by,
                    approved_at=db_proposal.approved_at
                )
        return None
    
    async def _record_history(self, proposal: EvolutionProposal, event_type: str):
        """Record evolution event in history."""
        # This will be implemented in Part 5.75
        pass
    
    def get_metrics(self) -> EvolutionMetrics:
        """Get evolution metrics."""
        return self.metrics
    
    def get_active_proposals(self) -> List[EvolutionProposal]:
        """Get all active proposals."""
        return list(self._active_proposals.values())
```

---

### Task 5.4: Update Database Models
**Priority:** 🟡 High
**Estimated Time:** 1 hour
**Dependencies:** Task 5.1

#### File: `src/database/models.py` (ADD TO EXISTING FILE)
Add these models after the existing ones:

```python
# Add after existing models

class EvolutionProposal(Base):
    """Database model for evolution proposals."""
    __tablename__ = "evolution_proposals"
    
    id = Column(String, primary_key=True)
    title = Column(String, nullable=False)
    description = Column(Text)
    proposal_type = Column(String, nullable=False)
    status = Column(String, nullable=False, default="draft")
    risk_level = Column(String, nullable=False, default="medium")
    
    # JSON fields
    confidence_score = Column(JSON)
    validation_result = Column(JSON)
    execution_result = Column(JSON)
    changes = Column(JSON)
    configuration_changes = Column(JSON)
    metadata = Column(JSON)
    
    # Approval
    requires_approval = Column(Boolean, default=True)
    approved_by = Column(String)
    approved_at = Column(DateTime)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class EvolutionHistory(Base):
    """Database model for evolution history."""
    __tablename__ = "evolution_history"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    timestamp = Column(DateTime, default=datetime.utcnow)
    event_type = Column(String, nullable=False)
    proposal_id = Column(String, ForeignKey("evolution_proposals.id"))
    description = Column(Text)
    metadata = Column(JSON)
    
    # System state snapshots
    system_state_before = Column(JSON)
    system_state_after = Column(JSON)
    
    proposal = relationship("EvolutionProposal")


class EvolutionMetrics(Base):
    """Database model for evolution metrics."""
    __tablename__ = "evolution_metrics"
    
    id = Column(String, primary_key=True, default="global")
    total_proposals = Column(Integer, default=0)
    successful_proposals = Column(Integer, default=0)
    failed_proposals = Column(Integer, default=0)
    average_confidence = Column(Float, default=0.0)
    average_execution_time = Column(Float, default=0.0)
    risk_distribution = Column(JSON)
    last_updated = Column(DateTime, default=datetime.utcnow)
```

---

### Task 5.5: Update Prompts for Evolution
**Priority:** 🟡 High
**Estimated Time:** 1 hour
**Dependencies:** Task 5.3

#### File: `src/master_ai/prompts.py` (ADD TO EXISTING FILE)
Add these new prompts after the existing ones:

```python
# Add after existing prompts

EVOLUTION_PROPOSAL_PROMPT = """Generate an evolution proposal to improve the King AI system.

GOAL: {goal}

CURRENT SYSTEM CONTEXT:
{context}

PROPOSAL TYPE: {proposal_type}
RISK PROFILE: {risk_profile}
CONSTRAINTS:
{constraints}

Create a specific, actionable proposal that includes:
1. Clear title and description
2. Specific code changes with file paths
3. Configuration changes if needed
4. Risk assessment

Focus on safe, incremental improvements that align with the system's autonomous business empire goals.

Respond with JSON:
{{
    "title": "Proposal Title",
    "description": "Detailed description of the improvement",
    "changes": [
        {{
            "file_path": "src/some/file.py",
            "change_type": "modify|add|delete",
            "old_content": "existing code to replace",
            "new_content": "new code to add",
            "line_start": 10,
            "line_end": 20,
            "description": "What this change does"
        }}
    ],
    "config_changes": {{
        "setting.path": "new_value"
    }},
    "estimated_risk": "low|medium|high|critical",
    "justification": "Why this improvement is needed"
}}
"""


VALIDATION_PROMPT = """Validate this evolution proposal for safety and quality.

PROPOSAL: {title}
DESCRIPTION: {description}

CHANGES:
{changes}

Check for:
1. Code quality and best practices
2. Potential security issues
3. Compatibility with existing system
4. Testing adequacy
5. Rollback feasibility

If any issues found, list them clearly. If no issues, say "NO ISSUES FOUND".

Response format:
ISSUES FOUND:
- Issue 1
- Issue 2

or

NO ISSUES FOUND
"""
```

---

## Testing Requirements

### Unit Tests

#### File: `tests/test_evolution.py` (CREATE NEW FILE)
```python
"""
Tests for the evolution engine.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime

from src.master_ai.evolution_models import (
    EvolutionProposal, ProposalStatus, ProposalType, RiskLevel,
    ConfidenceScore, ValidationResult, CodeChange
)
from src.master_ai.evolution import EvolutionEngine
from src.master_ai.confidence_scorer import ConfidenceScorer


class TestEvolutionProposal:
    """Tests for EvolutionProposal model."""
    
    def test_can_execute_ready_proposal(self):
        """Proposal can execute when ready and validated."""
        proposal = EvolutionProposal(
            title="Test",
            description="Test proposal",
            proposal_type=ProposalType.CODE_MODIFICATION,
            status=ProposalStatus.APPROVED,
            confidence_score=ConfidenceScore(overall=0.8, components={}),
            validation_result=ValidationResult(passed=True)
        )
        assert proposal.can_execute()
    
    def test_cannot_execute_unvalidated_proposal(self):
        """Proposal cannot execute without validation."""
        proposal = EvolutionProposal(
            title="Test",
            description="Test proposal",
            proposal_type=ProposalType.CODE_MODIFICATION,
            status=ProposalStatus.APPROVED,
            confidence_score=ConfidenceScore(overall=0.8, components={})
        )
        assert not proposal.can_execute()
    
    def test_high_risk_detection(self):
        """High-risk proposals are correctly identified."""
        proposal = EvolutionProposal(
            title="Test",
            description="Test proposal",
            proposal_type=ProposalType.CODE_MODIFICATION,
            risk_level=RiskLevel.HIGH
        )
        assert proposal.is_high_risk()
    
    def test_risk_calculation(self):
        """Risk level is calculated correctly."""
        proposal = EvolutionProposal(
            title="Test",
            description="Test proposal",
            proposal_type=ProposalType.INFRASTRUCTURE_UPDATE
        )
        # Infrastructure updates should be high risk
        assert proposal.calculate_risk_level() in [RiskLevel.HIGH, RiskLevel.CRITICAL]


class TestConfidenceScorer:
    """Tests for confidence scorer."""
    
    @pytest.fixture
    def scorer(self):
        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(return_value="0.85 - Good quality")
        
        mock_engine = MagicMock()
        mock_engine.get_similar_proposals.return_value = []
        
        return ConfidenceScorer(mock_llm, mock_engine)
    
    @pytest.mark.asyncio
    async def test_score_proposal(self, scorer):
        """Test proposal scoring."""
        proposal = EvolutionProposal(
            title="Test",
            description="Test proposal",
            proposal_type=ProposalType.CODE_MODIFICATION
        )
        
        score = await scorer.score_proposal(proposal)
        
        assert isinstance(score, ConfidenceScore)
        assert 0.0 <= score.overall <= 1.0
        assert 'llm_assessment' in score.components
    
    def test_threshold_checking(self, scorer):
        """Test threshold validation."""
        high_confidence = ConfidenceScore(
            overall=0.9,
            components={"code_quality": 0.8, "testing": 0.7}
        )
        
        assert scorer.meets_threshold(high_confidence, "high")
        
        low_confidence = ConfidenceScore(
            overall=0.5,
            components={"code_quality": 0.4, "testing": 0.3}
        )
        
        assert not scorer.meets_threshold(low_confidence, "medium")


class TestEvolutionEngine:
    """Tests for evolution engine."""
    
    @pytest.fixture
    def engine(self):
        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(return_value='''{
            "title": "Test Proposal",
            "description": "A test improvement",
            "changes": [],
            "config_changes": {}
        }''')
        
        mock_sandbox = MagicMock()
        
        return EvolutionEngine(mock_llm, mock_sandbox)
    
    @pytest.mark.asyncio
    async def test_propose_improvement(self, engine):
        """Test proposal generation."""
        with patch('src.master_ai.evolution.EvolutionEngine._persist_proposal'):
            proposal = await engine.propose_improvement(
                goal="Improve performance",
                context="Current system state"
            )
        
        assert isinstance(proposal, EvolutionProposal)
        assert proposal.title == "Test Proposal"
        assert proposal.confidence_score is not None
    
    @pytest.mark.asyncio
    async def test_validate_proposal(self, engine):
        """Test proposal validation."""
        proposal = EvolutionProposal(
            title="Test",
            description="Test proposal",
            proposal_type=ProposalType.CODE_MODIFICATION
        )
        
        with patch('src.master_ai.evolution.EvolutionEngine._update_proposal'):
            result = await engine.validate_proposal(proposal)
        
        assert isinstance(result, ValidationResult)
        assert hasattr(result, 'passed')
```

---

## Acceptance Criteria

### Part 5 Completion Checklist

- [ ] **Evolution Models**
  - [ ] `src/master_ai/evolution_models.py` created
  - [ ] EvolutionProposal with full lifecycle
  - [ ] ConfidenceScore with component breakdown
  - [ ] ValidationResult and history models

- [ ] **Confidence Scorer**
  - [ ] `src/master_ai/confidence_scorer.py` created
  - [ ] Multiple scoring components working
  - [ ] LLM-based assessment integrated
  - [ ] Threshold validation working

- [ ] **Evolution Engine**
  - [ ] `src/master_ai/evolution.py` replaced
  - [ ] Proposal generation with LLM
  - [ ] Validation pipeline working
  - [ ] Approval workflow implemented
  - [ ] Basic execution framework

- [ ] **Database Models**
  - [ ] EvolutionProposal table added
  - [ ] EvolutionHistory table added
  - [ ] EvolutionMetrics table added

- [ ] **Prompts Updated**
  - [ ] EVOLUTION_PROPOSAL_PROMPT added
  - [ ] VALIDATION_PROMPT added

- [ ] **Tests Passing**
  - [ ] All unit tests pass
  - [ ] Integration tests pass

---

## File Summary

| Action | File Path |
|--------|-----------|
| CREATE | `src/master_ai/evolution_models.py` |
| CREATE | `src/master_ai/confidence_scorer.py` |
| REPLACE | `src/master_ai/evolution.py` |
| MODIFY | `src/database/models.py` |
| MODIFY | `src/master_ai/prompts.py` |
| CREATE | `tests/test_evolution.py` |

---

## Next Part Preview

**Part 5.25: Evolution Engine - Code Analysis & AST Utilities** will cover:
- AST-based code analysis tools
- Dependency analysis and impact assessment
- Code quality metrics
- Refactoring detection
- Safe modification boundaries

---

*End of Part 5 - Evolution Engine - Core Models & Proposal System*
