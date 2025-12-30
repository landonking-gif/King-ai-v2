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
        # Get similar proposals (will be implemented in Part 3 with vector search)
        # For now, use a simpler approach
        try:
            import asyncio
            similar = asyncio.run(self.evolution_engine.get_similar_proposals(proposal))
        except:
            similar = []
        
        if not similar:
            return 0.5  # Neutral score with no history
        
        successful = sum(1 for p in similar if p.status.value in ["completed", "applied"])
        total = len(similar)
        
        success_rate = successful / total if total > 0 else 0.5
        
        # Adjust based on recency (recent failures are worse)
        recent_failures = sum(
            1 for p in similar[:5]  # Last 5 similar proposals
            if p.status.value == "failed" and
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
