"""
Skeptic Agent.

Devil's advocate agent that challenges assumptions and validates plans.
Based on the mother-harness skeptic/critic pattern.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from src.utils.structured_logging import get_logger

logger = get_logger("skeptic_agent")


class ChallengeType(str, Enum):
    """Types of challenges the skeptic can raise."""
    ASSUMPTION = "assumption"
    FEASIBILITY = "feasibility"
    RISK = "risk"
    COST = "cost"
    MARKET = "market"
    LEGAL = "legal"
    TECHNICAL = "technical"
    TIMELINE = "timeline"
    DEPENDENCY = "dependency"
    ETHICAL = "ethical"


class Severity(str, Enum):
    """Severity of a challenge."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Challenge:
    """A challenge raised by the skeptic."""
    id: str = field(default_factory=lambda: f"chal_{uuid4().hex[:8]}")
    challenge_type: ChallengeType = ChallengeType.ASSUMPTION
    severity: Severity = Severity.MEDIUM
    title: str = ""
    description: str = ""
    evidence: List[str] = field(default_factory=list)
    affected_areas: List[str] = field(default_factory=list)
    mitigation_suggestions: List[str] = field(default_factory=list)
    resolved: bool = False
    resolution: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "challenge_type": self.challenge_type.value,
            "severity": self.severity.value,
            "title": self.title,
            "description": self.description,
            "evidence": self.evidence,
            "affected_areas": self.affected_areas,
            "mitigation_suggestions": self.mitigation_suggestions,
            "resolved": self.resolved,
            "resolution": self.resolution,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class ChallengeReport:
    """Report from skeptic analysis."""
    id: str = field(default_factory=lambda: f"report_{uuid4().hex[:8]}")
    target: str = ""
    target_type: str = ""
    challenges: List[Challenge] = field(default_factory=list)
    overall_risk_score: float = 0.0
    recommendation: str = ""
    proceed_with_caution: bool = False
    must_address: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def critical_challenges(self) -> List[Challenge]:
        """Get critical challenges."""
        return [c for c in self.challenges if c.severity == Severity.CRITICAL]
    
    @property
    def high_severity_challenges(self) -> List[Challenge]:
        """Get high severity challenges."""
        return [c for c in self.challenges if c.severity in [Severity.CRITICAL, Severity.HIGH]]
    
    @property
    def unresolved_challenges(self) -> List[Challenge]:
        """Get unresolved challenges."""
        return [c for c in self.challenges if not c.resolved]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "target": self.target,
            "target_type": self.target_type,
            "challenges": [c.to_dict() for c in self.challenges],
            "overall_risk_score": self.overall_risk_score,
            "recommendation": self.recommendation,
            "proceed_with_caution": self.proceed_with_caution,
            "must_address": self.must_address,
            "critical_count": len(self.critical_challenges),
            "unresolved_count": len(self.unresolved_challenges),
            "created_at": self.created_at.isoformat(),
        }


class SkepticAgent:
    """
    Devil's advocate agent.
    
    Challenges assumptions, identifies risks, and validates plans
    before they're executed. Acts as a quality gate.
    """
    
    def __init__(self, llm_router=None):
        """
        Initialize skeptic agent.
        
        Args:
            llm_router: Router for LLM calls (optional, uses heuristics if None)
        """
        self.llm_router = llm_router
        
        # Challenge patterns to look for
        self._patterns = {
            ChallengeType.ASSUMPTION: [
                "assumes market exists",
                "assumes customers will pay",
                "assumes technology is ready",
                "assumes no competition",
                "assumes linear growth",
                "assumes resources available",
            ],
            ChallengeType.FEASIBILITY: [
                "technical complexity underestimated",
                "skills not available",
                "infrastructure required",
                "integration challenges",
                "scale limitations",
            ],
            ChallengeType.RISK: [
                "single point of failure",
                "vendor lock-in",
                "regulatory changes",
                "market volatility",
                "key person dependency",
            ],
            ChallengeType.COST: [
                "hidden costs",
                "underestimated expenses",
                "opportunity cost",
                "maintenance overhead",
                "scaling costs non-linear",
            ],
            ChallengeType.MARKET: [
                "market size unclear",
                "competition analysis lacking",
                "customer validation missing",
                "pricing strategy weak",
                "distribution unclear",
            ],
            ChallengeType.LEGAL: [
                "regulatory compliance needed",
                "intellectual property risks",
                "liability concerns",
                "contract obligations",
                "privacy requirements",
            ],
            ChallengeType.TIMELINE: [
                "unrealistic deadlines",
                "dependencies not mapped",
                "buffer time missing",
                "parallel work assumed",
                "external delays likely",
            ],
        }
    
    async def challenge_business_plan(
        self,
        plan: Dict[str, Any],
        business_type: str,
    ) -> ChallengeReport:
        """
        Challenge a business plan.
        
        Args:
            plan: Business plan to challenge
            business_type: Type of business
            
        Returns:
            Challenge report with identified issues
        """
        logger.info(f"Skeptic analyzing business plan for {business_type}")
        
        challenges = []
        
        # Use LLM if available for deep analysis
        if self.llm_router:
            challenges = await self._llm_challenge_plan(plan, business_type)
        else:
            challenges = self._heuristic_challenge_plan(plan, business_type)
        
        # Calculate overall risk score
        risk_score = self._calculate_risk_score(challenges)
        
        # Generate recommendation
        recommendation = self._generate_recommendation(challenges, risk_score)
        
        # Identify must-address items
        must_address = [
            c.title for c in challenges 
            if c.severity in [Severity.CRITICAL, Severity.HIGH] and not c.resolved
        ]
        
        report = ChallengeReport(
            target=f"Business Plan: {plan.get('business_name', 'Unknown')}",
            target_type="business_plan",
            challenges=challenges,
            overall_risk_score=risk_score,
            recommendation=recommendation,
            proceed_with_caution=risk_score > 0.5,
            must_address=must_address,
        )
        
        logger.info(
            f"Skeptic report complete: {len(challenges)} challenges, "
            f"risk score: {risk_score:.2f}"
        )
        
        return report
    
    async def challenge_research(
        self,
        research: Dict[str, Any],
        topic: str,
    ) -> ChallengeReport:
        """
        Challenge research findings.
        
        Args:
            research: Research findings to challenge
            topic: Research topic
            
        Returns:
            Challenge report
        """
        logger.info(f"Skeptic analyzing research on {topic}")
        
        challenges = []
        
        # Check for common research issues
        if not research.get("sources"):
            challenges.append(Challenge(
                challenge_type=ChallengeType.ASSUMPTION,
                severity=Severity.HIGH,
                title="No Sources Cited",
                description="Research has no cited sources, cannot verify claims.",
                mitigation_suggestions=["Add primary sources", "Include citations"],
            ))
        
        if research.get("sources") and len(research.get("sources", [])) < 3:
            challenges.append(Challenge(
                challenge_type=ChallengeType.ASSUMPTION,
                severity=Severity.MEDIUM,
                title="Limited Sources",
                description="Only few sources used, may not represent full picture.",
                mitigation_suggestions=["Expand source base", "Include contrarian views"],
            ))
        
        # Check for recency
        data_date = research.get("data_date")
        if not data_date:
            challenges.append(Challenge(
                challenge_type=ChallengeType.ASSUMPTION,
                severity=Severity.MEDIUM,
                title="Data Freshness Unknown",
                description="No indication of when data was collected.",
                mitigation_suggestions=["Add data timestamps", "Verify current relevance"],
            ))
        
        # Check for bias indicators
        if research.get("sentiment") == "positive" and not research.get("risks"):
            challenges.append(Challenge(
                challenge_type=ChallengeType.RISK,
                severity=Severity.HIGH,
                title="Potential Confirmation Bias",
                description="Research is overwhelmingly positive with no risks identified.",
                mitigation_suggestions=["Seek contrarian perspectives", "Identify potential downsides"],
            ))
        
        risk_score = self._calculate_risk_score(challenges)
        
        return ChallengeReport(
            target=f"Research: {topic}",
            target_type="research",
            challenges=challenges,
            overall_risk_score=risk_score,
            recommendation=self._generate_recommendation(challenges, risk_score),
            proceed_with_caution=risk_score > 0.4,
            must_address=[c.title for c in challenges if c.severity == Severity.HIGH],
        )
    
    async def challenge_code(
        self,
        code: str,
        language: str,
        purpose: str,
    ) -> ChallengeReport:
        """
        Challenge generated code.
        
        Args:
            code: Code to review
            language: Programming language
            purpose: What the code is for
            
        Returns:
            Challenge report
        """
        logger.info(f"Skeptic reviewing {language} code for {purpose}")
        
        challenges = []
        
        # Security patterns
        security_patterns = [
            ("eval(", "Potential code injection via eval()"),
            ("exec(", "Potential code injection via exec()"),
            ("os.system", "Shell command execution risk"),
            ("subprocess.call", "Shell command without shell=False check"),
            ("pickle.", "Pickle deserialization vulnerability"),
            ("password", "Hardcoded password detected"),
            ("api_key", "Potential hardcoded API key"),
            ("secret", "Potential hardcoded secret"),
            ("SQL", "Potential SQL injection if not parameterized"),
        ]
        
        for pattern, issue in security_patterns:
            if pattern.lower() in code.lower():
                challenges.append(Challenge(
                    challenge_type=ChallengeType.TECHNICAL,
                    severity=Severity.HIGH,
                    title=f"Security Concern: {pattern}",
                    description=issue,
                    affected_areas=["security"],
                    mitigation_suggestions=[
                        "Review usage carefully",
                        "Use safer alternatives",
                        "Add input validation",
                    ],
                ))
        
        # Error handling
        if "except:" in code or "except Exception:" in code:
            if "pass" in code:
                challenges.append(Challenge(
                    challenge_type=ChallengeType.TECHNICAL,
                    severity=Severity.MEDIUM,
                    title="Silent Exception Handling",
                    description="Exceptions are caught but silently ignored.",
                    mitigation_suggestions=["Log exceptions", "Handle specific exception types"],
                ))
        
        if "try:" not in code and len(code) > 100:
            challenges.append(Challenge(
                challenge_type=ChallengeType.TECHNICAL,
                severity=Severity.LOW,
                title="No Error Handling",
                description="Code has no try/except blocks for error handling.",
                mitigation_suggestions=["Add appropriate error handling"],
            ))
        
        risk_score = self._calculate_risk_score(challenges)
        
        return ChallengeReport(
            target=f"Code: {purpose}",
            target_type="code",
            challenges=challenges,
            overall_risk_score=risk_score,
            recommendation=self._generate_recommendation(challenges, risk_score),
            proceed_with_caution=len(challenges) > 0,
            must_address=[c.title for c in challenges if c.severity == Severity.HIGH],
        )
    
    async def challenge_decision(
        self,
        decision: str,
        context: Dict[str, Any],
        options_considered: List[str] = None,
    ) -> ChallengeReport:
        """
        Challenge a decision.
        
        Args:
            decision: The decision being made
            context: Context around the decision
            options_considered: Other options that were considered
            
        Returns:
            Challenge report
        """
        logger.info(f"Skeptic challenging decision: {decision[:50]}...")
        
        challenges = []
        
        # Check if alternatives were considered
        if not options_considered or len(options_considered) < 2:
            challenges.append(Challenge(
                challenge_type=ChallengeType.ASSUMPTION,
                severity=Severity.MEDIUM,
                title="Limited Options Considered",
                description="Decision made without evaluating multiple alternatives.",
                mitigation_suggestions=[
                    "Identify at least 3 alternatives",
                    "Document why alternatives were rejected",
                ],
            ))
        
        # Check for reversibility
        if context.get("irreversible", False):
            challenges.append(Challenge(
                challenge_type=ChallengeType.RISK,
                severity=Severity.HIGH,
                title="Irreversible Decision",
                description="This decision cannot be easily reversed.",
                mitigation_suggestions=[
                    "Build in checkpoints",
                    "Consider staged rollout",
                    "Add exit clauses",
                ],
            ))
        
        # Check for stakeholder input
        if not context.get("stakeholders_consulted"):
            challenges.append(Challenge(
                challenge_type=ChallengeType.RISK,
                severity=Severity.MEDIUM,
                title="Stakeholders Not Consulted",
                description="Key stakeholders may not have been involved.",
                mitigation_suggestions=[
                    "Identify affected stakeholders",
                    "Gather input before finalizing",
                ],
            ))
        
        risk_score = self._calculate_risk_score(challenges)
        
        return ChallengeReport(
            target=f"Decision: {decision[:50]}...",
            target_type="decision",
            challenges=challenges,
            overall_risk_score=risk_score,
            recommendation=self._generate_recommendation(challenges, risk_score),
            proceed_with_caution=risk_score > 0.3,
            must_address=[c.title for c in challenges if c.severity == Severity.HIGH],
        )
    
    def resolve_challenge(
        self,
        report: ChallengeReport,
        challenge_id: str,
        resolution: str,
    ) -> bool:
        """
        Mark a challenge as resolved.
        
        Args:
            report: The challenge report
            challenge_id: ID of challenge to resolve
            resolution: How it was resolved
            
        Returns:
            True if resolved, False if not found
        """
        for challenge in report.challenges:
            if challenge.id == challenge_id:
                challenge.resolved = True
                challenge.resolution = resolution
                return True
        return False
    
    # Private methods
    
    def _heuristic_challenge_plan(
        self,
        plan: Dict[str, Any],
        business_type: str,
    ) -> List[Challenge]:
        """Apply heuristic challenges to a plan."""
        challenges = []
        
        # Check financial projections
        if plan.get("revenue_projections"):
            projections = plan["revenue_projections"]
            if isinstance(projections, list) and len(projections) > 1:
                # Check for hockey stick growth
                growth_rates = []
                for i in range(1, len(projections)):
                    if projections[i-1] > 0:
                        rate = (projections[i] - projections[i-1]) / projections[i-1]
                        growth_rates.append(rate)
                
                if growth_rates and max(growth_rates) > 2.0:  # 200% growth
                    challenges.append(Challenge(
                        challenge_type=ChallengeType.ASSUMPTION,
                        severity=Severity.HIGH,
                        title="Hockey Stick Growth Assumption",
                        description="Revenue projections show extremely aggressive growth.",
                        evidence=[f"Growth rate exceeds 200% in some periods"],
                        mitigation_suggestions=[
                            "Justify growth assumptions",
                            "Create conservative scenario",
                        ],
                    ))
        
        # Check market analysis
        if not plan.get("market_analysis") or not plan.get("target_market"):
            challenges.append(Challenge(
                challenge_type=ChallengeType.MARKET,
                severity=Severity.HIGH,
                title="Market Analysis Missing",
                description="No clear market analysis or target market definition.",
                mitigation_suggestions=[
                    "Define target customer segments",
                    "Estimate market size",
                    "Identify market trends",
                ],
            ))
        
        # Check competitive analysis
        if not plan.get("competitors"):
            challenges.append(Challenge(
                challenge_type=ChallengeType.MARKET,
                severity=Severity.MEDIUM,
                title="Competition Not Analyzed",
                description="No competitive analysis included.",
                mitigation_suggestions=[
                    "Identify key competitors",
                    "Analyze competitive advantages",
                ],
            ))
        
        # Check for risk section
        if not plan.get("risks"):
            challenges.append(Challenge(
                challenge_type=ChallengeType.RISK,
                severity=Severity.MEDIUM,
                title="No Risk Analysis",
                description="Plan does not include risk identification.",
                mitigation_suggestions=[
                    "Identify key risks",
                    "Create mitigation strategies",
                ],
            ))
        
        # Check resource requirements
        if not plan.get("resources_required") and not plan.get("budget"):
            challenges.append(Challenge(
                challenge_type=ChallengeType.FEASIBILITY,
                severity=Severity.MEDIUM,
                title="Resource Requirements Unclear",
                description="No clear budget or resource requirements.",
                mitigation_suggestions=[
                    "Detail required resources",
                    "Create budget breakdown",
                ],
            ))
        
        # Check timeline
        if not plan.get("timeline") and not plan.get("milestones"):
            challenges.append(Challenge(
                challenge_type=ChallengeType.TIMELINE,
                severity=Severity.MEDIUM,
                title="No Timeline or Milestones",
                description="Plan lacks timeline or milestone definitions.",
                mitigation_suggestions=[
                    "Create project timeline",
                    "Define measurable milestones",
                ],
            ))
        
        return challenges
    
    async def _llm_challenge_plan(
        self,
        plan: Dict[str, Any],
        business_type: str,
    ) -> List[Challenge]:
        """Use LLM for deep challenge analysis."""
        # Start with heuristics
        challenges = self._heuristic_challenge_plan(plan, business_type)
        
        # Enhance with LLM if available
        if self.llm_router:
            try:
                prompt = f"""As a critical business analyst, review this {business_type} business plan and identify:
1. Unstated assumptions that could be wrong
2. Risks that aren't addressed
3. Feasibility concerns
4. Market/competitive issues
5. Financial red flags

Plan summary:
{str(plan)[:2000]}

For each issue, provide:
- Type (assumption/risk/feasibility/market/cost/timeline)
- Severity (low/medium/high/critical)
- Description
- Mitigation suggestion

Be thorough but constructive."""

                result = await self.llm_router.route(
                    model="llama3.2",
                    messages=[{"role": "user", "content": prompt}],
                )
                
                # Parse LLM response and add to challenges
                # This would need proper parsing logic
                
            except Exception as e:
                logger.warning(f"LLM challenge analysis failed: {e}")
        
        return challenges
    
    def _calculate_risk_score(self, challenges: List[Challenge]) -> float:
        """Calculate overall risk score from challenges."""
        if not challenges:
            return 0.0
        
        severity_weights = {
            Severity.LOW: 0.1,
            Severity.MEDIUM: 0.25,
            Severity.HIGH: 0.5,
            Severity.CRITICAL: 1.0,
        }
        
        total_weight = sum(
            severity_weights.get(c.severity, 0.25) 
            for c in challenges if not c.resolved
        )
        
        # Normalize to 0-1 range (assuming max 10 critical issues)
        return min(1.0, total_weight / 5.0)
    
    def _generate_recommendation(
        self,
        challenges: List[Challenge],
        risk_score: float,
    ) -> str:
        """Generate recommendation based on challenges."""
        unresolved = [c for c in challenges if not c.resolved]
        critical = [c for c in unresolved if c.severity == Severity.CRITICAL]
        high = [c for c in unresolved if c.severity == Severity.HIGH]
        
        if critical:
            return (
                f"HOLD: {len(critical)} critical issue(s) must be addressed "
                f"before proceeding. Risk score: {risk_score:.2f}"
            )
        elif high:
            return (
                f"CAUTION: {len(high)} high-severity issue(s) should be "
                f"addressed. Risk score: {risk_score:.2f}"
            )
        elif risk_score > 0.3:
            return (
                f"PROCEED WITH MONITORING: Some issues identified. "
                f"Risk score: {risk_score:.2f}"
            )
        else:
            return f"PROCEED: Low risk identified. Risk score: {risk_score:.2f}"


# Global instance
_skeptic_agent: Optional[SkepticAgent] = None


def get_skeptic_agent(llm_router=None) -> SkepticAgent:
    """Get or create the global skeptic agent."""
    global _skeptic_agent
    if _skeptic_agent is None:
        _skeptic_agent = SkepticAgent(llm_router)
    return _skeptic_agent
