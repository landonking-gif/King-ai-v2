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
    
    def test_cannot_execute_low_confidence_proposal(self):
        """Proposal cannot execute with low confidence."""
        proposal = EvolutionProposal(
            title="Test",
            description="Test proposal",
            proposal_type=ProposalType.CODE_MODIFICATION,
            status=ProposalStatus.APPROVED,
            confidence_score=ConfidenceScore(overall=0.5, components={}),
            validation_result=ValidationResult(passed=True)
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
    
    def test_many_changes_high_risk(self):
        """Proposals with many changes are high risk."""
        changes = [
            CodeChange(file_path=f"file{i}.py", change_type="modify")
            for i in range(15)
        ]
        proposal = EvolutionProposal(
            title="Test",
            description="Test proposal",
            proposal_type=ProposalType.CODE_MODIFICATION,
            changes=changes
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
        risk = proposal.calculate_risk_level()
        assert risk in [RiskLevel.HIGH, RiskLevel.CRITICAL]
    
    def test_to_dict_backward_compatibility(self):
        """to_dict method provides backward compatibility."""
        proposal = EvolutionProposal(
            title="Test",
            description="Test proposal",
            proposal_type=ProposalType.CODE_MODIFICATION
        )
        data = proposal.to_dict()
        assert "id" in data
        assert "title" in data
        assert "status" in data
        assert "risk_level" in data


class TestCodeChange:
    """Tests for CodeChange model."""
    
    def test_safe_modification(self):
        """Small modifications are considered safe."""
        change = CodeChange(
            file_path="test.py",
            change_type="modify",
            old_content="x = 1",
            new_content="x = 2"
        )
        assert change.is_safe
    
    def test_unsafe_deletion(self):
        """Deletions are considered unsafe."""
        change = CodeChange(
            file_path="test.py",
            change_type="delete",
            old_content="x = 1"
        )
        assert not change.is_safe
    
    def test_unsafe_large_change(self):
        """Large changes are considered unsafe."""
        change = CodeChange(
            file_path="test.py",
            change_type="modify",
            old_content="x" * 1001,
            new_content="y" * 500
        )
        assert not change.is_safe


class TestConfidenceScorer:
    """Tests for confidence scorer."""
    
    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM router."""
        llm = MagicMock()
        llm.complete = AsyncMock(return_value="0.85 - Good quality")
        return llm
    
    @pytest.fixture
    def mock_engine(self):
        """Create mock evolution engine."""
        engine = MagicMock()
        engine.get_similar_proposals = AsyncMock(return_value=[])
        return engine
    
    @pytest.fixture
    def scorer(self, mock_llm, mock_engine):
        """Create confidence scorer."""
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
        assert 'code_quality' in score.components
    
    @pytest.mark.asyncio
    async def test_score_with_changes(self, scorer):
        """Test scoring with code changes."""
        proposal = EvolutionProposal(
            title="Test",
            description="Test proposal",
            proposal_type=ProposalType.CODE_MODIFICATION,
            changes=[
                CodeChange(
                    file_path="test.py",
                    change_type="modify",
                    new_content="def foo(): pass"
                )
            ]
        )
        
        score = await scorer.score_proposal(proposal)
        
        assert isinstance(score, ConfidenceScore)
        assert score.overall > 0.0
    
    def test_threshold_checking_high_confidence(self, scorer):
        """Test threshold validation with high confidence."""
        high_confidence = ConfidenceScore(
            overall=0.9,
            components={"code_quality": 0.8, "testing": 0.7, "historical": 0.8, "llm": 0.9}
        )
        
        assert scorer.meets_threshold(high_confidence, "high")
        assert scorer.meets_threshold(high_confidence, "medium")
    
    def test_threshold_checking_low_confidence(self, scorer):
        """Test threshold validation with low confidence."""
        low_confidence = ConfidenceScore(
            overall=0.5,
            components={"code_quality": 0.4, "testing": 0.3}
        )
        
        assert not scorer.meets_threshold(low_confidence, "medium")
        assert not scorer.meets_threshold(low_confidence, "high")
    
    def test_threshold_checking_one_low_component(self, scorer):
        """Test threshold with one low component score."""
        mixed_confidence = ConfidenceScore(
            overall=0.8,
            components={"code_quality": 0.9, "testing": 0.4, "historical": 0.8}
        )
        
        # Should fail medium threshold due to low testing score
        assert not scorer.meets_threshold(mixed_confidence, "medium")


class TestEvolutionEngine:
    """Tests for evolution engine."""
    
    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM router."""
        llm = MagicMock()
        llm.complete = AsyncMock(return_value='''{
            "title": "Test Proposal",
            "description": "A test improvement",
            "changes": [],
            "config_changes": {},
            "estimated_risk": "low",
            "justification": "Testing purposes"
        }''')
        return llm
    
    @pytest.fixture
    def engine(self, mock_llm):
        """Create evolution engine."""
        return EvolutionEngine(mock_llm, sandbox=None)
    
    @pytest.mark.asyncio
    async def test_propose_improvement_new_interface(self, engine):
        """Test proposal generation with new interface."""
        with patch('src.master_ai.evolution.EvolutionEngine._persist_proposal'):
            proposal = await engine.propose_improvement(
                goal="Improve performance",
                context="Current system state"
            )
        
        assert isinstance(proposal, EvolutionProposal)
        assert proposal.title == "Test Proposal"
        assert proposal.confidence_score is not None
        assert proposal.status in [ProposalStatus.READY, ProposalStatus.APPROVED]
    
    @pytest.mark.asyncio
    async def test_propose_improvement_legacy_interface(self, engine):
        """Test proposal generation with legacy interface (context only)."""
        engine.llm.complete = AsyncMock(return_value='''{
            "is_beneficial": true,
            "type": "code_mod",
            "description": "Legacy improvement",
            "confidence": 0.8
        }''')
        
        with patch('src.master_ai.evolution.EvolutionEngine._persist_proposal'):
            with patch('src.master_ai.kpi_monitor.kpi_monitor.get_system_health', return_value="healthy"):
                proposal = await engine.propose_improvement(context="System context")
        
        assert isinstance(proposal, EvolutionProposal)
        assert proposal.confidence_score is not None
    
    @pytest.mark.asyncio
    async def test_propose_improvement_not_beneficial(self, engine):
        """Test proposal when improvement is not beneficial."""
        engine.llm.complete = AsyncMock(return_value='''{
            "is_beneficial": false,
            "reason": "System is optimal"
        }''')
        
        with patch('src.master_ai.evolution.EvolutionEngine._persist_proposal'):
            with patch('src.master_ai.kpi_monitor.kpi_monitor.get_system_health', return_value="healthy"):
                proposal = await engine.propose_improvement(context="System context")
        
        assert isinstance(proposal, EvolutionProposal)
        assert proposal.status == ProposalStatus.REJECTED
    
    @pytest.mark.asyncio
    async def test_validate_proposal(self, engine):
        """Test proposal validation."""
        proposal = EvolutionProposal(
            title="Test",
            description="Test proposal",
            proposal_type=ProposalType.CODE_MODIFICATION,
            changes=[
                CodeChange(
                    file_path="test.py",
                    change_type="add",
                    new_content="def test(): pass"
                )
            ]
        )
        
        with patch('src.master_ai.evolution.EvolutionEngine._update_proposal'):
            result = await engine.validate_proposal(proposal)
        
        assert isinstance(result, ValidationResult)
        assert hasattr(result, 'passed')
        assert proposal.validation_result is not None
    
    @pytest.mark.asyncio
    async def test_validate_syntax_error(self, engine):
        """Test validation with syntax error."""
        proposal = EvolutionProposal(
            title="Test",
            description="Test proposal",
            proposal_type=ProposalType.CODE_MODIFICATION,
            changes=[
                CodeChange(
                    file_path="test.py",
                    change_type="add",
                    new_content="def test( invalid syntax"
                )
            ]
        )
        
        with patch('src.master_ai.evolution.EvolutionEngine._update_proposal'):
            result = await engine.validate_proposal(proposal)
        
        assert not result.passed
        assert len(result.errors) > 0
    
    @pytest.mark.asyncio
    async def test_approve_proposal(self, engine):
        """Test proposal approval."""
        proposal = EvolutionProposal(
            title="Test",
            description="Test proposal",
            proposal_type=ProposalType.CODE_MODIFICATION,
            status=ProposalStatus.READY
        )
        
        engine._active_proposals[proposal.id] = proposal
        
        with patch('src.master_ai.evolution.EvolutionEngine._update_proposal'):
            success = await engine.approve_proposal(proposal.id, approver="test_user")
        
        assert success
        assert proposal.status == ProposalStatus.APPROVED
        assert proposal.approved_by == "test_user"
        assert proposal.approved_at is not None
    
    @pytest.mark.asyncio
    async def test_execute_proposal_success(self, engine):
        """Test successful proposal execution."""
        proposal = EvolutionProposal(
            title="Test",
            description="Test proposal",
            proposal_type=ProposalType.CODE_MODIFICATION,
            status=ProposalStatus.APPROVED,
            confidence_score=ConfidenceScore(overall=0.8, components={}),
            validation_result=ValidationResult(passed=True)
        )
        
        with patch('src.master_ai.evolution.EvolutionEngine._update_proposal'):
            with patch('src.master_ai.evolution.EvolutionEngine._create_rollback_data', return_value={}):
                with patch('src.master_ai.evolution.EvolutionEngine._record_history'):
                    result = await engine.execute_proposal(proposal)
        
        assert result["success"]
        assert proposal.status == ProposalStatus.COMPLETED
    
    @pytest.mark.asyncio
    async def test_execute_proposal_not_ready(self, engine):
        """Test executing proposal that's not ready."""
        proposal = EvolutionProposal(
            title="Test",
            description="Test proposal",
            proposal_type=ProposalType.CODE_MODIFICATION,
            status=ProposalStatus.DRAFT
        )
        
        with pytest.raises(ValueError, match="not ready for execution"):
            await engine.execute_proposal(proposal)
    
    def test_get_metrics(self, engine):
        """Test getting evolution metrics."""
        metrics = engine.get_metrics()
        assert hasattr(metrics, 'total_proposals')
        assert hasattr(metrics, 'success_rate')
    
    def test_get_active_proposals(self, engine):
        """Test getting active proposals."""
        proposal = EvolutionProposal(
            title="Test",
            description="Test proposal",
            proposal_type=ProposalType.CODE_MODIFICATION
        )
        
        engine._active_proposals[proposal.id] = proposal
        
        active = engine.get_active_proposals()
        assert len(active) == 1
        assert active[0].id == proposal.id
