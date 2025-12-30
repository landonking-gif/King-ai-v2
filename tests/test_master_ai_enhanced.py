"""
Tests for enhanced Master AI functionality.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime

from src.master_ai.brain import MasterAI
from src.master_ai.models import (
    ClassifiedIntent, IntentType, ActionType, MasterAIResponse
)


class TestIntentClassification:
    """Tests for intent classification."""
    
    @pytest.fixture
    def master_ai(self):
        with patch('src.master_ai.brain.LLMRouter'):
            with patch('src.master_ai.brain.ContextManager'):
                with patch('src.master_ai.brain.Planner'):
                    with patch('src.master_ai.brain.EvolutionEngine'):
                        with patch('src.master_ai.brain.AgentRouter'):
                            return MasterAI()
    
    @pytest.mark.asyncio
    async def test_conversation_intent(self, master_ai):
        """Test classification of conversation intent."""
        master_ai.llm_router.complete = AsyncMock(return_value='''
        {"type": "conversation", "action": null, "parameters": {}, "confidence": 0.9}
        ''')
        master_ai.context.build_context = AsyncMock(return_value="test context")
        
        intent = await master_ai._classify_intent("Hello there", "context")
        assert intent.type == IntentType.CONVERSATION
    
    @pytest.mark.asyncio
    async def test_command_intent(self, master_ai):
        """Test classification of command intent."""
        master_ai.llm_router.complete = AsyncMock(return_value='''
        {"type": "command", "action": "start_business", "parameters": {"niche": "pets"}, "confidence": 0.85}
        ''')
        
        intent = await master_ai._classify_intent("Start a pet store", "context")
        assert intent.type == IntentType.COMMAND
        assert intent.action == ActionType.START_BUSINESS
    
    @pytest.mark.asyncio
    async def test_query_intent(self, master_ai):
        """Test classification of query intent."""
        master_ai.llm_router.complete = AsyncMock(return_value='''
        {"type": "query", "action": null, "parameters": {}, "confidence": 0.9}
        ''')
        
        intent = await master_ai._classify_intent("What is our revenue?", "context")
        assert intent.type == IntentType.QUERY
    
    @pytest.mark.asyncio
    async def test_fallback_on_parse_error(self, master_ai):
        """Test fallback to conversation on parse error."""
        master_ai.llm_router.complete = AsyncMock(return_value="invalid json")
        
        intent = await master_ai._classify_intent("Test input", "context")
        assert intent.type == IntentType.CONVERSATION
        assert intent.confidence < 0.5


class TestMasterAIResponse:
    """Tests for response handling."""
    
    @pytest.fixture
    def master_ai(self):
        with patch('src.master_ai.brain.LLMRouter'):
            with patch('src.master_ai.brain.ContextManager'):
                with patch('src.master_ai.brain.Planner'):
                    with patch('src.master_ai.brain.EvolutionEngine'):
                        with patch('src.master_ai.brain.AgentRouter'):
                            return MasterAI()
    
    @pytest.mark.asyncio
    async def test_process_input_returns_response(self, master_ai):
        """Test that process_input returns structured response."""
        master_ai.context.build_context = AsyncMock(return_value="context")
        master_ai._classify_intent = AsyncMock(return_value=ClassifiedIntent(
            type=IntentType.CONVERSATION
        ))
        master_ai._handle_conversation = AsyncMock(return_value="Hello!")
        
        response = await master_ai.process_input("Hi")
        
        assert isinstance(response, MasterAIResponse)
        assert response.type == "conversation"
        assert response.response == "Hello!"
    
    @pytest.mark.asyncio
    async def test_process_input_handles_error(self, master_ai):
        """Test that errors are handled gracefully."""
        master_ai.context.build_context = AsyncMock(side_effect=Exception("Test error"))
        
        response = await master_ai.process_input("Hi")
        
        assert isinstance(response, MasterAIResponse)
        assert response.type == "error"
        assert "error" in response.response.lower()


class TestTokenBudget:
    """Tests for token budget management."""
    
    @pytest.fixture
    def master_ai(self):
        with patch('src.master_ai.brain.LLMRouter'):
            with patch('src.master_ai.brain.ContextManager'):
                with patch('src.master_ai.brain.Planner'):
                    with patch('src.master_ai.brain.EvolutionEngine'):
                        with patch('src.master_ai.brain.AgentRouter'):
                            return MasterAI()
    
    @pytest.mark.asyncio
    async def test_token_budget_enforcement(self, master_ai):
        """Test that token budget is enforced."""
        master_ai._total_tokens_today = master_ai._token_budget_daily
        master_ai.llm_router.complete = AsyncMock(return_value="response")
        
        from src.utils.retry import TransientError
        with pytest.raises(TransientError, match="token budget"):
            await master_ai._call_llm("test prompt")
    
    @pytest.mark.asyncio
    async def test_token_counting(self, master_ai):
        """Test that tokens are counted."""
        master_ai.llm_router.complete = AsyncMock(return_value="test response")
        initial_tokens = master_ai._total_tokens_today
        
        await master_ai._call_llm("test prompt")
        
        assert master_ai._total_tokens_today > initial_tokens


class TestCommandHandling:
    """Tests for command handling."""
    
    @pytest.fixture
    def master_ai(self):
        with patch('src.master_ai.brain.LLMRouter'):
            with patch('src.master_ai.brain.ContextManager'):
                with patch('src.master_ai.brain.Planner'):
                    with patch('src.master_ai.brain.EvolutionEngine'):
                        with patch('src.master_ai.brain.AgentRouter'):
                            return MasterAI()
    
    @pytest.mark.asyncio
    async def test_command_creates_plan(self, master_ai):
        """Test that commands trigger plan creation."""
        intent = ClassifiedIntent(
            type=IntentType.COMMAND,
            action=ActionType.START_BUSINESS,
            parameters={"niche": "pets"}
        )
        
        master_ai.planner.create_plan = AsyncMock(return_value={
            "goal": "Start pet business",
            "steps": []
        })
        master_ai._generate_action_summary = AsyncMock(return_value="Done!")
        
        response = await master_ai._handle_command("Start a pet business", intent, "context")
        
        assert isinstance(response, MasterAIResponse)
        assert response.type == "action"
        master_ai.planner.create_plan.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_high_risk_tasks_require_approval(self, master_ai):
        """Test that high-risk tasks are queued for approval."""
        intent = ClassifiedIntent(
            type=IntentType.COMMAND,
            action=ActionType.START_BUSINESS
        )
        
        master_ai.planner.create_plan = AsyncMock(return_value={
            "goal": "Start business",
            "steps": [{
                "name": "Purchase domain",
                "description": "Buy domain name",
                "agent": "commerce",
                "requires_approval": True,
                "dependencies": [],
                "estimated_duration": "5 minutes",
                "input": {"cost": 100},
                "risk_level": "high"
            }]
        })
        master_ai._create_approval_task = AsyncMock(return_value={
            "id": "123",
            "name": "Purchase domain",
            "description": "Buy domain name",
            "agent": "commerce"
        })
        master_ai._generate_action_summary = AsyncMock(return_value="Queued for approval")
        
        response = await master_ai._handle_command("Buy a domain", intent, "context")
        
        assert len(response.pending_approvals) == 1
        assert len(response.actions_taken) == 0


class TestEvolutionProposals:
    """Tests for evolution proposals."""
    
    @pytest.fixture
    def master_ai(self):
        with patch('src.master_ai.brain.LLMRouter'):
            with patch('src.master_ai.brain.ContextManager'):
                with patch('src.master_ai.brain.Planner'):
                    with patch('src.master_ai.brain.EvolutionEngine'):
                        with patch('src.master_ai.brain.AgentRouter'):
                            with patch('src.master_ai.brain.settings') as mock_settings:
                                mock_settings.enable_self_modification = True
                                mock_settings.max_evolutions_per_hour = 5
                                mock_settings.evolution_confidence_threshold = 0.8
                                return MasterAI()
    
    @pytest.mark.asyncio
    async def test_evolution_rate_limiting(self, master_ai):
        """Test that evolution proposals are rate limited."""
        master_ai._evolution_count_this_hour = 10  # Already at limit
        master_ai.evolution.propose_improvement = AsyncMock(return_value={
            "is_beneficial": True,
            "confidence": 0.9
        })
        
        await master_ai._consider_evolution("context")
        
        # Should not call propose_improvement since we're at limit
        master_ai.evolution.propose_improvement.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_low_confidence_rejected(self, master_ai):
        """Test that low confidence proposals are rejected."""
        master_ai.evolution.propose_improvement = AsyncMock(return_value={
            "is_beneficial": True,
            "confidence": 0.5,  # Below threshold
            "changes": {"file.py": "new code"}
        })
        master_ai._process_evolution_proposal = AsyncMock()
        
        await master_ai._consider_evolution("context")
        
        # Should not process due to low confidence
        master_ai._process_evolution_proposal.assert_not_called()


class TestJSONParsing:
    """Tests for JSON parsing utility."""
    
    @pytest.fixture
    def master_ai(self):
        with patch('src.master_ai.brain.LLMRouter'):
            with patch('src.master_ai.brain.ContextManager'):
                with patch('src.master_ai.brain.Planner'):
                    with patch('src.master_ai.brain.EvolutionEngine'):
                        with patch('src.master_ai.brain.AgentRouter'):
                            return MasterAI()
    
    def test_parse_clean_json(self, master_ai):
        """Test parsing clean JSON."""
        response = '{"type": "conversation"}'
        parsed = master_ai._parse_json_response(response)
        assert parsed["type"] == "conversation"
    
    def test_parse_json_with_markdown(self, master_ai):
        """Test parsing JSON wrapped in markdown."""
        response = '```json\n{"type": "command"}\n```'
        parsed = master_ai._parse_json_response(response)
        assert parsed["type"] == "command"
    
    def test_parse_json_with_simple_backticks(self, master_ai):
        """Test parsing JSON wrapped in simple backticks."""
        response = '```\n{"type": "query"}\n```'
        parsed = master_ai._parse_json_response(response)
        assert parsed["type"] == "query"
