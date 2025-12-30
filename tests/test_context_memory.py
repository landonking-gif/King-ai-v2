"""
Tests for context and memory system.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime

from src.master_ai.context import ContextManager
from src.database.vector_store import VectorStore, MemoryType
from src.utils.token_manager import (
    ContextBudget, ContextSection, estimate_tokens,
    smart_truncate, TokenBudget
)
from src.utils.embedding_client import EmbeddingClient


class TestTokenManager:
    """Tests for token management utilities."""
    
    def test_estimate_tokens(self):
        """Test token estimation."""
        text = "Hello world this is a test"
        tokens = estimate_tokens(text)
        # Roughly 4 chars per token
        assert 5 <= tokens <= 10
    
    def test_context_budget_allocation(self):
        """Test budget is properly allocated."""
        budget = ContextBudget(total_tokens=10000)
        
        # Check all sections have allocations
        for section in ContextSection:
            assert budget.get_budget(section).allocated > 0
    
    def test_smart_truncate_preserves_structure(self):
        """Test smart truncation preserves paragraphs."""
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        truncated = smart_truncate(text, max_tokens=20)
        
        # Should end at a paragraph boundary
        assert "paragraph" in truncated
    
    def test_budget_can_add(self):
        """Test budget capacity checking."""
        budget = ContextBudget(total_tokens=1000)
        
        assert budget.can_add(500)
        budget.set_content(ContextSection.CURRENT_STATE, "x" * 4000)  # 1000 tokens
        assert not budget.can_add(500)


class TestEmbeddingClient:
    """Tests for embedding generation."""
    
    @pytest.fixture
    def client(self):
        return EmbeddingClient()
    
    def test_cosine_similarity(self, client):
        """Test cosine similarity calculation."""
        a = [1.0, 0.0, 0.0]
        b = [1.0, 0.0, 0.0]
        
        similarity = client.cosine_similarity(a, b)
        assert similarity == pytest.approx(1.0)
    
    def test_cosine_similarity_orthogonal(self, client):
        """Test orthogonal vectors have 0 similarity."""
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        
        similarity = client.cosine_similarity(a, b)
        assert similarity == pytest.approx(0.0)


class TestVectorStore:
    """Tests for vector store operations."""
    
    @pytest.fixture
    def mock_store(self):
        with patch('src.database.vector_store.Pinecone'):
            with patch('src.database.vector_store.embedding_client') as mock_embed:
                mock_embed.embed = AsyncMock(return_value=[0.1] * 768)
                mock_embed.dimension = 768
                store = VectorStore()
                store.index = MagicMock()
                return store
    
    @pytest.mark.asyncio
    async def test_store_memory(self, mock_store):
        """Test storing a memory."""
        memory_id = await mock_store.store_memory(
            text="Test memory content",
            memory_type=MemoryType.DECISION,
            metadata={"key": "value"}
        )
        
        assert memory_id is not None
        mock_store.index.upsert.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_search_memories(self, mock_store):
        """Test searching memories."""
        mock_store.index.query.return_value = MagicMock(
            matches=[
                MagicMock(
                    id="test-id",
                    score=0.9,
                    metadata={"text": "Test", "type": "decision", "full_text": "Full test"}
                )
            ]
        )
        
        results = await mock_store.search_memories("test query")
        
        assert len(results) == 1
        assert results[0]["score"] == 0.9


class TestContextManager:
    """Tests for context manager."""
    
    @pytest.fixture
    def context_manager(self):
        return ContextManager()
    
    @pytest.mark.asyncio
    async def test_build_context_returns_string(self, context_manager):
        """Test that build_context returns a string."""
        with patch('src.master_ai.context.get_db') as mock_db:
            mock_session = AsyncMock()
            mock_session.execute.return_value.scalars.return_value.all.return_value = []
            mock_db.return_value.__aenter__.return_value = mock_session
            
            context = await context_manager.build_context()
            
            assert isinstance(context, str)
            assert "CURRENT TIME" in context
    
    @pytest.mark.asyncio
    async def test_add_to_conversation(self, context_manager):
        """Test adding messages to conversation."""
        with patch('src.master_ai.context.get_db') as mock_db:
            mock_session = AsyncMock()
            mock_db.return_value.__aenter__.return_value = mock_session
            
            await context_manager.add_to_conversation(
                role="user",
                content="Hello",
                persist=True
            )
            
            assert len(context_manager._conversation_cache) == 1
            assert context_manager._conversation_cache[0]["role"] == "user"
