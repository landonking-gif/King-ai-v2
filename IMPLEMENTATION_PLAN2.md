King AI v2 - Phase 2 Implementation Plan
Verification Report (Phase 1)
✅ Completed (All Imports Pass)
Module	Status	Files
config/settings.py	✅ Working	Pydantic settings loading from .env
src/utils/ollama_client.py	✅ Working	Async Ollama wrapper with streaming
src/database/models.py	✅ Working	SQLAlchemy models for BusinessUnit, Task, etc.
src/database/connection.py	✅ Working	Async PostgreSQL connection manager
src/master_ai/brain.py	✅ Working	MasterAI class with full implementation
src/master_ai/context.py	✅ Working	Context window builder
src/master_ai/planner.py	✅ Working	ReAct planning loop
src/master_ai/evolution.py	✅ Working	Self-modification proposals
src/master_ai/prompts.py	✅ Working	System prompts
src/agents/*.py	✅ Working	All 7 agents implemented
src/business/*.py	✅ Working	Unit, lifecycle, portfolio, playbook loader
src/api/*.py	✅ Working	FastAPI with all routes
src/utils/logging.py	✅ Working	Structlog integration
src/utils/metrics.py	✅ Working	Prometheus metrics
⚠️ Incomplete (Stub Files)
File	Status	Required Action
tests/test_master_ai.py	❌ Stub	Write unit tests
tests/test_agents.py	❌ Stub	Write agent tests
tests/test_business.py	❌ Stub	Write business tests
src/database/vector_store.py	❌ Stub	Implement Pinecone RAG
scripts/seed_data.py	❌ Stub	Create demo data seeder
🔧 Missing Features (Not in Original Scope)
CLI Interface - No command-line REPL like the old empire.js
Dashboard Frontend - React dashboard not started
Alembic Migrations - Not initialized
Production Dockerfile - Current is dev-focused
Phase 2 Tasks
1. Testing Suite (Priority: HIGH)
Implement comprehensive tests for core functionality.

tests/test_master_ai.py
import pytest
from unittest.mock import AsyncMock, patch
from src.master_ai.brain import MasterAI
@pytest.mark.asyncio
async def test_intent_classification_conversation():
    """Test that greetings are classified as conversation."""
    with patch('src.master_ai.brain.OllamaClient') as mock:
        mock.return_value.complete = AsyncMock(return_value='{"type": "conversation", "action": null, "parameters": {}}')
        ai = MasterAI()
        result = await ai._classify_intent("Hello!", "")
        assert result["type"] == "conversation"
@pytest.mark.asyncio
async def test_intent_classification_command():
    """Test that action requests are classified as commands."""
    with patch('src.master_ai.brain.OllamaClient') as mock:
        mock.return_value.complete = AsyncMock(return_value='{"type": "command", "action": "start_business", "parameters": {"niche": "pet toys"}}')
        ai = MasterAI()
        result = await ai._classify_intent("Start a pet toy dropshipping business", "")
        assert result["type"] == "command"
        assert result["action"] == "start_business"
@pytest.mark.asyncio
async def test_evolution_rate_limiting():
    """Test that evolution proposals are rate-limited."""
    ai = MasterAI()
    ai._evolution_count_this_hour = 5  # At limit
    # Should not propose when at limit
    # (implementation depends on settings.max_evolutions_per_hour)
tests/test_agents.py
import pytest
from src.agents.research import ResearchAgent
from src.agents.router import AgentRouter
@pytest.mark.asyncio
async def test_research_agent_execute():
    agent = ResearchAgent()
    result = await agent.execute({
        "input": {"type": "market_analysis", "niche": "eco-friendly products"}
    })
    assert result["success"] == True
def test_agent_router_unknown_agent():
    router = AgentRouter()
    import asyncio
    result = asyncio.run(router.execute({"agent": "nonexistent"}))
    assert result["success"] == False
    assert "Unknown agent" in result["error"]
2. CLI Interface (Priority: HIGH)
Create a command-line REPL for direct interaction.

cli.py (New File)
"""
Command-line interface for King AI v2.
Run with: py -3 cli.py
"""
import asyncio
from src.master_ai.brain import MasterAI
from src.database.connection import init_db
async def main():
    print("🤴 King AI v2 - Autonomous Business Empire")
    print("=" * 50)
    
    await init_db()
    ai = MasterAI()
    
    print("Type 'quit' to exit, 'auto' to toggle autonomous mode\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            if user_input.lower() == 'quit':
                break
            if user_input.lower() == 'auto':
                ai.autonomous_mode = not ai.autonomous_mode
                print(f"Autonomous mode: {'ON' if ai.autonomous_mode else 'OFF'}")
                continue
            
            result = await ai.process_input(user_input)
            print(f"\n👑 King AI: {result['response']}\n")
            
            if result.get('actions_taken'):
                print("Actions taken:", result['actions_taken'])
            if result.get('pending_approvals'):
                print("⚠️  Pending approvals:", result['pending_approvals'])
                
        except KeyboardInterrupt:
            break
    
    print("\nGoodbye!")
if __name__ == "__main__":
    asyncio.run(main())
3. Pinecone Vector Store (Priority: MEDIUM)
src/database/vector_store.py
"""
Pinecone integration for RAG (Retrieval Augmented Generation).
Stores business data embeddings for semantic search.
"""
from pinecone import Pinecone
from config.settings import settings
class VectorStore:
    def __init__(self):
        if settings.pinecone_api_key:
            self.pc = Pinecone(api_key=settings.pinecone_api_key)
            self.index = self.pc.Index(settings.pinecone_index)
        else:
            self.pc = None
            self.index = None
    
    async def upsert_business(self, business_id: str, text: str, embedding: list[float]):
        """Store a business summary embedding."""
        if not self.index:
            return
        self.index.upsert(vectors=[{
            "id": business_id,
            "values": embedding,
            "metadata": {"text": text}
        }])
    
    async def search(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        """Find similar business contexts."""
        if not self.index:
            return []
        results = self.index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
        return results.matches
4. Demo Data Seeder (Priority: LOW)
scripts/seed_data.py
"""Seeds demo business data for testing."""
import asyncio
from src.database.connection import init_db, get_db
from src.database.models import BusinessUnit, BusinessStatus
from uuid import uuid4
DEMO_BUSINESSES = [
    {"name": "PetPal Dropshipping", "type": "dropshipping", "revenue": 15000, "expenses": 8000},
    {"name": "DevTools SaaS", "type": "saas", "revenue": 5000, "expenses": 1500},
    {"name": "EcoGadgets Store", "type": "dropshipping", "revenue": 8500, "expenses": 5200},
]
async def seed():
    await init_db()
    async with get_db() as db:
        for biz in DEMO_BUSINESSES:
            unit = BusinessUnit(
                id=str(uuid4()),
                name=biz["name"],
                type=biz["type"],
                status=BusinessStatus.OPERATION,
                total_revenue=biz["revenue"],
                total_expenses=biz["expenses"],
            )
            db.add(unit)
        await db.commit()
    print(f"Seeded {len(DEMO_BUSINESSES)} demo businesses")
if __name__ == "__main__":
    asyncio.run(seed())
5. Production Readiness (Priority: MEDIUM)
Initialize Alembic
cd king-ai-v2
alembic init src/database/migrations
# Edit alembic.ini to point to DATABASE_URL
alembic revision --autogenerate -m "Initial schema"
alembic upgrade head
Production Dockerfile
FROM python:3.12-slim as builder
WORKDIR /app
COPY pyproject.toml .
RUN pip install --no-cache-dir build && pip wheel -w /wheels -e .
FROM python:3.12-slim
WORKDIR /app
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl
COPY src/ src/
COPY config/ config/
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
File Summary
Priority	Task	Files
HIGH	Tests	tests/test_master_ai.py, tests/test_agents.py, tests/test_business.py
HIGH	CLI	cli.py
MEDIUM	Vector Store	src/database/vector_store.py
MEDIUM	Alembic	src/database/migrations/
LOW	Seed Data	scripts/seed_data.py
LOW	Dashboard	dashboard/ (React)
Verification Commands
After implementation, run:

# Test imports
py -3 -c "from cli import main; print('CLI OK')"
# Run tests
py -3 -m pytest tests/ -v
# Start API
py -3 -m uvicorn src.api.main:app --reload
# Test CLI
py -3 cli.py