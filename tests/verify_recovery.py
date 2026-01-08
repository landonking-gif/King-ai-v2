import asyncio
import json
from unittest.mock import MagicMock, AsyncMock
from src.master_ai.brain import MasterAI
from src.master_ai.models import IntentType, ActionType
from src.utils.llm_router import TaskContext

async def test_intent_recovery():
    print("Testing MasterAI intent classification recovery...")
    
    # 1. Initialize MasterAI
    brain = MasterAI()
    
    # 2. Mock _call_llm to return a refusal string (not JSON)
    refusal_response = "I can't assist with creating or implementing a real-world dropshipping business due to its potential for fraudulent activities."
    brain._call_llm = AsyncMock(return_value=refusal_response)
    
    # 3. Test research query
    user_input = "find the highest ROI dropshipping product then analyze it"
    print(f"User Input: {user_input}")
    
    intent = await brain._classify_intent(user_input, "mock context")
    
    print(f"Classified Intent Type: {intent.type}")
    print(f"Classified Action: {intent.action}")
    print(f"Reasoning: {intent.reasoning}")
    
    # Assertions
    assert intent.type == IntentType.COMMAND
    assert intent.action == ActionType.RESEARCH_MARKET
    assert "Detected likely research intent" in intent.reasoning
    
    print("\u2705 Intent recovery test passed!")

if __name__ == "__main__":
    asyncio.run(test_intent_recovery())
