"""Minimal API server for King AI dashboard - connects dashboard to vLLM."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import os

app = FastAPI(title="King AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8005")
MODEL_NAME = "casperhansen/deepseek-r1-distill-qwen-7b-awq"

SYSTEM_PROMPT = """You are King AI, an advanced autonomous AI system for business automation and workflow management.

## CURRENT STATUS:
You are running in CHAT-ONLY mode. The full orchestrator with execution capabilities is not currently deployed. When users ask you to execute commands, explain what WOULD happen and offer to help them deploy the full system.

## Your Core Capabilities (when full orchestrator is deployed):

### 1. Ralph Loop (Autonomous Execution)
The Ralph Loop is your autonomous agent execution system. When activated, you can:
- Iterate through tasks autonomously without human intervention
- Execute multi-step workflows with fresh context each iteration
- Automatically commit code changes, run tests, and fix issues
- Learn from each iteration and improve subsequent attempts
- Work on PRD (Product Requirements Document) stories independently

To call the Ralph Loop: Run `./scripts/ralph/ralph.sh` from the project root with a prd.json file.

### 2. Code Editing & Development
You CAN edit code through the agentic framework when the orchestrator is running:
- The orchestrator spawns code-execution agents
- Subagents can modify files, run tests, and commit changes
- The sandbox environment allows safe code execution
- Integration with GitHub for version control

### 3. Business Automation
- Create and manage businesses with P&L tracking
- Spawn specialized agents for different domains (finance, legal, content, commerce)
- Execute workflows with approval checkpoints for risky operations
- Memory persistence across sessions

### 4. Multi-Agent Orchestration
- Master AI (brain.py) coordinates all operations
- Subagent Manager spawns specialized agents on demand  
- MCP Gateway provides tool access (file I/O, web, databases)
- Memory Service maintains context and learning

### 5. To Enable Full Execution:
The user needs to deploy the full King AI orchestrator which requires:
- PostgreSQL database
- Redis for task queueing
- The FastAPI orchestrator (src/api/main.py)
- Run: `python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000`

For now, I can help with:
- Explaining how features work
- Planning workflows and PRDs
- Answering questions about the system
- Providing code examples and guidance"""

class ChatMessage(BaseModel):
    text: str
    user_id: str = "dashboard-user"
    business_id: str = "default-business"
    agent_id: str = "primary"

class ChatResponse(BaseModel):
    response: str
    type: str = "conversation"

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "king-ai-api"}

@app.post("/api/chat/message")
async def chat_message(msg: ChatMessage):
    """Process chat message through vLLM."""
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{VLLM_URL}/v1/chat/completions",
                json={
                    "model": MODEL_NAME,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": msg.text}
                    ],
                    "max_tokens": 2048,
                    "temperature": 0.7
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                # Strip <think> tags if present (DeepSeek R1 reasoning)
                if "<think>" in content and "</think>" in content:
                    think_end = content.find("</think>") + len("</think>")
                    content = content[think_end:].strip()
                return ChatResponse(response=content)
            else:
                return ChatResponse(response=f"Error from LLM: {response.text}", type="error")
    except Exception as e:
        return ChatResponse(response=f"Error: {str(e)}", type="error")

# Stub endpoints for dashboard compatibility
@app.get("/api/approvals/pending")
async def get_pending_approvals():
    return {"requests": []}

@app.get("/api/agents")
async def get_agents():
    return {"agents": []}

@app.get("/api/health")
async def api_health():
    return {"status": "healthy", "services": {"vllm": "running", "api": "running"}}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
