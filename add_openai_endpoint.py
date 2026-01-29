#!/usr/bin/env python3
"""
Add OpenAI-compatible /v1/chat/completions endpoint to the orchestrator.
This allows MoltBot and other OpenAI-compatible clients to use King AI.
"""

import re

# Read the original file
with open('/home/ubuntu/king-ai-v3/agentic-framework-main/orchestrator/service/main.py', 'r') as f:
    content = f.read()

# OpenAI-compatible endpoint code
openai_endpoint_code = '''

# ============================================================================
# OpenAI-Compatible API Endpoints (for MoltBot and other clients)
# ============================================================================

from pydantic import BaseModel, Field
from typing import List, Optional, Union
import time


class OpenAIChatMessage(BaseModel):
    """OpenAI-compatible message format."""
    role: str = Field(..., description="Role: system, user, or assistant")
    content: str = Field(..., description="Message content")


class OpenAIChatRequest(BaseModel):
    """OpenAI-compatible chat completion request."""
    model: str = Field(default="deepseek-r1", description="Model to use")
    messages: List[OpenAIChatMessage] = Field(..., description="List of messages")
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: Optional[int] = Field(default=2048, ge=1)
    stream: bool = Field(default=False, description="Whether to stream the response")
    top_p: Optional[float] = Field(default=1.0, ge=0, le=1)
    frequency_penalty: Optional[float] = Field(default=0, ge=-2, le=2)
    presence_penalty: Optional[float] = Field(default=0, ge=-2, le=2)


class OpenAIChatChoice(BaseModel):
    """OpenAI-compatible choice in response."""
    index: int
    message: OpenAIChatMessage
    finish_reason: str = "stop"


class OpenAIUsage(BaseModel):
    """OpenAI-compatible usage statistics."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class OpenAIChatResponse(BaseModel):
    """OpenAI-compatible chat completion response."""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[OpenAIChatChoice]
    usage: OpenAIUsage


class OpenAIModelInfo(BaseModel):
    """OpenAI-compatible model info."""
    id: str
    object: str = "model"
    created: int
    owned_by: str = "king-ai"


class OpenAIModelList(BaseModel):
    """OpenAI-compatible model list."""
    object: str = "list"
    data: List[OpenAIModelInfo]


@app.post("/v1/chat/completions", response_model=OpenAIChatResponse)
async def openai_chat_completions(request: OpenAIChatRequest):
    """
    OpenAI-compatible chat completions endpoint.
    
    This endpoint provides compatibility with MoltBot and other OpenAI-compatible clients.
    It proxies requests to the configured LLM backend (Ollama/DeepSeek).
    """
    logger.info(f"OpenAI-compatible request: model={request.model}, messages={len(request.messages)}")
    
    try:
        # Create LLM adapter based on config
        if config.default_llm_provider == "local":
            llm_adapter = create_adapter(
                provider="local",
                model=config.ollama_model,
                endpoint=config.ollama_endpoint
            )
        else:
            llm_adapter = create_adapter(
                provider="vllm",
                model=config.vllm_model,
                endpoint=config.vllm_endpoint
            )
        
        # Convert OpenAI messages to internal format
        llm_messages = []
        for msg in request.messages:
            role_map = {
                "system": MessageRole.SYSTEM,
                "user": MessageRole.USER,
                "assistant": MessageRole.ASSISTANT
            }
            llm_messages.append(LLMMessage(
                role=role_map.get(msg.role, MessageRole.USER),
                content=msg.content
            ))
        
        # Add King AI system prompt if not present
        has_system = any(msg.role == "system" for msg in request.messages)
        if not has_system:
            llm_messages.insert(0, LLMMessage(
                role=MessageRole.SYSTEM,
                content=(
                    "You are King AI, an intelligent orchestration assistant. "
                    "You help users with various tasks, providing helpful, accurate, and concise responses."
                )
            ))
        
        # Get LLM response
        response = await llm_adapter.complete(
            messages=llm_messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens or 2048
        )
        
        response_text = response.content
        
        # Estimate token counts (rough approximation)
        prompt_tokens = sum(len(msg.content.split()) * 4 // 3 for msg in request.messages)
        completion_tokens = len(response_text.split()) * 4 // 3
        
        # Build OpenAI-compatible response
        return OpenAIChatResponse(
            id=f"chatcmpl-{uuid4().hex[:24]}",
            created=int(time.time()),
            model=request.model,
            choices=[
                OpenAIChatChoice(
                    index=0,
                    message=OpenAIChatMessage(
                        role="assistant",
                        content=response_text
                    ),
                    finish_reason="stop"
                )
            ],
            usage=OpenAIUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
        )
        
    except Exception as e:
        logger.error(f"OpenAI-compatible endpoint error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"LLM request failed: {str(e)}"
        )


@app.get("/v1/models", response_model=OpenAIModelList)
async def list_openai_models():
    """
    List available models in OpenAI-compatible format.
    """
    models = [
        OpenAIModelInfo(
            id="deepseek-r1",
            created=int(time.time()),
            owned_by="king-ai"
        ),
        OpenAIModelInfo(
            id="deepseek-r1:7b",
            created=int(time.time()),
            owned_by="ollama"
        ),
        OpenAIModelInfo(
            id="llama3.2:3b",
            created=int(time.time()),
            owned_by="ollama"
        )
    ]
    return OpenAIModelList(data=models)


@app.get("/v1/models/{model_id}")
async def get_openai_model(model_id: str):
    """
    Get model info in OpenAI-compatible format.
    """
    return OpenAIModelInfo(
        id=model_id,
        created=int(time.time()),
        owned_by="king-ai"
    )

'''

# Find the position to insert (before the main() function definition)
insert_position = content.find('\ndef main() -> None:')

if insert_position == -1:
    # Try alternative pattern
    insert_position = content.find('\nif __name__ == "__main__":')

if insert_position == -1:
    print("ERROR: Could not find insertion point!")
    exit(1)

# Insert the OpenAI-compatible endpoints
new_content = content[:insert_position] + openai_endpoint_code + content[insert_position:]

# Write the updated file
with open('/home/ubuntu/king-ai-v3/agentic-framework-main/orchestrator/service/main.py', 'w') as f:
    f.write(new_content)

print("SUCCESS: Added OpenAI-compatible endpoints to orchestrator!")
print("Endpoints added:")
print("  - POST /v1/chat/completions")
print("  - GET /v1/models")
print("  - GET /v1/models/{model_id}")
