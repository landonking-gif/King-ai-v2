#!/usr/bin/env python3
"""Fix the orchestrator main.py to use configurable LLM provider."""

filepath = "/home/ubuntu/king-ai-v3/agentic-framework-main/orchestrator/service/main.py"

with open(filepath, "r") as f:
    content = f.read()

# Find and replace the create_adapter block
old_block = '''        # Create vLLM adapter
        llm_adapter = create_adapter(
            provider="vllm",
            model=config.ollama_model if config.default_llm_provider == local else config.vllm_model,
            endpoint=config.ollama_endpoint if config.default_llm_provider == local else config.vllm_endpoint
        )'''

new_block = '''        # Create LLM adapter based on config
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
            )'''

if old_block in content:
    content = content.replace(old_block, new_block)
    print("Found and replaced the broken block")
else:
    # Try alternate pattern - the original unmodified block
    old_block2 = '''        # Create vLLM adapter
        llm_adapter = create_adapter(
            provider="vllm",
            model=config.vllm_model,
            endpoint=config.vllm_endpoint
        )'''
    
    if old_block2 in content:
        content = content.replace(old_block2, new_block)
        print("Found and replaced the original block")
    else:
        print("Could not find expected block. Searching for create_adapter...")
        # Show what's around create_adapter
        import re
        match = re.search(r'.{200}create_adapter.{200}', content, re.DOTALL)
        if match:
            print(f"Found context: {match.group()}")
        else:
            print("No create_adapter found!")

with open(filepath, "w") as f:
    f.write(content)

print("Done!")
