#!/usr/bin/env python3
"""Test vLLM directly"""
import requests
import json

# Test direct vLLM call
r = requests.post(
    "http://localhost:8005/v1/chat/completions",
    json={
        "model": "casperhansen/deepseek-r1-distill-qwen-7b-awq",
        "messages": [{"role": "user", "content": "What is 2+2?"}],
        "max_tokens": 100
    }
)
print(f"Status: {r.status_code}")
print(f"Response: {json.dumps(r.json(), indent=2)[:500]}")
