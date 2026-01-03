#!/usr/bin/env python3
"""
Test which model is actually being used from .env
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from parent directory
parent_env = Path(__file__).parent.parent.parent / ".env"
if parent_env.exists():
    load_dotenv(parent_env)
else:
    load_dotenv()

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from kautilya.llm_client import KautilyaLLMClient

print("=" * 80)
print("MODEL CONFIGURATION TEST")
print("=" * 80)

# Check environment variables
print("\n1. Environment Variables:")
print(f"   OPENAI_MODEL = {os.getenv('OPENAI_MODEL', 'NOT SET')}")
print(f"   OPENAI_API_KEY = {'SET (' + str(len(os.getenv('OPENAI_API_KEY', ''))) + ' chars)' if os.getenv('OPENAI_API_KEY') else 'NOT SET'}")

# Initialize client (no model parameter - should use env var)
print("\n2. Initializing LLM Client (no model parameter)...")
client = KautilyaLLMClient()

print(f"   ✓ Client model: {client.model}")

# Check if it matches .env
expected_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
if client.model == expected_model:
    print(f"   ✓ Using model from .env: {expected_model}")
else:
    print(f"   ⚠️  Expected {expected_model}, but using {client.model}")

# Test with explicit model parameter
print("\n3. Testing with explicit model parameter...")
client2 = KautilyaLLMClient(model="gpt-4o")
print(f"   ✓ Explicit model: {client2.model}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Default model from .env: {expected_model}")
print(f"Client is using: {client.model}")
print(f"Match: {'✓ YES' if client.model == expected_model else '✗ NO'}")
print("=" * 80)
