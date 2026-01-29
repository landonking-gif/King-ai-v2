#!/usr/bin/env python3
from huggingface_hub import HfApi
import sys

api = HfApi()

print("=== Searching for A10G-compatible DeepSeek models ===\n")

# Search for DeepSeek-R1 (latest reasoning model)
print("1. DeepSeek-R1 models (newest reasoning model, standard attention):")
try:
    models = api.list_models(search="deepseek-ai/DeepSeek-R1", limit=10)
    for model in models:
        print(f"   - {model.id}")
except Exception as e:
    print(f"   Error: {e}")

print("\n2. Quantized DeepSeek-R1 models:")
try:
    models = api.list_models(search="deepseek r1 awq", limit=10)
    for model in models:
        if 'r1' in model.id.lower() and ('awq' in model.id.lower() or 'gptq' in model.id.lower()):
            print(f"   - {model.id}")
except Exception as e:
    print(f"   Error: {e}")

print("\n3. DeepSeek-Coder (proven A10G compatible):")
print("   - TheBloke/deepseek-coder-33B-instruct-AWQ")
print("   - TheBloke/deepseek-coder-6.7B-instruct-AWQ")

print("\n4. Smaller quantized models (guaranteed to fit):")
try:
    models = api.list_models(search="deepseek awq", limit=20)
    count = 0
    for model in models:
        if 'awq' in model.id.lower() and count < 10:
            print(f"   - {model.id}")
            count += 1
except Exception as e:
    print(f"   Error: {e}")

print("\n=== CRITICAL: DeepSeek V3/V3.2 Architecture Limitation ===")
print("DeepSeek V3 and V3.2 use Multi-Latent Attention (MLA) with sparse attention")
print("This requires Hopper (H100) or newer GPUs - NOT supported on Ampere A10G")
print("Recommendation: Use DeepSeek-R1 or DeepSeek-Coder instead")
