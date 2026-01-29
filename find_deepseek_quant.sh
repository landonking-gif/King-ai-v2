#!/bin/bash
cd /home/ubuntu/agentic-framework-main
source .venv/bin/activate
python3 << 'EOF'
from huggingface_hub import HfApi
api = HfApi()

print("Searching for DeepSeek AWQ quantized models...")
models = list(api.list_models(search="deepseek awq", limit=20))
for m in models:
    if 'awq' in m.id.lower() or 'AWQ' in str(m.tags):
        print(f"\n{m.id}")
        if hasattr(m, 'tags'):
            size_tags = [t for t in m.tags if any(x in t.lower() for x in ['32b', '33b', '7b', '14b'])]
            if size_tags:
                print(f"  Size: {size_tags}")

print("\n\nSearching for DeepSeek GPTQ models...")
models = list(api.list_models(search="deepseek gptq", limit=20))
for m in models:
    if 'gptq' in m.id.lower() or 'GPTQ' in str(m.tags):
        print(f"\n{m.id}")
        if hasattr(m, 'tags'):
            size_tags = [t for t in m.tags if any(x in t.lower() for x in ['32b', '33b', '7b', '14b'])]
            if size_tags:
                print(f"  Size: {size_tags}")
EOF
