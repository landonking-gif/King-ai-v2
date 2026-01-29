#!/bin/bash
cd /home/ubuntu/agentic-framework-main
source .venv/bin/activate
python3 -c "
from huggingface_hub import HfApi
import json
api = HfApi()
models = list(api.list_models(search='moonshotai/Kimi-K2', limit=10))
print('Available Kimi-K2 models:')
for m in models:
    print(f'- {m.id}')
    if hasattr(m, 'tags') and m.tags:
        print(f'  Tags: {m.tags[:5]}')  # Show first 5 tags
    print()
"