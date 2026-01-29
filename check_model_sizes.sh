#!/bin/bash
cd /home/ubuntu/agentic-framework-main
source .venv/bin/activate
python3 -c "
from huggingface_hub import HfApi
import json
api = HfApi()

# Check GGUF versions for quantization info
gguf_models = [
    'DevQuasar/moonshotai.Kimi-K2-Instruct-GGUF',
    'bartowski/moonshotai_Kimi-K2-Instruct-0905-GGUF'
]

print('GGUF Model Details:')
for model_id in gguf_models:
    try:
        files = list(api.list_repo_files(model_id, limit=20))
        gguf_files = [f for f in files if f.endswith('.gguf')]
        print(f'\\n{model_id}:')
        for f in gguf_files[:5]:  # Show first 5 GGUF files
            print(f'  - {f}')
    except Exception as e:
        print(f'  Error: {e}')

# Check model sizes
print('\\nModel Size Estimates:')
models_to_check = [
    'moonshotai/Kimi-K2-Thinking',
    'moonshotai/Kimi-K2-Instruct',
    'DevQuasar/moonshotai.Kimi-K2-Instruct-GGUF'
]

for model_id in models_to_check:
    try:
        info = api.model_info(model_id)
        if hasattr(info, 'safetensors') and info.safetensors:
            total_size = sum(f.size for f in info.safetensors)
            print(f'{model_id}: ~{total_size / (1024**3):.1f} GB')
        elif hasattr(info, 'siblings'):
            total_size = sum(f.size for f in info.siblings if f.size)
            print(f'{model_id}: ~{total_size / (1024**3):.1f} GB')
    except Exception as e:
        print(f'{model_id}: Error getting size - {e}')
"