#!/bin/bash
cd /home/ubuntu/agentic-framework-main
source .venv/bin/activate
python3 << 'EOF'
from huggingface_hub import HfApi
api = HfApi()

print("Searching for DeepSeek V3/V3.2 quantized models...")
searches = [
    "deepseek v3 awq",
    "deepseek v3 gptq", 
    "deepseek v3.2 awq",
    "deepseek v3.2 gptq",
    "deepseek v3 4bit",
    "deepseek-v3"
]

found_models = set()
for search_term in searches:
    try:
        models = list(api.list_models(search=search_term, limit=30))
        for m in models:
            model_id = m.id.lower()
            if ('deepseek' in model_id and 'v3' in model_id) or 'deepseek-v3' in model_id:
                if any(q in model_id for q in ['awq', 'gptq', 'gguf', '4bit', 'int4']):
                    found_models.add(m.id)
    except:
        pass

print("\nQuantized DeepSeek V3/V3.2 models found:")
for model in sorted(found_models):
    print(f"  - {model}")

# Also check for official deepseek-ai models
print("\n\nOfficial deepseek-ai V3 models:")
models = list(api.list_models(author="deepseek-ai", limit=50))
for m in models:
    if 'v3' in m.id.lower() or 'DeepSeek-V3' in m.id:
        print(f"  - {m.id}")
        if hasattr(m, 'tags'):
            quant_tags = [t for t in m.tags if any(q in t.lower() for q in ['awq', 'gptq', '4bit', 'int4', 'quantized'])]
            if quant_tags:
                print(f"    Quantization: {quant_tags}")
EOF
