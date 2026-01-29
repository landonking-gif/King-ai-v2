#!/bin/bash

echo "Searching for DeepSeek models without MLA (compatible with A10G)..."
echo ""
echo "=== DeepSeek-Coder models (standard attention, A10G compatible) ==="
huggingface-cli search deepseek-ai/DeepSeek-Coder --filter modelId | grep -i "deepseek-coder"
echo ""
echo "=== Quantized DeepSeek-Coder models ===" 
huggingface-cli search deepseek-coder --filter modelId | grep -iE "awq|gptq|gguf" | head -20
echo ""
echo "=== DeepSeek-R1 models (if available) ==="
huggingface-cli search "deepseek-r1" --filter modelId | head -10
echo ""
echo "Note: DeepSeek V3/V3.2 require Hopper/Blackwell GPUs (H100/H200) due to MLA sparse attention"
echo "A10G (Ampere) only supports standard attention mechanisms"
