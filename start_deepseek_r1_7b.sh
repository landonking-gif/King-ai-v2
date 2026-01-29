#!/bin/bash

echo "Stopping current vLLM process..."
pkill -f 'vllm.entrypoints.openai.api_server'

sleep 3

echo "Starting DeepSeek-R1-Distill-Qwen-7B (AWQ, fits on A10G)..."
cd /home/ubuntu/agentic-framework-main
source .venv/bin/activate

nohup python -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --port 8005 \
    --model casperhansen/deepseek-r1-distill-qwen-7b-awq \
    --trust-remote-code \
    --quantization awq \
    --dtype float16 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 4096 \
    > /tmp/vllm.log 2>&1 &

VLLM_PID=$!
echo "vLLM started with PID: $VLLM_PID"

echo "Waiting for model to load (7B model takes ~90 seconds)..."
sleep 100

echo ""
echo "=== Latest vLLM Logs ==="
tail -n 40 /tmp/vllm.log

echo ""
echo "=== Checking if vLLM is responding ==="
curl -s http://localhost:8005/v1/models || echo "vLLM not responding yet"
