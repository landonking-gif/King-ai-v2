#!/bin/bash

echo "Stopping current vLLM process..."
pkill -f 'vllm.entrypoints.openai.api_server'

sleep 3

echo "Starting DeepSeek V3.2 AWQ with float16..."
cd /home/ubuntu/agentic-framework-main
source .venv/bin/activate

nohup python -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --port 8005 \
    --model QuantTrio/DeepSeek-V3.2-AWQ \
    --trust-remote-code \
    --quantization awq \
    --dtype float16 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    > /tmp/vllm.log 2>&1 &

VLLM_PID=$!
echo "vLLM started with PID: $VLLM_PID"

echo "Waiting for model to load (this may take 2-3 minutes)..."
sleep 120

echo ""
echo "=== Latest vLLM Logs ==="
tail -n 30 /tmp/vllm.log

echo ""
echo "=== Checking if vLLM is responding ==="
curl -s http://localhost:8005/v1/models || echo "vLLM not responding yet, check logs with: ssh ubuntu@52.90.242.99 'tail -f /tmp/vllm.log'"
