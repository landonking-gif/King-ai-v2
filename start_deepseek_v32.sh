#!/bin/bash
echo "Stopping current vLLM process..."
pkill -f 'vllm.entrypoints.openai.api_server'
sleep 3

echo "Starting DeepSeek V3.2 AWQ..."
cd /home/ubuntu/agentic-framework-main
source .venv/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"

nohup python -m vllm.entrypoints.openai.api_server \
    --model QuantTrio/DeepSeek-V3.2-AWQ \
    --port 8005 \
    --host 0.0.0.0 \
    --quantization awq \
    --max-model-len 8192 \
    --dtype auto \
    --trust-remote-code \
    --gpu-memory-utilization 0.90 \
    > /tmp/vllm.log 2>&1 &

VLLM_PID=$!
echo "vLLM started with PID: $VLLM_PID"
echo $VLLM_PID > /tmp/vllm.pid

echo "Waiting for model to load (this may take 2-3 minutes)..."
sleep 120

echo ""
echo "=== Latest vLLM Logs ==="
tail -50 /tmp/vllm.log

echo ""
echo "=== Checking if vLLM is responding ==="
curl -s http://localhost:8005/v1/models | head -20
