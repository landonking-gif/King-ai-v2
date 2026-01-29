#!/bin/bash
pkill -f 'vllm.entrypoints.openai.api_server'
sleep 3
cd /home/ubuntu/agentic-framework-main
source .venv/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"
nohup python -m vllm.entrypoints.openai.api_server \
    --model TheBloke/deepseek-coder-33B-instruct-AWQ \
    --port 8005 \
    --host 0.0.0.0 \
    --quantization awq \
    --max-model-len 8192 \
    --dtype auto \
    --trust-remote-code \
    --gpu-memory-utilization 0.90 \
    > /tmp/vllm.log 2>&1 &
echo "vLLM started with PID $!"
echo "Waiting 90 seconds for model to load..."
sleep 90
echo "=== vLLM Logs ==="
tail -50 /tmp/vllm.log
