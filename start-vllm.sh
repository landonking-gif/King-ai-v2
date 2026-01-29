#!/bin/bash
# Start vLLM service
pkill -f vllm
sleep 2
nohup python3 -m vllm.entrypoints.openai.api_server \
  --model casperhansen/deepseek-r1-distill-qwen-7b-awq \
  --dtype float16 \
  --api-key token-abc123 \
  --port 8005 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 \
  > /tmp/vllm.log 2>&1 &
echo $! > /tmp/vllm.pid
sleep 5
ps aux | grep vllm | grep -v grep
echo "vLLM started, check /tmp/vllm.log for details"
