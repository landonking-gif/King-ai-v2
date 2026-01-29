#!/bin/bash
# Start vLLM via Docker with GPU support
docker stop vllm 2>/dev/null
docker rm vllm 2>/dev/null
sleep 2

echo "Starting vLLM in Docker..."
docker run -d --gpus all \
  --name vllm \
  -p 8005:8000 \
  --ipc=host \
  -e HF_TOKEN=${HF_TOKEN:-} \
  vllm/vllm-openai:latest \
  --model casperhansen/deepseek-r1-distill-qwen-7b-awq \
  --dtype float16 \
  --api-key token-abc123 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096

sleep 5
docker logs vllm --tail 20
echo ""
echo "vLLM container started. Check logs with: docker logs -f vllm"
echo "API available at: http://localhost:8005"
