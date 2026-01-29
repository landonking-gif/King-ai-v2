#!/bin/bash

echo "Restarting orchestrator with DeepSeek-R1 model..."
pkill -f 'orchestrator.*run_service'

sleep 2

cd /home/ubuntu/agentic-framework-main/orchestrator
source ../.venv/bin/activate
export PYTHONPATH="$PWD:$PWD/..:$PYTHONPATH"
export VLLM_MODEL="casperhansen/deepseek-r1-distill-qwen-7b-awq"

nohup python run_service.py > /tmp/orchestrator.log 2>&1 &

echo "Orchestrator PID: $(pgrep -f 'orchestrator.*run_service')"

sleep 3

echo ""
echo "=== Orchestrator Logs ==="
tail -20 /tmp/orchestrator.log
