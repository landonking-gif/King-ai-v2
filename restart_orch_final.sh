#!/bin/bash

echo "Restarting orchestrator with corrected config..."
pkill -f 'orchestrator.*run_service'

sleep 2

cd /home/ubuntu/agentic-framework-main/orchestrator
source ../.venv/bin/activate
export PYTHONPATH="$PWD:$PWD/..:$PYTHONPATH"
export VLLM_MODEL="casperhansen/deepseek-r1-distill-qwen-7b-awq"

nohup python run_service.py > /tmp/orchestrator.log 2>&1 &

sleep 4

echo ""
echo "=== Orchestrator Config Check ==="
grep -A2 "vllm_model" /home/ubuntu/agentic-framework-main/orchestrator/service/config.py

echo ""
echo "=== Orchestrator Logs (last 10 lines) ==="
tail -10 /tmp/orchestrator.log

echo ""
echo "=== Orchestrator PID ==="
pgrep -f 'orchestrator.*run_service'
