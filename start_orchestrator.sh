#!/bin/bash
cd /home/ubuntu/agentic-framework-main/orchestrator
source ../.venv/bin/activate
export PYTHONPATH="/home/ubuntu/agentic-framework-main:/home/ubuntu:$PYTHONPATH"
export AWS_IP="52.90.206.76"
pkill -f "python.*orchestrator" || true
sleep 1
nohup python run_service.py > /tmp/orchestrator.log 2>&1 &
echo $! > /tmp/orchestrator.pid
echo "Orchestrator started with PID: $(cat /tmp/orchestrator.pid)"
sleep 2
ps aux | grep orchestrator | grep -v grep
