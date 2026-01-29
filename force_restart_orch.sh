#!/bin/bash

echo "Force killing all orchestrator processes..."
ps aux | grep orchestrator | grep python | awk '{print $2}' | xargs -r kill -9

sleep 2

echo "Starting orchestrator..."
cd /home/ubuntu/agentic-framework-main/orchestrator
source ../.venv/bin/activate

nohup python run_service.py > /tmp/orchestrator.log 2>&1 &

sleep 5

PID=$(pgrep -f 'orchestrator.*run_service')
echo "Orchestrator PID: $PID"

if [ -n "$PID" ]; then
    echo "✓ Orchestrator running successfully"
    echo ""
    echo "=== Latest Logs ==="
    tail -20 /tmp/orchestrator.log
else
    echo "✗ Orchestrator failed to start"
    echo ""
    echo "=== Error Logs ==="
    tail -30 /tmp/orchestrator.log
fi
