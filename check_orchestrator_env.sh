#!/bin/bash

echo "Checking orchestrator environment..."
PID=$(pgrep -f 'orchestrator.*run_service')
echo "Orchestrator PID: $PID"
echo ""
echo "VLLM-related environment variables:"
cat /proc/$PID/environ | tr '\0' '\n' | grep VLLM
echo ""
echo "Model configuration in code:"
grep -A2 "vllm_model" /home/ubuntu/agentic-framework-main/orchestrator/service/config.py
