#!/bin/bash
# Startup script for Subagent Manager service

set -e

echo "Starting Subagent Manager script"

# Activate virtual environment if exists
if [ -d "/home/ubuntu/agentic-framework-main/.venv" ]; then
    echo "Activating venv..."
    source /home/ubuntu/agentic-framework-main/.venv/bin/activate
    echo "Venv activated, python: $(which python)"
else
    echo "Venv not found"
    exit 1
fi

# Set PYTHONPATH to include the project root
export PYTHONPATH="$PWD:$PWD/../..:$PYTHONPATH"
echo "PYTHONPATH: $PYTHONPATH"

# Run the service
echo "Starting Subagent Manager on port 8001..."
uvicorn subagent_manager.service.main:app --host 0.0.0.0 --port 8001