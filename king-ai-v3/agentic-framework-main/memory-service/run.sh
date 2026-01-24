#!/bin/bash
# Memory Service Runner Script

set -e

# Change to memory-service directory
cd "$(dirname "$0")"

# Activate virtual environment if it exists
if [ -d "/home/ubuntu/agentic-framework-main/.venv" ]; then
    echo "Activating venv..."
    source /home/ubuntu/agentic-framework-main/.venv/bin/activate
    echo "Venv activated, python: $(which python)"
else
    echo "Venv not found, exiting"
    exit 1
fi

# Set PYTHONPATH to include current and parent directory
export PYTHONPATH="$PWD:$PWD/..:$PYTHONPATH"

# Check if .env exists
if [ ! -f ".env" ] && [ -f "../.env" ]; then
    echo "Using parent .env file"
    export $(cat ../.env | grep -v '^#' | xargs)
elif [ -f ".env" ]; then
    export $(cat .env | grep -v '^#' | xargs)
else
    echo "Warning: No .env file found. Using defaults."
fi

# Run the service
echo "Starting Memory Service on ${MEMORY_SERVICE_HOST:-0.0.0.0}:${MEMORY_SERVICE_PORT:-8002}"
uvicorn service.main:app \
    --host "${MEMORY_SERVICE_HOST:-0.0.0.0}" \
    --port "${MEMORY_SERVICE_PORT:-8002}"

