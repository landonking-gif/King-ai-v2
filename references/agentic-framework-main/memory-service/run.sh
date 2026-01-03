#!/bin/bash
# Memory Service Runner Script

set -e

# Change to memory-service directory
cd "$(dirname "$0")"

# Activate virtual environment if it exists
if [ -d "../../.venv" ]; then
    source ../../.venv/bin/activate
fi

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
echo "Starting Memory Service on ${MEMORY_SERVICE_HOST:-0.0.0.0}:${MEMORY_SERVICE_PORT:-8001}"
python -m uvicorn service.main:app \
    --host "${MEMORY_SERVICE_HOST:-0.0.0.0}" \
    --port "${MEMORY_SERVICE_PORT:-8001}" \
    --reload
