#!/bin/bash
#
# Run Code Executor Service
#

set -e

# Change to script directory
cd "$(dirname "$0")"

# Activate virtual environment if it exists
if [ -d "../.venv" ]; then
    echo "Activating virtual environment..."
    source ../.venv/bin/activate
fi

# Load environment variables
if [ -f ".env" ]; then
    echo "Loading environment from .env..."
    export $(cat .env | grep -v '^#' | xargs)
fi

# Run service
echo "Starting Code Executor Service..."
echo "Service will be available at http://localhost:8002"
echo "API docs at http://localhost:8002/docs"
echo ""

uvicorn service.main:app \
    --host "${CODE_EXEC_HOST:-0.0.0.0}" \
    --port "${CODE_EXEC_PORT:-8002}" \
    --reload \
    --log-level "${CODE_EXEC_LOG_LEVEL:-info}"
