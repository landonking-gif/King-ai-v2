#!/bin/bash
# Run Subagent Manager service

set -e

# Change to project root
cd "$(dirname "$0")/.."

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv --python 3.11
fi

# Activate virtual environment
source .venv/bin/activate

# Install dependencies if needed
if [ ! -f ".venv/installed" ]; then
    echo "Installing dependencies..."
    uv pip install -r requirements.txt
    touch .venv/installed
fi

# Set PYTHONPATH to include project root
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# Load environment variables if .env exists
if [ -f "subagent-manager/service/.env" ]; then
    export $(cat subagent-manager/service/.env | grep -v '^#' | xargs)
fi

# Create schema registry directory if it doesn't exist
mkdir -p docs/schema_registry

# Run the service
echo "Starting Subagent Manager service..."
python -m subagent_manager.service.main
