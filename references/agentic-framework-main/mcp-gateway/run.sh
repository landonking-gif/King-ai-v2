#!/bin/bash
# Startup script for MCP Gateway service

set -e

# Activate virtual environment if exists
if [ -d "../../.venv" ]; then
    source ../../.venv/bin/activate
fi

# Check if Redis is running
if ! redis-cli ping > /dev/null 2>&1; then
    echo "WARNING: Redis is not running. Rate limiting will fail."
    echo "Start Redis with: redis-server"
fi

# Run the service
echo "Starting MCP Gateway on port 8080..."
python -m mcp_gateway.service.main
