#!/bin/bash
# Simple startup script for MCP Gateway

set -e

echo "Starting MCP Gateway..."

# Set default JWT secret if not set
if [ -z "$JWT_SECRET_KEY" ]; then
    export JWT_SECRET_KEY="dev-secret-key-change-in-production"
fi

# Activate venv if exists
if [ -d "../.venv/bin" ]; then
    source ../.venv/bin/activate
elif [ -d ".venv/bin" ]; then
    source .venv/bin/activate
fi

# Start the gateway
echo ""
echo "MCP Gateway starting on http://localhost:8080"
echo "API Docs: http://localhost:8080/docs"
echo ""
echo "Press Ctrl+C to stop"
echo ""

uvicorn service.main:app --host 0.0.0.0 --port 8080 --reload
