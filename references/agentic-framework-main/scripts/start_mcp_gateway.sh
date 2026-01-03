#!/bin/bash
# Quick start script for MCP Gateway

set -e

PROJECT_ROOT="/Users/paragpradhan/Projects/Agent framework/agent-framework"
cd "$PROJECT_ROOT"

echo "========================================="
echo "  Starting MCP Gateway"
echo "========================================="

# 1. Activate virtual environment
echo ""
echo "[1/4] Activating virtual environment..."
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "✓ Virtual environment activated"
else
    echo "✗ No .venv found. Creating one..."
    uv venv --python 3.11
    source .venv/bin/activate
    echo "✓ Virtual environment created and activated"
fi

# 2. Install dependencies
echo ""
echo "[2/4] Checking dependencies..."
if ! python -c "import fastapi" 2>/dev/null; then
    echo "Installing dependencies..."
    uv pip install -r requirements.txt
    echo "✓ Dependencies installed"
else
    echo "✓ Dependencies already installed"
fi

# 3. Check Redis (optional for Sprint 0, but recommended)
echo ""
echo "[3/4] Checking Redis..."
if redis-cli ping > /dev/null 2>&1; then
    echo "✓ Redis is running"
else
    echo "⚠ Redis is not running (rate limiting will be limited)"
    echo "  To start Redis: brew services start redis"
    echo "  Or: redis-server"
fi

# 4. Check environment variables
echo ""
echo "[4/4] Checking environment..."
if [ -z "$JWT_SECRET_KEY" ]; then
    echo "⚠ JWT_SECRET_KEY not set, using default (insecure for production)"
    export JWT_SECRET_KEY="dev-secret-key-change-in-production"
fi
echo "✓ Environment ready"

# Start the gateway
echo ""
echo "========================================="
echo "  Starting MCP Gateway on port 8080"
echo "========================================="
echo ""
echo "API Documentation: http://localhost:8080/docs"
echo "Health Check: http://localhost:8080/health"
echo ""
echo "Press Ctrl+C to stop"
echo ""

cd mcp-gateway

# Run with uvicorn directly (avoids module path issues)
uvicorn service.main:app --host 0.0.0.0 --port 8080 --reload
