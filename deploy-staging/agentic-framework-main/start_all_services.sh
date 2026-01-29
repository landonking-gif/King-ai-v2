#!/bin/bash
# =============================================================================
# King AI v3 - Start All Services
# =============================================================================
# Starts all agentic framework services in the correct order.
# Run from the agentic-framework-main directory.
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo -e "${GREEN}King AI v3 - Starting Services${NC}"
echo "================================"

# Activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
fi

export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# Load environment variables
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Kill any existing service processes
echo -e "${YELLOW}Stopping existing services...${NC}"
pkill -f "python.*service.main" 2>/dev/null || true
pkill -f "uvicorn.*service.main:app" 2>/dev/null || true
pkill -f "uvicorn.*main:app.*8000" 2>/dev/null || true
pkill -f "uvicorn.*main:app.*8001" 2>/dev/null || true
pkill -f "uvicorn.*main:app.*8002" 2>/dev/null || true
pkill -f "uvicorn.*main:app.*3000" 2>/dev/null || true
pkill -f "npm run preview" 2>/dev/null || true
sleep 3

echo -e "${GREEN}Starting services...${NC}"

# Start Memory Service (port 8002) - First, as others may depend on it
echo "  Starting Memory Service (port 8002)..."
cd "$SCRIPT_DIR/memory-service"
nohup python3 -m uvicorn service.main:app --host 0.0.0.0 --port 8002 > /tmp/memory-service.log 2>&1 &
echo $! > /tmp/memory-service.pid
sleep 2

# Start MCP Gateway (port 8080)
echo "  Starting MCP Gateway (port 8080)..."
cd "$SCRIPT_DIR/mcp-gateway"
if [ -f "service/main.py" ]; then
    nohup python3 -m uvicorn service.main:app --host 0.0.0.0 --port 8080 > /tmp/mcp-gateway.log 2>&1 &
    echo $! > /tmp/mcp-gateway.pid
else
    echo "    (MCP Gateway service not found, skipping)"
fi
sleep 2

# Start Subagent Manager (port 8001)
echo "  Starting Subagent Manager (port 8001)..."
cd "$SCRIPT_DIR/subagent-manager"
if [ -f "service/main.py" ]; then
    nohup python3 -m uvicorn service.main:app --host 0.0.0.0 --port 8001 > /tmp/subagent-manager.log 2>&1 &
    echo $! > /tmp/subagent-manager.pid
else
    echo "    (Subagent Manager service not found, skipping)"
fi
sleep 2

# Start Orchestrator (port 8000) - Main API
echo "  Starting Orchestrator (port 8000)..."
cd "$SCRIPT_DIR/orchestrator"
nohup python3 -m uvicorn service.main:app --host 0.0.0.0 --port 8000 > /tmp/orchestrator.log 2>&1 &
echo $! > /tmp/orchestrator.pid
sleep 2

# Start Control Panel (port 3000)
echo "  Starting Control Panel (port 3000)..."
cd "$SCRIPT_DIR/control-panel"
nohup python3 -m uvicorn main:app --host 0.0.0.0 --port 3000 > /tmp/control-panel.log 2>&1 &
echo $! > /tmp/control-panel.pid
sleep 2

echo ""
echo "Service Status:"
echo "==============="

for service in "MCP Gateway:8080" "Memory Service:8002" "Subagent Manager:8001" "Dashboard:3000" "Orchestrator:8000" "vLLM:8005"; do
    name=${service%:*}
    port=${service#*:}
    if [ "$name" = "vLLM" ]; then
        if curl -sf http://localhost:$port/v1/models > /dev/null 2>&1; then
            echo "✓ $name (port $port) - HEALTHY"
        else
            echo "✗ $name (port $port) - FAILED"
        fi
    elif curl -sf http://localhost:$port/health > /dev/null 2>&1 || curl -sf http://localhost:$port > /dev/null 2>&1; then
        echo "✓ $name (port $port) - HEALTHY"
    else
        echo "✗ $name (port $port) - FAILED"
        if [ -f "/tmp/${name// /-}.log" ]; then
            echo "  Last 5 lines of log:"
            tail -5 "/tmp/$(echo $name | tr ' ' '-' | tr '[:upper:]' '[:lower:]').log" | sed 's/^/    /'
        fi
    fi
done

echo ""
echo "Services started. Logs are in /tmp/"
echo "To view logs: tail -f /tmp/mcp-gateway.log"
