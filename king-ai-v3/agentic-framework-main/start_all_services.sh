#!/bin/bash
# Start all services script

set -e

cd ~/agentic-framework-main
source .venv/bin/activate
export PYTHONPATH="$HOME/agentic-framework-main:$PYTHONPATH"

# Kill any existing service processes
echo "Stopping existing services..."
pkill -f "python.*service.main" || true
pkill -f "uvicorn.*service.main:app" || true
pkill -f "npm run preview" || true
sleep 3

echo "Starting services..."

# Start MCP Gateway (port 8080)
cd ~/agentic-framework-main/mcp-gateway
python -m service.main > /tmp/mcp-gateway.log 2>&1 &
echo $! > /tmp/mcp-gateway.pid
echo "Started MCP Gateway (PID: $(cat /tmp/mcp-gateway.pid))"

# Start Memory Service (port 8002)
cd ~/agentic-framework-main/memory-service
uvicorn service.main:app --host 0.0.0.0 --port 8002 > /tmp/memory-service.log 2>&1 &
echo $! > /tmp/memory-service.pid
echo "Started Memory Service (PID: $(cat /tmp/memory-service.pid))"

# Start Subagent Manager (port 8001)
cd ~/agentic-framework-main/subagent-manager
uvicorn service.main:app --host 0.0.0.0 --port 8001 > /tmp/subagent-manager.log 2>&1 &
echo $! > /tmp/subagent-manager.pid
echo "Started Subagent Manager (PID: $(cat /tmp/subagent-manager.pid))"

# Start Dashboard (port 3000)
cd ~/agentic-framework-main/dashboard
npm run preview -- --host 0.0.0.0 --port 3000 > /tmp/dashboard.log 2>&1 &
echo $! > /tmp/dashboard.pid
echo "Started Dashboard (PID: $(cat /tmp/dashboard.pid))"

# Start Orchestrator (port 8000)
cd ~/agentic-framework-main/orchestrator
python -m service.main --host 0.0.0.0 --port 8000 > /tmp/orchestrator.log 2>&1 &
echo $! > /tmp/orchestrator.pid
echo "Started Orchestrator (PID: $(cat /tmp/orchestrator.pid))"

echo ""
echo "Waiting 10 seconds for services to start..."
sleep 10

echo ""
echo "Service Status:"
echo "==============="

for service in "MCP Gateway:8080" "Memory Service:8002" "Subagent Manager:8001" "Dashboard:3000" "Orchestrator:8000"; do
    name=${service%:*}
    port=${service#*:}
    if curl -sf http://localhost:$port/health > /dev/null 2>&1 || curl -sf http://localhost:$port > /dev/null 2>&1; then
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
