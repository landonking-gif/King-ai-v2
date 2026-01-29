#!/bin/bash
set -e

echo "Starting all King AI services..."

# Start Docker services
cd /home/ubuntu/agentic-framework-main
echo "Starting Docker services..."
docker compose up -d

sleep 5

# Check Docker services
echo "Docker services status:"
docker compose ps

# Start dashboard
echo "Starting dashboard..."
cd /home/ubuntu/dashboard
pkill -f 'serve.*3000' || true
sleep 2

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "Installing dashboard dependencies..."
    npm install
fi

# Build dashboard if dist doesn't exist
if [ ! -d "dist" ] || [ ! -f "dist/index.html" ]; then
    echo "Building dashboard..."
    npm run build
fi

# Start dashboard server
nohup npx serve -l 3000 -s dist > /tmp/dashboard.log 2>&1 &
DASHBOARD_PID=$!
echo $DASHBOARD_PID > /tmp/dashboard.pid
sleep 3

# Verify dashboard is running
if ps -p $DASHBOARD_PID > /dev/null; then
    echo "Dashboard started successfully (PID: $DASHBOARD_PID)"
else
    echo "Dashboard failed to start"
    tail -20 /tmp/dashboard.log
    exit 1
fi

# Test dashboard HTTP response
if curl -s http://localhost:3000 | grep -q "<!doctype html>"; then
    echo "Dashboard is responding correctly"
else
    echo "Dashboard not responding correctly"
    tail -20 /tmp/dashboard.log
fi

echo ""
echo "===== Service Status ====="
echo "vLLM: http://localhost:8005"
echo "Orchestrator: http://localhost:8000"
echo "Dashboard: http://localhost:3000"
echo "Nginx: http://$(curl -s ifconfig.me)"
echo ""
echo "Check logs:"
echo "  vLLM: tail -f /tmp/vllm.log"
echo "  Dashboard: tail -f /tmp/dashboard.log"
echo "  Docker: docker compose logs -f"
