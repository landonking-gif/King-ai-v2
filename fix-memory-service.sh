#!/bin/bash
# Fix Memory Service - Start it properly on AWS server

echo "🔧 Starting Memory Service Fix..."
echo "================================"

# Navigate to project directory
cd /home/ubuntu/agentic-framework-main || exit 1

# Kill any existing memory service processes
echo "1. Stopping any existing memory service..."
pkill -f "memory-service.service.main" || echo "No existing process found"
sleep 2

# Check if .venv exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found!"
    exit 1
fi

echo "2. Starting Memory Service on port 8002..."

# Start the service with proper environment
export PYTHONUNBUFFERED=1

# Try to find and use virtual environment
PYTHON_CMD=""
if [ -f ".venv/bin/python" ]; then
    PYTHON_CMD=".venv/bin/python"
    echo "Using local .venv"
elif [ -f "../.venv/bin/python" ]; then
    PYTHON_CMD="../.venv/bin/python"
    echo "Using parent .venv"
elif [ -f "/home/ubuntu/agentic-framework-main/.venv/bin/python" ]; then
    PYTHON_CMD="/home/ubuntu/agentic-framework-main/.venv/bin/python"
    echo "Using AWS .venv"
elif [ -f "/mnt/c/Users/dmilner.AGV-040318-PC/Downloads/landon/king-ai-v2/venv/Scripts/python.exe" ]; then
    PYTHON_CMD="/mnt/c/Users/dmilner.AGV-040318-PC/Downloads/landon/king-ai-v2/venv/Scripts/python.exe"
    echo "Using main Windows venv"
else
    PYTHON_CMD="python3"
    echo "No venv found, using system python3"
fi

nohup $PYTHON_CMD -m memory-service.service.main > /tmp/memory-service.log 2>&1 &
MEMORY_PID=$!

echo "   Process ID: $MEMORY_PID"
sleep 5

# Check if it's running
if ps -p $MEMORY_PID > /dev/null; then
    echo "✅ Memory Service started successfully (PID: $MEMORY_PID)"
else
    echo "❌ Memory Service failed to start"
    echo "Last 30 lines of log:"
    tail -30 /tmp/memory-service.log
    exit 1
fi

# Test the health endpoint
echo "3. Testing health endpoint..."
sleep 3

HEALTH_CHECK=$(curl -s http://localhost:8002/health 2>&1)
if echo "$HEALTH_CHECK" | grep -q "status"; then
    echo "✅ Memory Service is responding"
    echo "   Response: $HEALTH_CHECK"
else
    echo "⚠️  Health check failed, checking logs..."
    tail -20 /tmp/memory-service.log
fi

# Check all ports
echo "4. Active services:"
ss -tuln | grep -E ':(3000|8000|8001|8002|8080|11434)'

echo ""
echo "================================"
echo "✅ Fix complete!"
echo ""
echo "To check status:"
echo "  curl http://localhost:8002/health"
echo "  tail -f /tmp/memory-service.log"
