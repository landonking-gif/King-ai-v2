#!/bin/bash
set -e

echo "Starting King AI Dashboard..."

cd /home/ubuntu/dashboard

# Check if build exists
if [ ! -d "dist" ]; then
    echo "Building dashboard..."
    npm run build
fi

# Kill existing dashboard
pkill -f 'serve.*3000' 2>/dev/null || true
sleep 2

# Start dashboard
echo "Starting serve on port 3000..."
serve -l 3000 -s dist > /tmp/dashboard.log 2>&1 &
DASH_PID=$!
echo $DASH_PID > /tmp/dashboard.pid

sleep 2

# Verify it started
if ps -p $DASH_PID > /dev/null 2>&1; then
    echo "✓ Dashboard started successfully (PID: $DASH_PID)"
    curl -s http://localhost:3000 > /dev/null && echo "✓ Dashboard responding on port 3000"
else
    echo "✗ Dashboard failed to start"
    tail -20 /tmp/dashboard.log
    exit 1
fi
