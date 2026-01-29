#!/bin/bash

echo "Starting dashboard service..."

cd /home/ubuntu/dashboard

if [ ! -f "package.json" ]; then
    echo "ERROR: package.json not found in /home/ubuntu/dashboard"
    exit 1
fi

# Check if dependencies are installed
if [ ! -d "node_modules" ]; then
    echo "Installing dependencies..."
    npm install
fi

# Build the dashboard
echo "Building dashboard..."
npm run build

# Determine build directory
BUILD_DIR="dist"
[ -d "build" ] && BUILD_DIR="build"

if [ ! -d "$BUILD_DIR" ]; then
    echo "ERROR: Build failed - no $BUILD_DIR directory"
    exit 1
fi

# Kill existing dashboard process
pkill -f 'serve.*3000' 2>/dev/null
sleep 2

# Check if serve is installed
if ! command -v serve &> /dev/null; then
    echo "Installing serve globally..."
    sudo npm install -g serve
fi

# Start dashboard server
echo "Starting dashboard on port 3000..."
nohup serve -l 3000 -s $BUILD_DIR > /tmp/dashboard.log 2>&1 &
echo $! > /tmp/dashboard.pid

sleep 3

# Check if it started
if kill -0 $(cat /tmp/dashboard.pid) 2>/dev/null; then
    echo "✓ Dashboard started successfully (PID: $(cat /tmp/dashboard.pid))"
    echo ""
    echo "Dashboard logs:"
    tail -10 /tmp/dashboard.log
    echo ""
    echo "Test locally: curl http://localhost:3000"
else
    echo "✗ Dashboard failed to start"
    echo ""
    tail -20 /tmp/dashboard.log
fi
