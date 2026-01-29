#!/bin/bash
# MoltBot Startup Script for King AI Integration
# This script starts the MoltBot gateway and connects it to King AI orchestrator

set -e

MOLTBOT_DIR="/home/ubuntu/king-ai-v3/moltbot"
LOG_FILE="/tmp/moltbot.log"
PID_FILE="/tmp/moltbot.pid"

# Change to MoltBot directory
cd "$MOLTBOT_DIR"

# Check if MoltBot is already running
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "MoltBot is already running with PID $PID"
        exit 0
    else
        echo "Removing stale PID file"
        rm "$PID_FILE"
    fi
fi

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "Installing MoltBot dependencies..."
    pnpm install
fi

# Check if dist exists
if [ ! -d "dist" ]; then
    echo "Building MoltBot..."
    pnpm build
fi

# Start MoltBot gateway
echo "Starting MoltBot gateway on port 18789..."
nohup pnpm moltbot gateway --port 18789 --verbose > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"

echo "MoltBot started with PID $(cat $PID_FILE)"
echo "Logs: $LOG_FILE"
echo ""
echo "MoltBot Gateway: http://localhost:18789"
echo "King AI Orchestrator: http://localhost:8000"
echo ""
echo "To view logs: tail -f $LOG_FILE"
echo "To stop: kill \$(cat $PID_FILE)"
