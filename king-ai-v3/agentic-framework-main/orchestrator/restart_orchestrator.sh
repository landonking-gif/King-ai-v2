#!/bin/bash
# Script to restart orchestrator service on AWS
# Usage: echo 'YOUR_IP' | bash restart_orchestrator.sh

set -e

echo "=================================="
echo "Restart Orchestrator Service"
echo "=================================="
echo ""

# Get AWS IP address
read -p "Enter AWS server IP address: " aws_ip

ssh_key="$(cd .. && pwd)/king-ai-studio.pem"
ssh_user=ubuntu

echo "Restarting orchestrator on $aws_ip..."

# SSH to AWS and restart the service
/mnt/c/Windows/System32/OpenSSH/ssh.exe -i "$ssh_key" -o StrictHostKeyChecking=no "$ssh_user@$aws_ip" << 'EOF'
    echo "Stopping existing orchestrator processes..."
    pkill -9 -f "python.*service.main" || echo "No orchestrator process found"
    pkill -9 -f "uvicorn.*orchestrator" || echo "No uvicorn process found"
    
    echo "Checking port 8000..."
    if sudo lsof -i :8000 -t > /dev/null 2>&1; then
        echo "Killing process on port 8000..."
        sudo kill -9 $(sudo lsof -i :8000 -t)
    else
        echo "Port 8000 is free"
    fi
    
    sleep 2
    
    echo "Starting orchestrator service..."
    cd ~/agentic-framework-main/orchestrator
    source ../.venv/bin/activate
    
    # Start in background with nohup
    nohup python -m service.main > orchestrator.log 2>&1 &
    
    sleep 3
    
    # Check if service started
    if curl -s http://localhost:8000/health > /dev/null; then
        echo "✓ Orchestrator started successfully"
        echo "Service available at: http://$aws_ip:8000"
    else
        echo "✗ Failed to start orchestrator. Check logs:"
        tail -n 20 orchestrator.log
        exit 1
    fi
EOF

echo ""
echo "=================================="
echo "Service restarted successfully!"
echo "=================================="
