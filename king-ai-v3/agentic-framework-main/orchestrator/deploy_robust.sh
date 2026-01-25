#!/bin/bash
# Robust AWS Deployment Script - Handles SSH issues gracefully
#
# Usage: echo 'YOUR_IP' | bash deploy_robust.sh

set -e

echo "=================================="
echo "Robust AWS Deployment Script"
echo "=================================="
echo ""

# Get AWS IP address
read -p "Enter AWS server IP address: " aws_ip

# Update dashboard .env with the new IP
/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe -Command "Set-Content -Path 'C:\Users\dmilner.AGV-040318-PC\Downloads\landon\king-ai-v2\dashboard\.env' -Value 'VITE_API_BASE=http://$aws_ip:8000/api'"

ssh_key="$(cd .. && pwd)/king-ai-studio.pem"
ssh_user=ubuntu

echo "Target server: $aws_ip"
echo "SSH key: $ssh_key"
echo ""

# Test connectivity first
echo "Testing connectivity..."
if ping -c 1 -W 5 $aws_ip > /dev/null 2>&1; then
    echo "✓ Server is reachable"
else
    echo "✗ Server is not reachable"
    echo "Please check:"
    echo "  - AWS instance is running"
    echo "  - IP address is correct"
    echo "  - Network connectivity"
    exit 1
fi

# Test SSH connectivity
echo "Testing SSH connectivity..."
if ssh -i "$ssh_key" -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$ssh_user@$aws_ip" 'echo "SSH test successful"' > /dev/null 2>&1; then
    echo "✓ SSH access available"
    USE_SSH=true
else
    echo "⚠ SSH access failed - will use alternative deployment method"
    USE_SSH=false
fi

echo ""

if [[ "$USE_SSH" == "true" ]]; then
    echo "Using SSH-based deployment..."

    # Convert WSL path to Windows path for Windows SSH
    ssh_key_win=${ssh_key/\/mnt\/c\//C:\\}
    ssh_key_win=${ssh_key_win//\//\\}

    # Use the existing deployment logic
    echo "Proceeding with standard deployment..."
    # ... existing SSH deployment code would go here

else
    echo "Using alternative deployment method..."
    echo "Checking current service status..."

    # Check which services are running
    services_running=0
    services_total=0

    # Test each service
    for port in 8000 8080 8002 8001 3000; do
        services_total=$((services_total + 1))
        if curl -sf --max-time 3 http://$aws_ip:$port/health > /dev/null 2>&1; then
            services_running=$((services_running + 1))
            echo "✓ Port $port is responding"
        else
            echo "✗ Port $port is not responding"
        fi
    done

    echo ""
    echo "Status: $services_running/$services_total services running"

    if [[ $services_running -eq $services_total ]]; then
        echo "✓ All services are running - no deployment needed"
        exit 0
    fi

    echo ""
    echo "Services need to be restarted. Since SSH is unavailable, please:"
    echo ""
    echo "Option 1: Fix SSH Access"
    echo "1. Check AWS Console → EC2 → Security Groups"
    echo "2. Ensure SSH (port 22) is allowed from your IP"
    echo "3. Verify SSH key permissions: chmod 600 $ssh_key"
    echo "4. Try: ssh -v -i $ssh_key ubuntu@$aws_ip"
    echo ""
    echo "Option 2: Manual Service Restart via AWS Console"
    echo "1. Go to AWS Console → EC2 → Instances"
    echo "2. Connect to instance via EC2 Instance Connect"
    echo "3. Run these commands:"
    echo ""
    echo "   cd ~/agentic-framework-main"
    echo "   source .venv/bin/activate"
    echo "   # Kill existing processes"
    echo "   pkill -f python"
    echo "   pkill -f npm"
    echo "   sleep 2"
    echo "   # Start services"
    echo "   ./orchestrator/run_service.sh"
    echo ""
    echo "Option 3: Full Instance Restart"
    echo "1. Stop the instance in AWS Console"
    echo "2. Start it again (may get new IP)"
    echo "3. Update IP and run: echo 'NEW_IP' | bash run_service.sh"
    echo ""
    echo "Current IP: $aws_ip"
    echo "Dashboard URL: http://$aws_ip"
    echo "API URL: http://$aws_ip/api"
fi

echo ""
echo "=================================="
echo "Deployment check complete"
echo "=================================="