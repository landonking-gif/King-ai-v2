#!/bin/bash
# Service Recovery Script - Restart only missing/down services
# Usage: echo 'YOUR_IP' | bash recover_services.sh

set -e

echo "=================================="
echo "Service Recovery Script"
echo "=================================="
echo ""

# Get AWS IP address
read -p "Enter AWS server IP address: " aws_ip

ssh_key="$(cd .. && pwd)/king-ai-studio.pem"
ssh_user=ubuntu

echo "Checking service status on $aws_ip..."

# Check service health via HTTP (no SSH needed for this)
echo "Service Health Status:"
echo "======================"

services=(
    "Orchestrator:8000"
    "MCP Gateway:8080"
    "Memory Service:8002"
    "Subagent Manager:8001"
    "Dashboard:3000"
)

declare -A service_status
for service_info in "${services[@]}"; do
    name=${service_info%:*}
    port=${service_info#*:}
    if curl -sf --max-time 5 http://$aws_ip:$port/health > /dev/null 2>&1; then
        echo "✓ $name healthy"
        service_status[$name]="healthy"
    else
        echo "✗ $name unhealthy/down"
        service_status[$name]="down"
    fi
done

echo ""
echo "Recovery Plan:"
echo "=============="

# Determine what needs to be restarted
needs_restart=()
if [[ ${service_status["Memory Service"]} == "down" ]]; then
    needs_restart+=("memory-service")
    echo "• Restart Memory Service (port 8002)"
fi

if [[ ${service_status["Subagent Manager"]} == "down" ]]; then
    needs_restart+=("subagent-manager")
    echo "• Restart Subagent Manager (port 8001)"
fi

if [[ ${service_status["MCP Gateway"]} == "down" ]]; then
    needs_restart+=("mcp-gateway")
    echo "• Restart MCP Gateway (port 8080)"
fi

if [[ ${service_status["Dashboard"]} == "down" ]]; then
    needs_restart+=("dashboard")
    echo "• Restart Dashboard (port 3000)"
fi

if [[ ${service_status["Orchestrator"]} == "down" ]]; then
    needs_restart+=("orchestrator")
    echo "• Restart Orchestrator (port 8000)"
fi

if [[ ${#needs_restart[@]} -eq 0 ]]; then
    echo "✓ All services are healthy - no recovery needed"
    exit 0
fi

echo ""
echo "Proceeding with recovery..."

# Try SSH connection for recovery
echo "Attempting SSH connection for service recovery..."
if ssh -i "$ssh_key" -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$ssh_user@$aws_ip" 'echo "SSH connection successful"' 2>/dev/null; then
    echo "✓ SSH access available - performing targeted restart"

    # SSH-based recovery
    ssh -i "$ssh_key" -o StrictHostKeyChecking=no "$ssh_user@$aws_ip" << EOF
        cd ~/agentic-framework-main
        source .venv/bin/activate
        export PYTHONPATH="\$HOME/agentic-framework-main:\$PYTHONPATH"

        echo "Stopping unhealthy services..."
        for service in ${needs_restart[*]}; do
            case \$service in
                "memory-service")
                    pkill -f "memory-service" || true
                    ;;
                "subagent-manager")
                    pkill -f "subagent-manager" || true
                    ;;
                "mcp-gateway")
                    pkill -f "mcp-gateway" || true
                    ;;
                "orchestrator")
                    pkill -f "orchestrator" || true
                    ;;
                "dashboard")
                    pkill -f "npm.*preview" || true
                    ;;
            esac
        done

        sleep 3

        echo "Starting recovered services..."
        for service in ${needs_restart[*]}; do
            case \$service in
                "memory-service")
                    echo "Starting Memory Service..."
                    cd memory-service
                    nohup ./run.sh > memory-service.log 2>&1 &
                    cd ..
                    ;;
                "subagent-manager")
                    echo "Starting Subagent Manager..."
                    cd subagent-manager
                    nohup ./run.sh > subagent-manager.log 2>&1 &
                    cd ..
                    ;;
                "mcp-gateway")
                    echo "Starting MCP Gateway..."
                    cd mcp-gateway
                    nohup ./run.sh > mcp-gateway.log 2>&1 &
                    cd ..
                    ;;
                "orchestrator")
                    echo "Starting Orchestrator..."
                    cd orchestrator
                    nohup python -m service.main --host 0.0.0.0 --port 8000 > orchestrator.log 2>&1 &
                    cd ..
                    ;;
                "dashboard")
                    echo "Starting Dashboard..."
                    cd dashboard
                    nohup npm run preview -- --host 0.0.0.0 --port 3000 > dashboard.log 2>&1 &
                    cd ..
                    ;;
            esac
        done

        echo "Waiting for services to start..."
        sleep 10

        echo "Recovery complete!"
EOF

else
    echo "✗ SSH access failed - attempting alternative recovery methods"

    # Alternative: Try to trigger service restart via API if orchestrator is healthy
    if [[ ${service_status["Orchestrator"]} == "healthy" ]]; then
        echo "✓ Orchestrator is healthy - attempting API-based recovery"

        # Could add API endpoints for service management here if they exist
        echo "Note: API-based service management not implemented yet"
        echo "Please check AWS Console for manual service restart options"
    else
        echo "✗ No recovery options available"
        echo ""
        echo "Manual recovery steps:"
        echo "1. Check AWS Console - ensure instance is running"
        echo "2. Verify Security Group allows SSH (port 22)"
        echo "3. If IP changed, update deployment scripts"
        echo "4. Try full redeploy: echo '$aws_ip' | bash run_service.sh"
    fi
fi

echo ""
echo "Post-recovery verification:"
echo "=========================="

# Re-check services
for service_info in "${services[@]}"; do
    name=${service_info%:*}
    port=${service_info#*:}
    if curl -sf --max-time 5 http://$aws_ip:$port/health > /dev/null 2>&1; then
        echo "✓ $name recovered"
    else
        echo "✗ $name still down"
    fi
done

echo ""
echo "=================================="
echo "Recovery attempt complete"
echo "=================================="