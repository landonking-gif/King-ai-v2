#!/bin/bash
# Check AWS instance status and processes
echo "Checking AWS instance connection..."

ssh -i /mnt/c/Users/dmilner.AGV-040318-PC/Downloads/landon/king-ai-v2/king-ai-v3/agentic-framework-main/king-ai-studio.pem \
    -o StrictHostKeyChecking=no \
    -o ConnectTimeout=30 \
    ubuntu@54.224.134.220 << 'EOF'
echo "✓ Connection successful"
echo ""
echo "Python processes:"
ps aux | grep python | grep -v grep || echo "No Python processes found"
echo ""
echo "Port 8000 status:"
sudo netstat -tlnp | grep :8000 || echo "Port 8000 not in use"
echo ""
echo "Checking orchestrator service:"
curl -s http://localhost:8000/health || echo "Service not responding"
EOF