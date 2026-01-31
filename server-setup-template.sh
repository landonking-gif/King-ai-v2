#!/bin/bash
set -e
export DEBIAN_FRONTEND=noninteractive

echo "=== CLEANUP ==="
cd /home/ubuntu
pkill -f "python.*orchestrator" 2>/dev/null || true
pkill -f "python.*mcp-gateway" 2>/dev/null || true
pkill -f "python.*memory-service" 2>/dev/null || true
pkill -f "python.*subagent-manager" 2>/dev/null || true
pkill -f "serve.*3000" 2>/dev/null || true
pkill -f "node.*3000" 2>/dev/null || true
pkill -f "pnpm.*moltbot" 2>/dev/null || true
sleep 2

echo "=== SYSTEM PACKAGES ==="
apt-get update -qq 2>&1 | grep -E "(packages can be upgraded|Reading)" || true
apt-get install -y --no-install-recommends python3-pip python3-venv postgresql redis-server nginx curl htop unzip 2>&1 | tail -3
systemctl start postgresql redis-server 2>/dev/null || true
systemctl enable postgresql redis-server 2>/dev/null || true

echo "=== DATABASE SETUP ==="
sudo -u postgres psql -c "CREATE USER agentic_user WITH PASSWORD 'agentic_pass';" 2>/dev/null || echo "User exists"
sudo -u postgres psql -c "CREATE DATABASE agentic_framework OWNER agentic_user;" 2>/dev/null || echo "DB exists"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE agentic_framework TO agentic_user;" 2>/dev/null || echo "Granted"

echo "=== PYTHON ENVIRONMENT ==="
cd agentic-framework-main
python3 -m venv .venv --clear
source .venv/bin/activate
pip install --upgrade pip setuptools wheel --quiet
pip install -r requirements.txt
echo "✓ Python environment ready"

echo "=== SERVICE STARTUP ==="
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Set common environment variables
export PYTHONPATH="/home/ubuntu/agentic-framework-main:/home/ubuntu:$PYTHONPATH"
export AWS_IP="__AWS_IP__"

cd orchestrator
source ../.venv/bin/activate
export PYTHONPATH="/home/ubuntu/king-ai-v3/agentic-framework-main:/home/ubuntu/agentic-framework-main:/home/ubuntu:$PYTHONPATH"
export AWS_IP="__AWS_IP__"
nohup python run_service.py > /tmp/orchestrator.log 2>&1 &
echo $! > /tmp/orchestrator.pid
cd ..
echo "✓ Orchestrator started"

cd mcp-gateway
source ../.venv/bin/activate
export PYTHONPATH="/home/ubuntu/king-ai-v3/agentic-framework-main:/home/ubuntu/agentic-framework-main:/home/ubuntu:$PYTHONPATH"
nohup python -m service.main > /tmp/mcp-gateway.log 2>&1 &
echo $! > /tmp/mcp-gateway.pid
cd ..
echo "✓ MCP Gateway started"

cd memory-service
source ../.venv/bin/activate
export POSTGRES_URL="postgresql://agentic_user:agentic_pass@localhost:5432/agentic_framework"
export PYTHONPATH="/home/ubuntu/king-ai-v3/agentic-framework-main:/home/ubuntu/agentic-framework-main:/home/ubuntu:$PYTHONPATH"
nohup python -m service.main > /tmp/memory-service.log 2>&1 &
echo $! > /tmp/memory-service.pid
cd ..
echo "✓ Memory Service started"

cd subagent-manager
source ../.venv/bin/activate
export PYTHONPATH="/home/ubuntu/king-ai-v3/agentic-framework-main:/home/ubuntu/agentic-framework-main:/home/ubuntu:$PYTHONPATH"
nohup python -m service.main > /tmp/subagent-manager.log 2>&1 &
echo $! > /tmp/subagent-manager.pid
cd ..
echo "✓ Subagent Manager started"

echo "=== NGINX CONFIG ==="
cat > /tmp/nginx.conf << 'EOF'
server {
    listen 80 default_server;
    server_name _;
    proxy_connect_timeout 300;
    proxy_send_timeout 300;
    proxy_read_timeout 300;
    location /health { proxy_pass http://127.0.0.1:8000/health; }
    location /api { proxy_pass http://127.0.0.1:8000; proxy_set_header Host $host; proxy_set_header X-Real-IP $remote_addr; }
    location /docs { proxy_pass http://127.0.0.1:8000/docs; proxy_set_header Host $host; }
    location / { return 200 "{\"service\":\"King AI v3\",\"status\":\"running\",\"api\":\"/api\",\"docs\":\"/docs\"}"; }
}
EOF

sudo cp /tmp/nginx.conf /etc/nginx/sites-available/king-ai
sudo ln -sf /etc/nginx/sites-available/king-ai /etc/nginx/sites-enabled/default 2>/dev/null || true
sudo nginx -t && sudo systemctl restart nginx

echo "=== DEPLOYMENT COMPLETE ==="
echo "Services running on ports:"
echo "  8000 (Orchestrator)"
echo "  8001 (Subagent Manager)"
echo "  8002 (Memory Service)"
echo "  8080 (MCP Gateway)"
