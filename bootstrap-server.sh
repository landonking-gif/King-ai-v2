#!/bin/bash
set -e

echo "==== King AI Complete Server Setup ===="
echo "This script will install and configure all services"
echo ""

# Install Node.js
echo "[1/7] Installing Node.js..."
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash - 2>&1 | grep -v "^W:"
sudo apt-get install -y nodejs nginx 2>&1 | grep -v "^W:"
node --version
npm --version

# Start vLLM via Docker
echo "[2/7] Starting vLLM service..."
docker stop vllm 2>/dev/null || true
docker rm vllm 2>/dev/null || true
docker run -d --gpus all --name vllm -p 8005:8000 --ipc=host \
  vllm/vllm-openai:latest \
  --model casperhansen/deepseek-r1-distill-qwen-7b-awq \
  --dtype float16 \
  --api-key token-abc123 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096

# Build dashboard
echo "[3/7] Building dashboard..."
cd /home/ubuntu/dashboard
npm install
npm run build

# Configure nginx
echo "[4/7] Configuring nginx..."
sudo tee /etc/nginx/sites-available/default > /dev/null <<'EOF'
server {
    listen 80 default_server;
    server_name _;

    location / {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }

    location /api {
        return 503 '{"error": "Orchestrator not running. Deploy agentic-framework-main first."}';
    }
}
EOF

sudo nginx -t &&  sudo systemctl reload nginx

# Start dashboard
echo "[5/7] Starting dashboard..."
pkill -f 'serve.*3000' || true
sleep 2
cd /home/ubuntu/dashboard
nohup npx serve -l 3000 -s dist > /tmp/dashboard.log 2>&1 &
DASH_PID=$!
echo $DASH_PID > /tmp/dashboard.pid
sleep 3

# Verify services
echo "[6/7] Verifying services..."
echo "  - Docker vLLM: $(docker ps | grep vllm > /dev/null && echo 'Running' || echo 'NOT RUNNING')"
echo "  - Dashboard: $(ps -p $DASH_PID > /dev/null && echo 'Running (PID '$DASH_PID')' || echo 'NOT RUNNING')"
echo "  - Nginx: $(systemctl is-active nginx)"

# Test endpoints
echo "[7/7] Testing endpoints..."
sleep 5
curl -s http://localhost:3000 > /dev/null && echo "  ✓ Dashboard responding on :3000" || echo "  ✗ Dashboard not responding"
curl -s http://localhost:80 > /dev/null && echo "  ✓ Nginx responding on :80" || echo "  ✗ Nginx not responding"

echo ""
echo "==== Setup Complete ===="
echo ""
echo "Access dashboard at: http://$(curl -s ifconfig.me)/"
echo ""
echo "Logs:"
echo "  vLLM: docker logs -f vllm"
echo "  Dashboard: tail -f /tmp/dashboard.log"
echo "  Nginx: sudo tail -f /var/log/nginx/access.log"
echo ""
echo "Note: vLLM may take 2-3 minutes to load the model"
echo "Check vLLM status: docker logs vllm | tail -50"
