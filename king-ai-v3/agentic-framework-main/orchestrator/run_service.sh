#!/bin/bash
# Optimized deployment script for AWS server
#
# This script has been updated to address common deployment issues:
# - Missing Python dependencies: Now installs all packages from requirements.txt
# - Port conflicts: Comprehensive process cleanup before starting services
# - Database credential mismatches: Automatic fix for memory-service/.env
# - Incomplete dashboard sync: Full rsync of dashboard directory
# - Service startup reliability: Uses start_all_services.sh when available
# - Ollama LLM model warmup: Prevents first-request timeouts
# - Ollama keepalive service: Maintains model in memory
# - Nginx proxy timeouts: Increased for LLM response times
#
# Usage examples:
#   Interactive: ./run_service.sh
#   Automated: echo 'YOUR_IP' | bash run_service.sh
#   WSL example: wsl bash -c "cd /mnt/c/Users/dmilner.AGV-040318-PC/Downloads/landon/king-ai-v2/king-ai-v3/agentic-framework-main/orchestrator && sed -i 's/\r$//' run_service.sh && echo '54.224.134.220' | bash run_service.sh"

set -e

echo "=================================="
echo "AWS Deployment Script"
echo "=================================="
echo ""

# Get AWS IP address
read -p "Enter AWS server IP address: " aws_ip
# Update dashboard .env with the new IP
/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe -Command "Set-Content -Path 'C:\Users\dmilner.AGV-040318-PC\Downloads\landon\king-ai-v2\dashboard\.env' -Value 'VITE_API_BASE=http://$aws_ip:8000/api'"

ssh_key="$(cd .. && pwd)/king-ai-studio.pem"
ssh_user=ubuntu

# Convert WSL path to Windows path for Windows SSH
ssh_key=${ssh_key/\/mnt\/c\//C:\\}
ssh_key=${ssh_key//\//\\}

echo "Deploying to AWS server $aws_ip..."

# One-time system setup (check if already done to avoid redundant operations)
echo "Checking system setup on AWS server..."
/mnt/c/Windows/System32/OpenSSH/ssh.exe -i "$ssh_key" -o StrictHostKeyChecking=no "$ssh_user@$aws_ip" << 'EOF'
    # Check if Ollama is installed
    if ! command -v ollama &> /dev/null; then
        echo "Installing Ollama..."
        sudo apt update && sudo apt install -y curl
        curl -fsSL https://ollama.ai/install.sh | sh
        sudo systemctl enable ollama
        sudo systemctl start ollama
        ollama pull llama3.1:8b
    else
        echo "✓ Ollama already installed"
    fi
    
    # Check if required packages are installed
    if ! command -v redis-server &> /dev/null || ! command -v psql &> /dev/null; then
        echo "Installing system packages..."
        sudo apt update
        sudo apt install -y python3-venv redis-server postgresql postgresql-contrib nginx
        
        # Install Node.js only if not present
        if ! command -v node &> /dev/null; then
            curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
            sudo apt-get install -y nodejs
        fi
        
        # Start and enable services
        sudo systemctl enable redis-server postgresql nginx
        sudo systemctl start redis-server postgresql
        
        # Setup PostgreSQL database (only once)
        sudo -u postgres psql -tc "SELECT 1 FROM pg_database WHERE datname = 'agentic_framework'" | grep -q 1 || {
            sudo -u postgres psql -c "CREATE USER agentic_user WITH PASSWORD 'agentic_pass';"
            sudo -u postgres psql -c "CREATE DATABASE agentic_framework OWNER agentic_user;"
            sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE agentic_framework TO agentic_user;"
        }
        
        # Install MinIO only if not present
        if [ ! -f ~/minio ]; then
            wget -q https://dl.min.io/server/minio/release/linux-amd64/minio -O ~/minio
            chmod +x ~/minio
        fi
    else
        echo "✓ System packages already installed"
    fi
EOF

# Copy project files efficiently (only changed files)
echo "Syncing project files to AWS server..."
cd ..
rsync -az --delete --quiet -e "/mnt/c/Windows/System32/OpenSSH/ssh.exe -i \"$ssh_key\" -o StrictHostKeyChecking=no" \
    --exclude='.venv' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.git' \
    --exclude='node_modules' \
    --exclude='*.log' \
    . "$ssh_user@$aws_ip:~/agentic-framework-main"

# Copy dashboard
# Full sync ensures all files including package.json are present
echo "Syncing dashboard..."
# Check if dashboard directory exists locally
if [ ! -d "../../dashboard" ]; then
    echo "Error: Dashboard directory not found at ../../dashboard"
    echo "Current directory: $(pwd)"
    echo "Expected dashboard path: $(pwd)/../../dashboard"
    exit 1
fi
# Ensure destination directory exists
/mnt/c/Windows/System32/OpenSSH/ssh.exe -i "$ssh_key" -o StrictHostKeyChecking=no "$ssh_user@$aws_ip" "mkdir -p ~/agentic-framework-main/dashboard"
# Use rsync for efficient file transfer
rsync -az --delete --quiet -e "/mnt/c/Windows/System32/OpenSSH/ssh.exe -i \"$ssh_key\" -o StrictHostKeyChecking=no" \
    --exclude='node_modules' \
    --exclude='dist' \
    --exclude='.git' \
    ../../dashboard/ "$ssh_user@$aws_ip:~/agentic-framework-main/dashboard/"
# Verify dashboard files were copied
/mnt/c/Windows/System32/OpenSSH/ssh.exe -i "$ssh_key" -o StrictHostKeyChecking=no "$ssh_user@$aws_ip" "ls -la ~/agentic-framework-main/dashboard/ | head -5"
cd orchestrator

# Fix line endings
echo "Fixing line endings..."
/mnt/c/Windows/System32/OpenSSH/ssh.exe -i "$ssh_key" -o StrictHostKeyChecking=no "$ssh_user@$aws_ip" << 'EOF'
    cd ~/agentic-framework-main
    find . -name "*.sh" -type f -exec sed -i 's/\r$//' {} \;
    echo "✓ Line endings fixed"
EOF

# Setup Python environment and install dependencies efficiently
echo "Setting up Python environment..."
/mnt/c/Windows/System32/OpenSSH/ssh.exe -i "$ssh_key" -o StrictHostKeyChecking=no "$ssh_user@$aws_ip" << EOF
    cd ~/agentic-framework-main
    
    # Create or reuse virtual environment
    if [ ! -d ".venv" ]; then
        python3 -m venv .venv
    fi
    source .venv/bin/activate
    
    # Install all dependencies from requirements.txt
    pip install -q --upgrade pip setuptools wheel
    # Install critical packages that have caused deployment issues
    pip install -q python-jose[cryptography] asyncpg prometheus-client jsonschema
    pip install -q -r requirements.txt
    
    # Copy and configure .env files
    if [ ! -f ".env" ]; then
        cp .env.example .env
        sed -i 's|OLLAMA_ENDPOINT=.*|OLLAMA_ENDPOINT=http://localhost:11434|' .env
        sed -i 's|postgresql://user:password@localhost:5432/agentic_framework|postgresql://agentic_user:agentic_pass@localhost:5432/agentic_framework|' .env
        echo "MEMORY_SERVICE_PORT=8002" >> .env
        echo "AWS_IP=$aws_ip" >> .env
    fi
    
    # Copy .env to service directories only if they don't exist
    for dir in mcp-gateway memory-service subagent-manager orchestrator; do
        [ ! -f "\$dir/.env" ] && cp .env "\$dir/.env"
    done
    
    # Setup dashboard (only if package.json changed or node_modules missing)
    cd dashboard
    if [ ! -d "node_modules" ] || [ "package.json" -nt "node_modules" ]; then
        npm install --silent
    fi
    npm run build
    cd ..
    
    echo "✓ Environment setup complete"
EOF

# Start all services
echo "Starting services on AWS server..."
/mnt/c/Windows/System32/OpenSSH/ssh.exe -i "$ssh_key" -o StrictHostKeyChecking=no "$ssh_user@$aws_ip" << 'EOF'
    cd ~/agentic-framework-main
    source .venv/bin/activate
    
    # Kill all existing service processes completely
    # This prevents port conflicts (e.g., port 8000 already in use)
    echo "Stopping all existing services..."
    pkill -9 -f "python.*service.main" || true
    pkill -9 -f "uvicorn.*service.main:app" || true
    pkill -9 -f "uvicorn.*main:app" || true
    pkill -9 -f "npm run preview" || true
    pkill -9 -f minio || true
    
    # Kill processes on specific ports as backup
    lsof -ti:8000,8001,8002,8080,3000 2>/dev/null | xargs -r kill -9 || true
    
    echo "Waiting for ports to be released..."
    sleep 5
    
    # Ensure database credentials are correct in memory-service/.env
    # Fixes authentication failures when old credentials (king:password) are present
    if grep -q "king:password" memory-service/.env 2>/dev/null; then
        echo "Fixing database credentials..."
        sed -i 's|postgresql://king:password@localhost:5432/kingai|postgresql://agentic_user:agentic_pass@localhost:5432/agentic_framework|g' memory-service/.env
    fi
    
    # Create data directories
    mkdir -p ~/agentic-framework-main/data/{minio,chroma}
    mkdir -p ~/agentic-framework-main/memory-service/data/chroma
    
    # Start MinIO if not running
    if ! pgrep -f "minio server" > /dev/null; then
        nohup ~/minio server ~/agentic-framework-main/data/minio --console-address :9001 > /tmp/minio.log 2>&1 &
    fi
    
    # Use the comprehensive startup script if it exists
    # This provides reliable service startup with health monitoring
    if [ -f start_all_services.sh ]; then
        echo "Using start_all_services.sh..."
        bash start_all_services.sh
    else
        echo "Using legacy startup method..."
        export PYTHONPATH="$HOME/agentic-framework-main:$PYTHONPATH"
        
        # Start MCP Gateway
        cd ~/agentic-framework-main/mcp-gateway
        python -m service.main > /tmp/mcp-gateway.log 2>&1 &
        
        # Start Memory Service
        cd ~/agentic-framework-main/memory-service
        uvicorn service.main:app --host 0.0.0.0 --port 8002 > /tmp/memory-service.log 2>&1 &
        
        # Start Subagent Manager
        cd ~/agentic-framework-main/subagent-manager
        uvicorn service.main:app --host 0.0.0.0 --port 8001 > /tmp/subagent-manager.log 2>&1 &
        
        # Start Dashboard
        cd ~/agentic-framework-main/dashboard
        npm run preview -- --host 0.0.0.0 --port 3000 > /tmp/dashboard.log 2>&1 &
        
        # Start Orchestrator
        cd ~/agentic-framework-main/orchestrator
        python -m service.main --host 0.0.0.0 --port 8000 > /tmp/orchestrator.log 2>&1 &
        
        cd ~/agentic-framework-main
        
        echo "Waiting for services to start..."
        sleep 15
        
        # Health checks
        echo ""
        echo "Service Health Status:"
        echo "======================"
        for service in "MCP Gateway:8080" "Memory Service:8002" "Subagent Manager:8001" "Orchestrator:8000" "Dashboard:3000"; do
            name=${service%:*}
            port=${service#*:}
            if curl -sf http://localhost:$port/health > /dev/null 2>&1 || curl -sf http://localhost:$port > /dev/null 2>&1; then
                echo "✓ $name (port $port) - HEALTHY"
            else
                echo "✗ $name (port $port) - FAILED"
                if [ -f "/tmp/$(echo $name | tr ' ' '-' | tr '[:upper:]' '[:lower:]').log" ]; then
                    echo "  Last 5 lines of log:"
                    tail -5 "/tmp/$(echo $name | tr ' ' '-' | tr '[:upper:]' '[:lower:]').log" | sed 's/^/    /' || true
                fi
            fi
        done

        # Warm up Ollama model to prevent first-request timeouts
        echo ""
        echo "Warming up Ollama model..."
        if timeout 60 ollama run llama3.1:8b "Hello, King AI is ready." > /dev/null 2>&1; then
            echo "✓ Ollama model warmed up successfully"
        else
            echo "⚠ Ollama model warmup timed out, but keepalive service will handle it"
        fi

        echo ""
        echo "All services started. Logs are in /tmp/"
        echo "To view logs: tail -f /tmp/orchestrator.log"
    fi
    
    cd ~/agentic-framework-main
    cd ~/agentic-framework-main
    
    # Configure Nginx (only if not already configured)
    if [ ! -f /etc/nginx/sites-enabled/king-ai ]; then
        cat > /tmp/nginx.conf << 'NGINX_EOF'
server {
    listen 80 default_server;
    listen [::]:80 default_server;
    server_name _;

    # Increase timeouts for LLM responses
    proxy_connect_timeout 300;
    proxy_send_timeout 300;
    proxy_read_timeout 300;
    send_timeout 300;

    location /health {
        proxy_pass http://127.0.0.1:8000/health;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /api {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }

    location /docs {
        proxy_pass http://127.0.0.1:8000/docs;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
    }

    location /openapi.json {
        proxy_pass http://127.0.0.1:8000/openapi.json;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
    }

    location / {
        proxy_pass http://127.0.0.1:3000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
NGINX_EOF
        sudo rm -f /etc/nginx/sites-enabled/default
        sudo mv /tmp/nginx.conf /etc/nginx/sites-available/king-ai
        sudo ln -sf /etc/nginx/sites-available/king-ai /etc/nginx/sites-enabled/
        sudo nginx -t && sudo systemctl restart nginx
    fi

    # Setup Ollama keepalive service to prevent model unloading
    if [ ! -f /etc/systemd/system/ollama-keepalive.service ]; then
        echo "Setting up Ollama keepalive service..."
        cat > /home/ubuntu/keep_ollama_warm.py << 'KEEPALIVE_EOF'
#!/usr/bin/env python3
import requests
import time
import sys

MODEL = 'llama3.1:8b'
ENDPOINT = 'http://localhost:11434/api/generate'
INTERVAL = 240

print(f'Starting Ollama keepalive for {MODEL}')
sys.stdout.flush()

while True:
    try:
        requests.post(ENDPOINT, json={'model': MODEL, 'prompt': 'ping', 'stream': False}, timeout=300)
        ts = time.strftime('%H:%M:%S')
        print(f'Keepalive ping sent at {ts}')
        sys.stdout.flush()
    except Exception as e:
        print(f'Keepalive error: {e}')
        sys.stdout.flush()
    time.sleep(INTERVAL)
KEEPALIVE_EOF
        chmod +x /home/ubuntu/keep_ollama_warm.py

        sudo tee /etc/systemd/system/ollama-keepalive.service > /dev/null << 'SERVICE_EOF'
[Unit]
Description=Keep Ollama Model Warm
After=ollama.service
Requires=ollama.service

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu
ExecStart=/home/ubuntu/agentic-framework-main/.venv/bin/python3 /home/ubuntu/keep_ollama_warm.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
SERVICE_EOF

        sudo systemctl daemon-reload
        sudo systemctl enable ollama-keepalive
        sudo systemctl start ollama-keepalive
        echo "✓ Ollama keepalive service installed"
    fi
EOF

echo ""
echo "=================================="
echo "Deployment Complete!"
echo "=================================="
echo "Access your application at:"
echo "  http://$aws_ip"
echo "  API: http://$aws_ip/api"
echo "  Docs: http://$aws_ip/docs"
echo "=================================="

exit 0
