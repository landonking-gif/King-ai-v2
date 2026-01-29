#!/bin/bash
# set -e  # Temporarily disable exit on error for debugging
export DEBIAN_FRONTEND=noninteractive

echo "===================================="
echo "King AI v3 Server Setup Starting"
echo "===================================="
echo ""

# Function for progress updates
log_progress() {
    date_time=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$date_time] $1"
}

log_progress "Starting King AI v3 deployment..."

# Clean up any previous deployment
log_progress "Cleaning up previous deployment..."
rm -rf agentic-framework-main 2>/dev/null || (chmod -R u+w agentic-framework-main 2>/dev/null && rm -rf agentic-framework-main 2>/dev/null) || log_progress "Warning: Could not clean up previous deployment"
mkdir -p agentic-framework-main

log_progress "Extracting project files..."
cd ~
# Try to extract to /tmp first to avoid permission issues
mkdir -p /tmp/agentic-extract
cd /tmp/agentic-extract

# Extract to temp location
unzip -q ~/deploy.zip 2>/dev/null || {
    log_progress "Standard unzip failed, trying with different options..."
    unzip -q -o ~/deploy.zip 2>/dev/null || {
        log_progress "All unzip methods failed, trying Python extraction..."
        python3 -c "
import zipfile
import os
import sys
import stat

try:
    with zipfile.ZipFile('/home/ubuntu/deploy.zip', 'r') as zip_ref:
        for member in zip_ref.namelist():
            # Skip directories
            if member.endswith('/'):
                continue
            
            # Create directory if needed
            dir_path = os.path.dirname(member)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            
            # Extract file
            with open(member, 'wb') as f:
                f.write(zip_ref.read(member))
            
            # Set proper permissions (readable/writable by owner)
            os.chmod(member, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
                
        print('Python extraction completed')
except Exception as e:
    print(f'Python extraction failed: {e}')
    sys.exit(1)
" || {
    log_progress "All extraction methods failed"
    exit 1
}
    }
}

# Move extracted files to final location
if [ -d "agentic-framework-main" ]; then
    mv agentic-framework-main /home/ubuntu/
fi

if [ -d "dashboard" ]; then
    mv dashboard /home/ubuntu/
fi

# Ensure we're in the agentic-framework-main directory
if [ -d "/home/ubuntu/agentic-framework-main" ]; then
    cd /home/ubuntu/agentic-framework-main
else
    log_progress "ERROR: agentic-framework-main directory not found after extraction"
    exit 1
fi

# Clean up temp directory (force remove with proper permissions)
chmod -R u+rwX /tmp/agentic-extract 2>/dev/null || true
rm -rf /tmp/agentic-extract

# Verify extraction
if [ ! -f "requirements.txt" ]; then
    log_progress "ERROR: Extraction failed - requirements.txt not found"
    exit 1
fi

# Fix permissions on extracted files
log_progress "Fixing file permissions..."
chmod -R u+rwX .
find . -type f -name "*.sh" -exec chmod +x {} \;
find . -type f -name "*.py" -exec chmod +x {} \;

cd agentic-framework-main
log_progress "Successfully extracted and entered project directory"

log_progress "Updating system packages..."
sudo apt-get update -qq
sudo apt-get install -y python3-pip python3-venv postgresql redis-server nginx curl htop 2>/dev/null

# Install Node.js for dashboard
log_progress "Installing Node.js..."
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash - 2>/dev/null
sudo apt-get install -y nodejs 2>/dev/null

log_progress "Starting system services..."
sudo systemctl start postgresql redis-server 2>/dev/null || true
sudo systemctl enable postgresql redis-server 2>/dev/null || true

log_progress "Setting up database..."
sudo -u postgres psql -c "SELECT 1" >/dev/null 2>&1 || sudo service postgresql start
sudo -u postgres psql -c "CREATE USER agentic_user WITH PASSWORD 'agentic_pass';" 2>/dev/null || log_progress "User already exists"
sudo -u postgres psql -c "CREATE DATABASE agentic_framework OWNER agentic_user;" 2>/dev/null || log_progress "Database already exists"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE agentic_framework TO agentic_user;" 2>/dev/null || log_progress "Privileges already granted"

log_progress "Creating Python virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

log_progress "Installing Python dependencies (this may take several minutes)..."
pip install --upgrade pip
pip install --quiet --use-deprecated=legacy-resolver -r requirements.txt

log_progress "Stopping old services..."
pkill -f 'python.*orchestrator' 2>/dev/null || true
pkill -f 'python.*mcp-gateway' 2>/dev/null || true
pkill -f 'python.*memory-service' 2>/dev/null || true
pkill -f 'python.*subagent-manager' 2>/dev/null || true
sleep 3

# Clear Python cache files to prevent stale imports
log_progress "Clearing Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true

log_progress "Starting new services..."

# Start Orchestrator
if [ -d "orchestrator/service" ]; then
    cd orchestrator
    export PYTHONPATH="$PWD:$PWD/..:$PYTHONPATH"
    source ../.venv/bin/activate
    nohup python run_service.py > /tmp/orchestrator.log 2>&1 &
    echo $! > /tmp/orchestrator.pid
    cd ..
    echo "  Orchestrator started (PID: $(cat /tmp/orchestrator.pid))"
else
    echo "  Orchestrator directory not found, skipping"
fi

# Start MCP Gateway
if [ -d "mcp-gateway/service" ]; then
    cd mcp-gateway
    export PYTHONPATH="$PWD:$PWD/..:$PYTHONPATH"
    source ../.venv/bin/activate
    nohup python -m service.main > /tmp/mcp-gateway.log 2>&1 &
    echo $! > /tmp/mcp-gateway.pid
    cd ..
    echo "  MCP Gateway started (PID: $(cat /tmp/mcp-gateway.pid))"
else
    echo "  MCP Gateway directory not found, skipping"
fi

# Start Memory Service
if [ -d "memory-service/service" ]; then
    cd memory-service
    export PYTHONPATH="$PWD:$PWD/..:$PYTHONPATH"
    source ../.venv/bin/activate
    nohup uvicorn service.main:app --host 0.0.0.0 --port 8002 > /tmp/memory-service.log 2>&1 &
    echo $! > /tmp/memory-service.pid
    cd ..
    echo "  Memory Service started (PID: $(cat /tmp/memory-service.pid))"
else
    echo "  Memory Service directory not found, skipping"
fi

# Start Subagent Manager
if [ -d "subagent-manager/service" ]; then
    cd subagent-manager
    export PYTHONPATH="$PWD:$PWD/..:$PYTHONPATH"
    source ../.venv/bin/activate
    nohup python -m service.main > /tmp/subagent-manager.log 2>&1 &
    echo $! > /tmp/subagent-manager.pid
    cd ..
    echo "  Subagent Manager started (PID: $(cat /tmp/subagent-manager.pid))"
else
    echo "  Subagent Manager directory not found, skipping"
fi

# Build and start Dashboard
log_progress "Building and starting dashboard..."
if [ -d "/home/ubuntu/dashboard" ] && [ -f "/home/ubuntu/dashboard/package.json" ]; then
    # Stop any existing dashboard process
    pkill -f 'serve.*3000' 2>/dev/null || true
    pkill -f 'node.*3000' 2>/dev/null || true
    
    cd /home/ubuntu/dashboard
    log_progress "  Installing dashboard dependencies..."
    npm install 2>&1 | tee -a /tmp/dashboard.log
    
    log_progress "  Building dashboard..."
    npm run build 2>&1 | tee -a /tmp/dashboard.log
    
    if [ -d "dist" ] || [ -d "build" ]; then
        # Install serve globally if not available
        if ! command -v serve &> /dev/null; then
            log_progress "  Installing serve globally..."
            sudo npm install -g serve 2>&1 | tee -a /tmp/dashboard.log
        fi
        
        # Determine build output directory
        if [ -d "dist" ]; then
            BUILD_DIR="dist"
        else
            BUILD_DIR="build"
        fi
        
        # Start dashboard server
        log_progress "  Starting dashboard server..."
        nohup serve -s -p 3000 $BUILD_DIR > /tmp/dashboard.log 2>&1 &
        echo $! > /tmp/dashboard.pid
        echo "  Dashboard started (PID: $(cat /tmp/dashboard.pid))"
        
        # Verify dashboard is running
        sleep 3
        if kill -0 $(cat /tmp/dashboard.pid) 2>/dev/null; then
            log_progress "  Dashboard is running successfully"
        else
            log_progress "  WARNING: Dashboard may have failed to start, check /tmp/dashboard.log"
        fi
    else
        log_progress "  ERROR: Dashboard build failed - no dist or build directory found"
    fi
    
    cd /home/ubuntu/agentic-framework-main
else
    log_progress "  Dashboard directory not found - dashboard will not be available"
    log_progress "  API will still be accessible at /api and /docs"
fi

log_progress "Waiting for services to initialize..."
sleep 10

# Configure Nginx with proper proxy settings
log_progress "Configuring Nginx..."
cat > /tmp/nginx-king-ai.conf << 'EOF'
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

    location /favicon.ico {
        return 204;
        access_log off;
        log_not_found off;
    }

    # Dashboard - proxy to dashboard service with graceful fallback
    location / {
        proxy_pass http://127.0.0.1:3000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        
        # If dashboard is down, show friendly message instead of 502
        proxy_intercept_errors on;
        error_page 502 503 504 = @fallback;
    }

    location @fallback {
        default_type application/json;
        return 200 '{"service":"King AI v3","status":"running","message":"Dashboard not available. Access API at /api or docs at /docs","health":"/health","api_docs":"/docs"}';
    }
}
EOF

sudo rm -f /etc/nginx/sites-enabled/default
sudo mv /tmp/nginx-king-ai.conf /etc/nginx/sites-available/king-ai
sudo ln -sf /etc/nginx/sites-available/king-ai /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl restart nginx
log_progress "Nginx configured and restarted"

# =====================================================================
# MoltBot Multi-Channel Gateway Deployment
# =====================================================================
log_progress "Deploying MoltBot multi-channel gateway..."

# Check Node.js version
NODE_VERSION=$(node --version | cut -d'v' -f2 | cut -d'.' -f1)
if [ "$NODE_VERSION" -lt 22 ]; then
    log_progress "Upgrading Node.js to v22+ (required by MoltBot)..."
    curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
    sudo apt-get install -y nodejs
    log_progress "Node.js upgraded to $(node --version)"
fi

# Install pnpm if not present
if ! command -v pnpm &> /dev/null; then
    log_progress "Installing pnpm..."
    sudo npm install -g pnpm
fi

# Clone MoltBot if not already present
cd /home/ubuntu/king-ai-v3
if [ ! -d "moltbot" ]; then
    log_progress "Cloning MoltBot repository..."
    git clone https://github.com/moltbot/moltbot.git
fi

# Install MoltBot dependencies
cd moltbot
log_progress "Installing MoltBot dependencies..."
pnpm install --ignore-scripts 2>&1 | tail -20

# Build MoltBot
log_progress "Building MoltBot..."
pnpm build 2>&1 | tail -20

# Build UI
log_progress "Building MoltBot UI..."
pnpm ui:build 2>&1 | tail -20

# Create MoltBot config if not exists
mkdir -p ~/.moltbot
if [ ! -f ~/.moltbot/moltbot.json ]; then
    log_progress "Creating MoltBot configuration (using local DeepSeek R1 - no API keys required)..."
    cat > ~/.moltbot/moltbot.json << 'MOLTBOT_CONFIG'
{
  gateway: {
    port: 18789,
    mode: "local",
    auth: {
      mode: "token",
      token: "kingai-moltbot-token-2026"
    }
  },
  
  agents: {
    defaults: {
      workspace: "~/king-ai-workspace",
      model: { 
        primary: "ollama/deepseek-r1:7b"
      },
      models: {
        "ollama/deepseek-r1:7b": { alias: "DeepSeek-Local" },
        "kingai/deepseek-r1": { alias: "DeepSeek-API" }
      }
    }
  },
  
  models: {
    mode: "merge",
    providers: {
      ollama: {
        baseUrl: "http://localhost:11434/v1",
        apiKey: "not-required-local-only",
        api: "openai-completions",
        models: [{
          id: "deepseek-r1:7b",
          name: "DeepSeek-R1-7B-Local",
          input: ["text"],
          contextWindow: 32000,
          maxTokens: 4096
        }]
      },
      kingai: {
        baseUrl: "http://localhost:8000/v1",
        apiKey: "not-required-local-only",
        api: "openai-completions",
        models: [{
          id: "deepseek-r1",
          name: "DeepSeek-R1-via-Orchestrator",
          input: ["text"],
          contextWindow: 32000,
          maxTokens: 4096
        }]
      }
    }
  },
  
  channels: {
    telegram: { allowFrom: ["*"], groups: { "*": { requireMention: true } } },
    discord: { guilds: { "*": { requireMention: true } }, dm: { policy: "pairing", allowFrom: [] } },
    slack: { dm: { policy: "pairing", allowFrom: [] } },
    whatsapp: { allowFrom: [], groups: { "*": { requireMention: true } } }
  }
}
MOLTBOT_CONFIG
    log_progress "MoltBot configured to use local DeepSeek R1 7B via Ollama (no external API keys needed)"
fi

# Start MoltBot gateway
log_progress "Starting MoltBot gateway..."
nohup pnpm moltbot gateway --port 18789 > /tmp/moltbot.log 2>&1 &
echo $! > /tmp/moltbot.pid
sleep 5

# Verify MoltBot started
if kill -0 $(cat /tmp/moltbot.pid) 2>/dev/null; then
    log_progress "MoltBot gateway started successfully (PID: $(cat /tmp/moltbot.pid))"
else
    log_progress "WARNING: MoltBot gateway may have failed to start. Check /tmp/moltbot.log"
fi

log_progress "Checking service health..."
echo ""
echo "Service Status:"
echo "==============="
HEALTHY=0
TOTAL=6
for service in orchestrator mcp-gateway memory-service subagent-manager dashboard moltbot; do
    if [ -f /tmp/${service}.pid ] && kill -0 $(cat /tmp/${service}.pid) 2>/dev/null; then
        echo "  ✓ $service is running (PID: $(cat /tmp/${service}.pid))"
        HEALTHY=$((HEALTHY + 1))
    else
        echo "  ✗ $service failed to start or not deployed"
    fi
done

echo ""
if [ $HEALTHY -eq $TOTAL ]; then
    echo "===================================="
    echo "DEPLOYMENT SUCCESSFUL!"
    echo "===================================="
    echo "All $HEALTHY/$TOTAL services are running"
elif [ $HEALTHY -ge 4 ]; then
    echo "===================================="
    echo "DEPLOYMENT MOSTLY SUCCESSFUL"
    echo "===================================="
    echo "$HEALTHY/$TOTAL services are running"
    echo "Dashboard may not be available but API services are operational"
else
    echo "===================================="
    echo "DEPLOYMENT COMPLETED WITH WARNINGS"
    echo "===================================="
    echo "$HEALTHY/$TOTAL services are running"
    echo "Check logs for details"
fi

echo ""
echo "Service Health URLs (via localhost):"
echo "  curl http://localhost/health (Nginx proxy)"
echo "  curl http://localhost:8000/health (Orchestrator)"
echo "  curl http://localhost:8001/health (Subagent Manager)"
echo "  curl http://localhost:8002/health (Memory Service)"
echo "  curl http://localhost:8080/health (MCP Gateway)"
echo "  curl http://localhost:18789/ (MoltBot Gateway)"
if [ -f /tmp/dashboard.pid ] && kill -0 $(cat /tmp/dashboard.pid) 2>/dev/null; then
    echo "  curl http://localhost:3000/ (Dashboard)"
fi

echo ""
echo "Service logs available at:"
echo "  /tmp/orchestrator.log"
echo "  /tmp/mcp-gateway.log"
echo "  /tmp/memory-service.log"
echo "  /tmp/subagent-manager.log"
echo "  /tmp/dashboard.log"
echo "  /tmp/moltbot.log"
echo ""
echo "MoltBot Multi-Channel Access:"
echo "  Control UI: http://localhost:18789 (or http://YOUR_SERVER_IP:18789)"
echo "  Configure channels in: ~/.moltbot/moltbot.json"
echo "  Supported: Telegram, Discord, Slack, WhatsApp, Signal, Google Chat, Matrix"
echo ""
