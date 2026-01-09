
#!/bin/bash
set -euo pipefail

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >&2
}

error_exit() {
    log "ERROR: $1"
    exit 1
}

trap 'error_exit "Application deployment failed at line $LINENO"' ERR

# Go to project directory
cd ~/king-ai-v2 || error_exit "Project directory not found"

# Install Python dependencies
log "Installing Python dependencies..."
source ~/venv/bin/activate
pip install -e . || error_exit "Python dependencies installation failed"

# Run database migrations
log "Running database migrations..."
alembic upgrade heads || error_exit "Database migrations failed"

# Start Ollama
log "Starting Ollama service..."
if ! pgrep -f "ollama serve" > /dev/null; then
    ollama serve &
    sleep 5
fi

# Pull default model
timeout 300 ollama pull llama3.2:1b || log "Model download timed out"

# Start API server
log "Cleaning up old API processes..."
pkill -f uvicorn || true
fuser -k 8000/tcp || true

log "Starting API server..."
nohup ~/venv/bin/uvicorn src.api.main:app --host 0.0.0.0 --port 8000 > api.log 2>&1 &
API_PID=$!


# Wait for API to be ready
log "Waiting for API server..."
for i in {1..30}; do
    if curl -s --max-time 5 http://localhost:8000/health > /dev/null 2>&1; then
        log "API server is ready"
        break
    fi
    sleep 2
done

if ! curl -s --max-time 5 http://localhost:8000/health > /dev/null 2>&1; then
    error_exit "API server failed to start"
fi

log "Cleaning up old dashboard processes..."
pkill -f "npm run dev" || true
fuser -k 5173/tcp || true

log "Starting dashboard..."
cd dashboard
npm install --silent || error_exit "npm install failed"
nohup npm run dev -- --host 0.0.0.0 --port 5173 > dashboard.log 2>&1 &
DASHBOARD_PID=$!

# Wait for dashboard
sleep 10
if ! ps -p $DASHBOARD_PID > /dev/null 2>&1; then
    error_exit "Dashboard failed to start"
fi

cd ..

# Configure Nginx as reverse proxy on port 80
log "Configuring Nginx reverse proxy..."
sudo tee /etc/nginx/sites-available/king-ai > /dev/null << 'NGINX_EOF'
server {
    listen 80 default_server;
    listen [::]:80 default_server;
    server_name _;

    # Health check endpoint for load balancer
    location /health {
        proxy_pass http://127.0.0.1:8000/health;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # API endpoints
    location /api {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # API docs
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

    # Dashboard (default)
    location / {
        proxy_pass http://127.0.0.1:5173;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
NGINX_EOF

# Enable site and restart Nginx
sudo rm -f /etc/nginx/sites-enabled/default
sudo ln -sf /etc/nginx/sites-available/king-ai /etc/nginx/sites-enabled/king-ai
sudo nginx -t || error_exit "Nginx configuration test failed"
sudo systemctl restart nginx || error_exit "Nginx restart failed"
log "Nginx configured on port 80"

log "Application deployment completed successfully"
