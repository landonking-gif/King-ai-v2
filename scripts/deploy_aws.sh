#!/bin/bash
# King AI v3 - AWS Deployment Script
# This script deploys the complete King AI v3 system to an AWS EC2 instance

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="king-ai-v3"
PROJECT_DIR="/home/ubuntu/${PROJECT_NAME}"
BACKUP_DIR="/home/ubuntu/backups"
LOG_FILE="/home/ubuntu/deploy_$(date +%Y%m%d_%H%M%S).log"

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
    exit 1
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to wait for service to be healthy
wait_for_service() {
    local url=$1
    local service_name=$2
    local max_attempts=30
    local attempt=1

    log "Waiting for $service_name to be healthy..."

    while [ $attempt -le $max_attempts ]; do
        if curl -s --max-time 10 "$url" > /dev/null 2>&1; then
            success "$service_name is healthy"
            return 0
        fi

        log "Attempt $attempt/$max_attempts: $service_name not ready yet..."
        sleep 10
        ((attempt++))
    done

    error "$service_name failed to become healthy after $max_attempts attempts"
}

# Main deployment function
main() {
    log "🚀 Starting King AI v3 AWS Deployment"
    log "Log file: $LOG_FILE"

    # Create backup directory
    mkdir -p "$BACKUP_DIR"

    # Update system
    log "📦 Updating system packages..."
    sudo apt update && sudo apt upgrade -y
    sudo apt install -y curl wget git htop

    # Install Docker
    if ! command_exists docker; then
        log "🐳 Installing Docker..."
        curl -fsSL https://get.docker.com -o get-docker.sh
        sudo sh get-docker.sh
        sudo usermod -aG docker ubuntu
        success "Docker installed"
    else
        log "🐳 Docker already installed"
    fi

    # Install Docker Compose
    if ! command_exists docker-compose; then
        log "🐳 Installing Docker Compose..."
        sudo curl -L "https://github.com/docker/compose/releases/download/v2.24.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
        sudo chmod +x /usr/local/bin/docker-compose
        success "Docker Compose installed"
    else
        log "🐳 Docker Compose already installed"
    fi

    # Install Python 3.10+
    if ! command_exists python3.10; then
        log "🐍 Installing Python 3.10..."
        sudo apt install -y software-properties-common
        sudo add-apt-repository -y ppa:deadsnakes/ppa
        sudo apt update
        sudo apt install -y python3.10 python3.10-venv python3.10-dev
        success "Python 3.10 installed"
    else
        log "🐍 Python 3.10 already installed"
    fi

    # Install Node.js 20+
    if ! command_exists node; then
        log "📦 Installing Node.js 20..."
        curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
        sudo apt-get install -y nodejs
        success "Node.js 20 installed"
    else
        log "📦 Node.js already installed"
    fi

    # Create project directory
    if [ -d "$PROJECT_DIR" ]; then
        log "📁 Backing up existing installation..."
        sudo cp -r "$PROJECT_DIR" "$BACKUP_DIR/backup_$(date +%Y%m%d_%H%M%S)"
        sudo rm -rf "$PROJECT_DIR"
    fi

    sudo mkdir -p "$PROJECT_DIR"
    sudo chown ubuntu:ubuntu "$PROJECT_DIR"

    # Copy project files (assuming they're already uploaded)
    if [ -f "/home/ubuntu/king-ai-v3.tar.gz" ]; then
        log "📦 Extracting project files..."
        tar -xzf /home/ubuntu/king-ai-v3.tar.gz -C /home/ubuntu/
        mv /home/ubuntu/king-ai-v3/* "$PROJECT_DIR/" 2>/dev/null || true
        success "Project files extracted"
    else
        error "Project archive not found. Please upload king-ai-v3.tar.gz to /home/ubuntu/"
    fi

    cd "$PROJECT_DIR"

    # Set up Python virtual environment
    log "🐍 Setting up Python virtual environment..."
    python3.10 -m venv venv
    source venv/bin/activate

    # Install Python dependencies
    log "📦 Installing Python dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
    success "Python dependencies installed"

    # Install dashboard dependencies
    log "📦 Installing dashboard dependencies..."
    cd dashboard
    npm install
    cd ..
    success "Dashboard dependencies installed"

    # Configure environment variables
    log "⚙️ Configuring environment variables..."

    # Copy .env files if they don't exist
    for service in orchestrator mcp-gateway memory-service subagent-manager; do
        if [ ! -f "$service/.env" ]; then
            cp "$service/.env.example" "$service/.env" 2>/dev/null || warning "$service/.env.example not found"
        fi
    done

    # Update .env files with correct database credentials
    for env_file in orchestrator/.env mcp-gateway/.env memory-service/.env subagent-manager/.env; do
        if [ -f "$env_file" ]; then
            sed -i 's|POSTGRES_URL=.*|POSTGRES_URL=postgresql://agent_user:agent_pass@localhost:5432/agentic_framework|' "$env_file"
            sed -i 's|LOCAL_MODEL=.*|LOCAL_MODEL=llama3.1:70b|' "$env_file"
            sed -i 's|OLLAMA_ENDPOINT=.*|OLLAMA_ENDPOINT=http://localhost:11434|' "$env_file"
        fi
    done

    success "Environment configured"

    # Start infrastructure services
    log "🏗️ Starting infrastructure services..."
    docker-compose up -d postgres redis chroma
    success "Infrastructure services started"

    # Wait for databases
    log "⏳ Waiting for databases to be ready..."
    sleep 30

    # Run database migrations
    log "🗄️ Running database migrations..."
    source venv/bin/activate
    alembic upgrade head
    success "Database migrations completed"

    # Start Ollama
    log "🧠 Starting Ollama LLM service..."
    docker run -d --name ollama -p 11434:11434 ollama/ollama
    sleep 10

    # Pull the LLM model
    log "⬇️ Pulling llama3.1:70b model..."
    docker exec ollama ollama pull llama3.1:70b
    success "LLM model downloaded"

    # Start microservices
    log "🚀 Starting King AI microservices..."

    # Start in background
    nohup python orchestrator/service/main.py > orchestrator.log 2>&1 &
    sleep 5

    nohup python mcp-gateway/service/main.py > mcp-gateway.log 2>&1 &
    sleep 5

    nohup python memory-service/service/main.py > memory-service.log 2>&1 &
    sleep 5

    nohup python subagent-manager/service/main.py > subagent-manager.log 2>&1 &
    sleep 5

    success "Microservices started"

    # Build and start dashboard
    log "🎨 Building and starting dashboard..."
    cd dashboard
    npm run build
    nohup npm run preview -- --port 3000 --host 0.0.0.0 > dashboard.log 2>&1 &
    cd ..
    success "Dashboard started"

    # Configure Nginx (optional)
    if command_exists nginx; then
        log "🌐 Configuring Nginx reverse proxy..."
        sudo tee /etc/nginx/sites-available/king-ai > /dev/null <<EOF
server {
    listen 80;
    server_name _;

    # Dashboard
    location / {
        proxy_pass http://localhost:3000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }

    # API endpoints
    location /api/ {
        proxy_pass http://localhost:8000/;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF

        sudo ln -sf /etc/nginx/sites-available/king-ai /etc/nginx/sites-enabled/
        sudo rm -f /etc/nginx/sites-enabled/default
        sudo nginx -t && sudo systemctl restart nginx
        success "Nginx configured"
    fi

    # Health checks
    log "🔍 Running health checks..."

    # Wait a bit for services to fully start
    sleep 30

    # Check each service
    wait_for_service "http://localhost:8000/api/health" "Orchestrator"
    wait_for_service "http://localhost:8080/health" "MCP Gateway"
    wait_for_service "http://localhost:8002/health" "Memory Service"
    wait_for_service "http://localhost:8001/health" "Subagent Manager"
    wait_for_service "http://localhost:3000" "Dashboard"

    # Test LLM connectivity
    log "🧠 Testing LLM connectivity..."
    if curl -s "http://localhost:11434/api/tags" | grep -q "llama3.1:70b"; then
        success "LLM service is responding"
    else
        warning "LLM service may not be fully ready yet"
    fi

    # Test chat API
    log "💬 Testing chat API..."
    if curl -s -X POST "http://localhost:8000/api/chat/message" \
        -H "Content-Type: application/json" \
        -d '{"text": "Hello King AI", "session_id": "test"}' | grep -q "response"; then
        success "Chat API is working"
    else
        warning "Chat API test failed - service may still be starting"
    fi

    # Configure firewall
    log "🔥 Configuring firewall..."
    sudo ufw allow 22/tcp
    sudo ufw allow 80/tcp
    sudo ufw allow 3000/tcp
    sudo ufw allow 8000/tcp
    sudo ufw allow 8001/tcp
    sudo ufw allow 8002/tcp
    sudo ufw allow 8080/tcp
    sudo ufw allow 11434/tcp
    sudo ufw --force enable
    success "Firewall configured"

    # Create monitoring script
    log "📊 Creating monitoring script..."
    cat > monitor.sh << 'EOF'
#!/bin/bash
echo "=== King AI v3 System Status ==="
echo "Timestamp: $(date)"

echo -e "\n=== Service Health ==="
services=("Orchestrator:8000" "MCP Gateway:8080" "Memory Service:8002" "Subagent Manager:8001" "Dashboard:3000")

for service in "${services[@]}"; do
    name=$(echo $service | cut -d: -f1)
    port=$(echo $service | cut -d: -f2)

    if curl -s --max-time 5 "http://localhost:$port/api/health" > /dev/null 2>&1; then
        echo "✅ $name: Healthy"
    elif curl -s --max-time 5 "http://localhost:$port/health" > /dev/null 2>&1; then
        echo "✅ $name: Healthy"
    elif [ "$name" = "Dashboard" ] && curl -s --max-time 5 "http://localhost:$port" > /dev/null 2>&1; then
        echo "✅ $name: Healthy"
    else
        echo "❌ $name: Unhealthy"
    fi
done

echo -e "\n=== System Resources ==="
echo "CPU Usage: $(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1"%"}')"
echo "Memory Usage: $(free | grep Mem | awk '{printf "%.2f%%", $3/$2 * 100.0}')"
echo "Disk Usage: $(df / | tail -1 | awk '{print $5}')"

echo -e "\n=== Docker Containers ==="
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

echo -e "\n=== Recent Logs ==="
echo "Last 5 lines from orchestrator:"
tail -5 orchestrator.log 2>/dev/null || echo "No logs available"
EOF

    chmod +x monitor.sh
    success "Monitoring script created"

    # Final success message
    success "🎉 King AI v3 deployment completed successfully!"
    echo ""
    echo "=================================================================="
    echo "🚀 DEPLOYMENT COMPLETE!"
    echo "=================================================================="
    echo ""
    echo "Access your King AI v3 system:"
    echo "🌐 Dashboard:    http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):3000"
    echo "📚 API Docs:     http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):8000/docs"
    echo "💬 Chat API:     http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):8000/api/chat/message"
    echo ""
    echo "Monitoring:"
    echo "📊 Run: ./monitor.sh"
    echo "📝 Logs: View *.log files in project directory"
    echo ""
    echo "Next steps:"
    echo "1. Access the dashboard and try the 'Talk to King AI' tab"
    echo "2. Configure your domain and SSL if needed"
    echo "3. Set up monitoring and alerts"
    echo ""
    echo "=================================================================="

    # Save deployment info
    cat > deployment_info.txt << EOF
King AI v3 Deployment Information
=================================
Deployment Date: $(date)
Server IP: $(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)
AWS Instance: $(curl -s http://169.254.169.254/latest/meta-data/instance-id)

Service URLs:
- Dashboard: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):3000
- Orchestrator API: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):8000
- MCP Gateway: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):8080
- Memory Service: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):8002
- Subagent Manager: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):8001
- Ollama: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):11434

Log Files:
- Deployment: $LOG_FILE
- Orchestrator: orchestrator.log
- MCP Gateway: mcp-gateway.log
- Memory Service: memory-service.log
- Subagent Manager: subagent-manager.log
- Dashboard: dashboard/dashboard.log

Backup Location: $BACKUP_DIR
EOF

    success "Deployment information saved to deployment_info.txt"
}

# Run main function
main "$@"