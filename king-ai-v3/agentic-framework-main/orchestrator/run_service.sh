#!/bin/bash
# Quick start script for the Orchestrator service

set -e

echo "=================================="
echo "Orchestrator Service Quick Start"
echo "=================================="
echo ""

# Automatically deploy to AWS server
run_on_aws=y
read -p "Enter AWS server IP address: " aws_ip
# Update dashboard .env with the new IP
/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe -Command "Set-Content -Path 'C:\Users\dmilner.AGV-040318-PC\Downloads\landon\king-ai-v2\dashboard\.env' -Value 'VITE_API_BASE=http://$aws_ip:8000/api'"
# Create orchestrator .env with AWS_IP
cp ../.env.example .env
echo "AWS_IP=$aws_ip" >> .env
ssh_key="$(cd .. && pwd)/king-ai-studio.pem"
ssh_user=ubuntu

if [[ $run_on_aws =~ ^[Yy]$ ]]; then
    # If relative path, assume it's in the project root
    if [[ $ssh_key != /* ]]; then
        ssh_key="$(cd .. && pwd)/$ssh_key"
    fi
    # Convert WSL path to Windows path for Windows SSH
    ssh_key=${ssh_key/\/mnt\/c\//C:\\}
    ssh_key=${ssh_key//\//\\}

    echo "Deploying to AWS server $aws_ip..."

    # Install Ollama on AWS server
    echo "Installing Ollama on AWS server..."
    /mnt/c/Windows/System32/OpenSSH/ssh.exe -i "$ssh_key" -o StrictHostKeyChecking=no "$ssh_user@$aws_ip" << 'EOF'
        # Update system
        sudo apt update
        sudo apt install -y curl

        # Install Ollama
        curl -fsSL https://ollama.ai/install.sh | sh

        # Start Ollama service
        sudo systemctl enable ollama
        sudo systemctl start ollama

        # Pull default model
        ollama pull llama3.1:70b
EOF

    # Copy project to AWS server
    echo "Copying project to AWS server..."
    cd ..
    rsync -avz -e "/mnt/c/Windows/System32/OpenSSH/ssh.exe -i \"$ssh_key\" -o StrictHostKeyChecking=no" --exclude='.venv' --exclude='__pycache__' --exclude='.git' . "$ssh_user@$aws_ip:~/agentic-framework-main"
    
    # Copy dashboard from parent directory
    echo "Copying dashboard to AWS server..."
    rsync -avz -e "/mnt/c/Windows/System32/OpenSSH/ssh.exe -i \"$ssh_key\" -o StrictHostKeyChecking=no" --exclude='node_modules' --exclude='.git' ../../dashboard/ "$ssh_user@$aws_ip:~/agentic-framework-main/dashboard/"
    cd orchestrator

    # Fix line endings on AWS server
    echo "Fixing line endings on AWS server..."
    /mnt/c/Windows/System32/OpenSSH/ssh.exe -i "$ssh_key" -o StrictHostKeyChecking=no "$ssh_user@$aws_ip" << 'EOF'
        cd ~/agentic-framework-main
        sed -i 's/\r$//' mcp-gateway/run.sh
        sed -i 's/\r$//' memory-service/run.sh
        sed -i 's/\r$//' subagent-manager/run.sh
        sed -i 's/\r$//' orchestrator/run_service.sh
        echo "Line endings fixed"
EOF

    # Setup environment on AWS server
    echo "Setting up environment on AWS server..."
    /mnt/c/Windows/System32/OpenSSH/ssh.exe -i "$ssh_key" -o StrictHostKeyChecking=no "$ssh_user@$aws_ip" << EOF
        cd ~/agentic-framework-main

        # Install python3-venv, redis, postgresql, and nginx
        sudo apt update
        sudo apt install -y python3-venv redis-server postgresql postgresql-contrib wget nginx

        # Install Node.js and npm for dashboard
        curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
        sudo apt-get install -y nodejs

        # Start services
        sudo systemctl enable redis-server
        sudo systemctl start redis-server
        sudo systemctl enable postgresql
        sudo systemctl start postgresql

        # Setup PostgreSQL
        sudo -u postgres psql -c "CREATE USER agentic_user WITH PASSWORD 'agentic_pass';"
        sudo -u postgres psql -c "CREATE DATABASE agentic_framework OWNER agentic_user;"
        sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE agentic_framework TO agentic_user;"

        # Create virtual environment
        python3 -m venv .venv
        source .venv/bin/activate

        # Install dependencies (upgrade pip first, then core, then all)
        pip install --upgrade pip
        pip install pydantic fastapi uvicorn pydantic-settings httpx sqlmodel alembic redis psycopg2-binary pyyaml
        pip install -r requirements.txt

        # Install MinIO
        pkill -f minio || true
        rm -rf ~/minio
        wget https://dl.min.io/server/minio/release/linux-amd64/minio -O ~/minio
        chmod +x ~/minio

        # Copy env file
        cp .env.example .env
        sed -i 's|OLLAMA_ENDPOINT=http://localhost:11434|OLLAMA_ENDPOINT=http://localhost:11434|' .env
        sed -i 's|postgresql://user:password@localhost:5432/agentic_framework|postgresql://agentic_user:agentic_pass@localhost:5432/agentic_framework|' .env
        echo "MEMORY_SERVICE_PORT=8002" >> .env
        echo "AWS_IP=$aws_ip" >> .env

        # Copy .env to service directories
        cp .env mcp-gateway/.env
        cp .env memory-service/.env
        cp .env subagent-manager/.env
        cp .env orchestrator/.env

        # Setup dashboard
        echo "Setting up dashboard..."
        cd dashboard
        npm install
        npm run build
        cd ..
EOF

    # Run the services on AWS server
    echo "Starting services on AWS server..."
    /mnt/c/Windows/System32/OpenSSH/ssh.exe -i "$ssh_key" -o StrictHostKeyChecking=no "$ssh_user@$aws_ip" << EOF
        cd ~/agentic-framework-main
        source .venv/bin/activate
        
        # Start MinIO
        mkdir -p ~/agentic-framework-main/data/minio
        nohup ~/minio server ~/agentic-framework-main/data/minio --console-address :9001 > minio.log 2>&1 &
        
        # Start MCP Gateway in background
        echo "Starting MCP Gateway..."
        cd mcp-gateway
        nohup bash run.sh > mcp-gateway.log 2>&1 &
        cd ..
        
        # Start Memory Service in background
        echo "Starting Memory Service..."
        cd memory-service
        nohup bash run.sh > memory-service.log 2>&1 &
        cd ..
        
        # Start Subagent Manager in background
        echo "Starting Subagent Manager..."
        cd subagent-manager
        nohup bash run.sh > subagent-manager.log 2>&1 &
        cd ..
        
        # Start Dashboard
        echo "Starting Dashboard..."
        cd dashboard
        nohup npm run preview -- --host 0.0.0.0 --port 3000 > dashboard.log 2>&1 &
        cd ..

        # Configure and start Nginx
        echo "Configuring Nginx..."
        sudo rm -f /etc/nginx/sites-enabled/default
        cd ~
        cat > nginx.conf << 'NGINX_EOF'
server {
    listen 80 default_server;
    listen [::]:80 default_server;
    server_name _;

    # Health check endpoint
    location /health {
        proxy_pass http://127.0.0.1:8000/health;
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }

    # Unified API routing
    location /api {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }

    # API docs
    location /docs {
        proxy_pass http://127.0.0.1:8000/docs;
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
    }

    location /openapi.json {
        proxy_pass http://127.0.0.1:8000/openapi.json;
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
    }

    # Dashboard (default)
    location / {
        proxy_pass http://127.0.0.1:3000;
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
NGINX_EOF
        sudo cp nginx.conf /etc/nginx/sites-available/king-ai
        sudo ln -sf /etc/nginx/sites-available/king-ai /etc/nginx/sites-enabled/
        sudo nginx -t
        sudo systemctl restart nginx
        sudo systemctl enable nginx
        cd ~/agentic-framework-main
        
        # Create data directories
        mkdir -p ~/agentic-framework-main/data/chroma
        mkdir -p ~/agentic-framework-main/memory_service/data/chroma

        # Wait a bit for services to start
        sleep 30

        # Check if services are healthy
        echo "Checking service health..."
        if curl -f http://localhost:8080/health; then
            echo "MCP Gateway healthy"
        else
            echo "MCP Gateway unhealthy"
            echo "MCP Gateway log:"
            cat mcp-gateway/mcp-gateway.log 2>/dev/null || echo "No log file"
        fi
        
        if curl -f http://localhost:8002/health; then
            echo "Memory Service healthy"
        else
            echo "Memory Service unhealthy"
            echo "Memory Service log:"
            cat memory-service/memory-service.log 2>/dev/null || echo "No log file"
        fi
        
        if curl -f http://localhost:8001/health; then
            echo "Subagent Manager healthy"
        else
            echo "Subagent Manager unhealthy"
            echo "Subagent Manager log:"
            cat subagent-manager/subagent-manager.log 2>/dev/null || echo "No log file"
        fi
        
        # Kill any process using port 8000
        echo "Checking for processes on port 8000..."
        sudo lsof -ti:8000 | xargs sudo kill -9 2>/dev/null || echo "No process found on port 8000"
        sleep 2
        
        source /home/ubuntu/agentic-framework-main/.venv/bin/activate && python -m orchestrator.service.main --host 0.0.0.0 --port 8000
EOF

    exit 0
fi

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found. Creating it now..."
    uv venv --python 3.11
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Check if dependencies are installed
echo "Checking dependencies..."
if ! python -c "import fastapi, httpx" 2>/dev/null; then
    echo "❌ Dependencies not installed. Installing now..."
    pip install --upgrade pip
    pip install pydantic fastapi uvicorn pydantic-settings httpx
    pip install -r requirements.txt || echo "⚠️  Full requirements install failed, but core packages are installed. Service may have limited functionality."
    echo "✓ Dependencies installed"
else
    echo "✓ Dependencies already installed"
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found. Copying from .env.example..."
    cp .env.example .env
    echo "✓ Created .env file. Please edit it with your configuration."
    echo ""
    echo "Required configuration:"
    echo "  - ANTHROPIC_API_KEY or OPENAI_API_KEY"
    echo "  - Database URLs (POSTGRES_URL, REDIS_URL)"
    echo "  - Service URLs for dependent services"
    echo ""
    read -p "Press Enter to continue after configuring .env..."
fi

# Run code quality checks
echo ""
echo "Running code quality checks..."
echo "--------------------------------"

echo "1. Formatting with Black..."
black --check --line-length 100 orchestrator/service/*.py || {
    echo "⚠️  Code formatting issues found. Run: black --line-length 100 orchestrator/"
}

echo ""
echo "2. Type checking with mypy..."
mypy --strict orchestrator/service/*.py || {
    echo "⚠️  Type checking issues found. Please review."
}

echo ""
echo "All checks completed!"
echo ""

# Start the service
echo "=================================="
echo "Starting Orchestrator Service"
echo "=================================="
echo ""
echo "Service will be available at: http://localhost:8001"
echo "API documentation at: http://localhost:8001/docs"
echo ""
echo "Press Ctrl+C to stop the service"
echo ""

# Kill any process using port 8001
echo "Checking for processes on port 8001..."
sudo lsof -ti:8001 | xargs sudo kill -9 2>/dev/null || echo "No process found on port 8001"
sleep 2

# Run the service
python -m orchestrator.service.main --host 0.0.0.0 --port 8001
