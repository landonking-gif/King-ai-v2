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
    cd orchestrator

    # Setup environment on AWS server
    echo "Setting up environment on AWS server..."
    /mnt/c/Windows/System32/OpenSSH/ssh.exe -i "$ssh_key" -o StrictHostKeyChecking=no "$ssh_user@$aws_ip" << EOF
        cd ~/agentic-framework-main

        # Install python3-venv, redis, and postgresql
        sudo apt update
        sudo apt install -y python3-venv redis-server postgresql postgresql-contrib

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

        # Copy env file
        cp .env.example .env
        sed -i 's|OLLAMA_ENDPOINT=http://localhost:11434|OLLAMA_ENDPOINT=http://localhost:11434|' .env
        sed -i 's|postgresql://user:password@localhost:5432/agentic_framework|postgresql://agentic_user:agentic_pass@localhost:5432/agentic_framework|' .env

        # Copy .env to service directories
        cp .env mcp-gateway/.env
        cp .env memory-service/.env
        cp .env subagent-manager/.env
EOF

    # Run the services on AWS server
    echo "Starting services on AWS server..."
    /mnt/c/Windows/System32/OpenSSH/ssh.exe -i "$ssh_key" -o StrictHostKeyChecking=no "$ssh_user@$aws_ip" << EOF
        cd ~/agentic-framework-main
        source .venv/bin/activate
        
        # Start MCP Gateway in background
        echo "Starting MCP Gateway..."
        cd mcp-gateway && nohup bash run.sh > mcp-gateway.log 2>&1 &
        
        # Start Memory Service in background
        echo "Starting Memory Service..."
        export MEMORY_SERVICE_PORT=8002
        nohup bash memory-service/run.sh > memory-service.log 2>&1 &
        
        # Start Subagent Manager in background
        echo "Starting Subagent Manager..."
        cd subagent-manager && nohup uvicorn subagent_manager.service.main:app --host 0.0.0.0 --port 8001 > subagent-manager.log 2>&1 &
        
        # Create data directories
        mkdir -p ~/agentic-framework-main/data/chroma
        mkdir -p ~/agentic-framework-main/memory-service/data/chroma

        # Wait a bit for services to start
        sleep 30

        # Check if services are healthy
        echo "Checking service health..."
        curl -f http://localhost:8080/health && echo "MCP Gateway healthy" || echo "MCP Gateway unhealthy"
        curl -f http://localhost:8002/health && echo "Memory Service healthy" || echo "Memory Service unhealthy"
        curl -f http://localhost:8001/health && echo "Subagent Manager healthy" || echo "Subagent Manager unhealthy"
        
        # Kill any process using port 8000
        echo "Checking for processes on port 8000..."
        sudo lsof -ti:8000 | xargs sudo kill -9 2>/dev/null || echo "No process found on port 8000"
        sleep 2
        
        python -m orchestrator.service.main --host 0.0.0.0 --port 8000
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
