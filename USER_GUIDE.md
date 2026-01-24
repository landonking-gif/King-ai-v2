# King AI v3 - Complete User Guide

**Discipline. Autonomy. Success.**

This comprehensive guide covers everything you need to know about King AI v3 - from initial setup to operating your autonomous business empire. King AI v3 features a microservices architecture with Ollama LLM integration, real-time dashboard, and automated AWS deployment.

## 🟢 Current Production Deployment

### Live System Status
- **Domain**: https://king-ai-studio.me
- **Status**: ✅ Operational (AWS EC2 + Load Balancer)
- **Infrastructure**: AWS EC2 t3.medium, Docker containers, Nginx reverse proxy
- **SSL**: ✅ Enabled via AWS Load Balancer
- **Services**: Orchestrator, MCP Gateway, Memory Service, Subagent Manager, Dashboard

### Quick Access Points
- **Dashboard**: https://king-ai-studio.me
- **API Documentation**: https://king-ai-studio.me/api/docs
- **Health Check**: https://king-ai-studio.me/api/health
- **Direct API**: https://king-ai-studio.me/api/

### System Architecture
- **Orchestrator** (port 8000): Main AI brain, workflow management, chat interface
- **MCP Gateway** (port 8080): Model Context Protocol for tool integration
- **Memory Service** (port 8002): Long-term memory and vector storage
- **Subagent Manager** (port 8001): Manages specialized AI agents
- **Dashboard** (port 3000): React frontend with real-time updates
- **Ollama** (port 11434): Local LLM runtime with llama3.1:70b model

## 🚀 AWS Deployment Guide

### Prerequisites for AWS Deployment

wsl bash -c "cd /mnt/c/Users/dmilner.AGV-040318-PC/Downloads/landon/king-ai-v2/king-ai-v3/agentic-framework-main/orchestrator && sed -i 's/\r$//' run_service.sh && echo '54.167.201.176' | bash run_service.sh"

#### 1. AWS Account Setup
- Create AWS account with EC2 access
- Launch EC2 instance (t3.medium or larger recommended)
- Ubuntu 22.04 LTS AMI
- Security group with ports: 22, 80, 443, 3000, 8000-8003, 8080, 11434
- Key pair (.pem file) for SSH access

#### 2. Local Development Environment
- Python 3.10+
- Node.js 20+
- Git
- SSH client
- AWS CLI (optional)

### Automated AWS Deployment (Recommended)

The easiest way to deploy King AI v3 to AWS is using the automated deployment script:

#### Step 1: Prepare Your Local Environment
```bash
# Clone the repository
git clone <your-repo-url>
cd king-ai-v3/agentic-framework-main

# Ensure you have the SSH key
# Copy your AWS .pem key to the project root
cp ~/Downloads/your-key.pem king-ai-studio.pem
```

#### Step 2: Upload and Run Deployment Script
```bash
# On your local machine, upload the deployment script
scp -i "your-key.pem" scripts/deploy_aws.sh ubuntu@YOUR_EC2_IP:~/

# On AWS server, make it executable and run
chmod +x deploy_aws.sh
./deploy_aws.sh
```

The automated deployment script performs these steps:

1. **System Preparation**
   - Updates Ubuntu packages
   - Installs Docker and Docker Compose
   - Installs Python 3.10+ and Node.js 20+
   - Configures system for King AI v3

2. **Project Setup**
   - Extracts uploaded project files
   - Sets up Python virtual environment
   - Installs all dependencies
   - Configures environment variables

3. **Infrastructure Deployment**
   - Starts PostgreSQL, Redis, and ChromaDB containers
   - Runs database migrations
   - Initializes vector storage

4. **LLM Setup**
   - Starts Ollama container
   - Downloads llama3.1:70b model
   - Verifies LLM connectivity

5. **Service Startup**
   - Launches all microservices (Orchestrator, MCP Gateway, Memory Service, Subagent Manager)
   - Builds and starts React dashboard
   - Configures Nginx reverse proxy (optional)

6. **Health Verification**
   - Tests all service endpoints
   - Validates LLM integration
   - Confirms dashboard accessibility

7. **Security & Monitoring**
   - Configures firewall rules
   - Sets up monitoring script
   - Creates deployment documentation

### Manual AWS Deployment (Alternative)

If you prefer manual control:

#### Step 1: Connect to Your EC2 Instance
```bash
ssh -i "your-key.pem" ubuntu@YOUR_EC2_IP
```

#### Step 2: Update System and Install Dependencies
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.24.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Install Python and Node.js
sudo apt install -y python3.10 python3.10-venv nodejs npm
```

#### Step 3: Deploy Application Code
```bash
# On your local machine, create deployment archive
cd king-ai-v3/agentic-framework-main
tar -czf king-ai-deploy.tar.gz --exclude="node_modules" --exclude=".git" --exclude="__pycache__" .

# Upload to AWS
scp -i "your-key.pem" king-ai-deploy.tar.gz ubuntu@YOUR_EC2_IP:~/

# On AWS server
cd ~
tar -xzf king-ai-deploy.tar.gz
cd king-ai-v3/agentic-framework-main
```

#### Step 4: Configure Environment
```bash
# Copy and edit environment files
cp orchestrator/.env.example orchestrator/.env
cp mcp-gateway/.env.example mcp-gateway/.env
cp memory-service/.env.example memory-service/.env
cp subagent-manager/.env.example subagent-manager/.env

# Edit .env files with your configuration
nano orchestrator/.env
# Set: OLLAMA_ENDPOINT=http://localhost:11434
# Set: LOCAL_MODEL=llama3.1:70b
# Set: POSTGRES_URL=postgresql://agent_user:agent_pass@localhost:5432/agentic_framework
```

#### Step 5: Start Services
```bash
# Start infrastructure (PostgreSQL, Redis, ChromaDB)
docker-compose up -d postgres redis chroma

# Wait for databases to be ready
sleep 30

# Install Python dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Start Ollama
docker run -d --name ollama -p 11434:11434 ollama/ollama
docker exec ollama ollama pull llama3.1:70b

# Start microservices
python orchestrator/service/main.py &
python mcp-gateway/service/main.py &
python memory-service/service/main.py &
python subagent-manager/service/main.py &

# Build and start dashboard
cd dashboard
npm install
npm run build
npm run preview &
```

#### Step 6: Configure Nginx (Optional)
```bash
# Install Nginx
sudo apt install -y nginx

# Configure reverse proxy
sudo nano /etc/nginx/sites-available/king-ai
```

Add this configuration:
```nginx
server {
    listen 80;
    server_name YOUR_DOMAIN_OR_IP;

    # Dashboard
    location / {
        proxy_pass http://localhost:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # API endpoints
    location /api/ {
        proxy_pass http://localhost:8000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

```bash
# Enable site
sudo ln -s /etc/nginx/sites-available/king-ai /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### Deployment Verification

After deployment, verify everything is working:

#### Health Checks
```bash
# Check service health
curl http://localhost:8000/api/health
curl http://localhost:8080/health
curl http://localhost:8002/health
curl http://localhost:8001/health

# Check dashboard
curl http://localhost:3000
```

#### LLM Verification
```bash
# Test Ollama
curl http://localhost:11434/api/tags

# Test chat API
curl -X POST http://localhost:8000/api/chat/message \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello King AI", "session_id": "test"}'
```

#### Full System Test
- Access dashboard: `http://YOUR_EC2_IP:3000`
- Try the "Talk to King AI" tab
- Check all dashboard sections load properly

## 🛠️ Local Development Setup

### Prerequisites
- Python 3.10+
- PostgreSQL 14+
- Redis 6+
- Node.js 20+
- Git
- Ollama (for local LLM)

### Quick Local Setup
```bash
# Clone repository
git clone <your-repo-url>
cd king-ai-v3/agentic-framework-main

# Start infrastructure
docker-compose up -d postgres redis chroma

# Install Python dependencies
pip install -r requirements.txt

# Install dashboard dependencies
cd dashboard && npm install && cd ..

# Start Ollama
ollama pull llama3.1:70b
ollama serve

# Start services using the automated script
wsl bash -c "cd /mnt/c/Users/dmilner.AGV-040318-PC/Downloads/landon/king-ai-v2/king-ai-v3/agentic-framework-main/orchestrator && sed -i 's/\r$//' run_service.sh && echo 'YOUR_IP_ADRESS_HERE' | bash run_service.sh"

**Note**: Replace `YOUR_IP_ADRESS_HERE` with your AWS EC2 public IP address (e.g., `54.167.201.176`). This script will:
- Deploy all services to your AWS server
- Set up Ollama with the llama3.1:70b model
- Configure databases and environment variables
- Start all microservices automatically

# Alternative: Manual service startup (if script doesn't work)
# python orchestrator/service/main.py &
# python mcp-gateway/service/main.py &
# python memory-service/service/main.py &
# python subagent-manager/service/main.py &

# Start dashboard
cd dashboard && npm run dev
```

### Access Local Development
- **Dashboard**: http://localhost:3000
- **Orchestrator API**: http://localhost:8000
- **MCP Gateway**: http://localhost:8080
- **Memory Service**: http://localhost:8002
- **Subagent Manager**: http://localhost:8001

## ⚙️ Configuration Guide

### Environment Variables

#### Core Configuration (.env files in each service)
```env
# LLM Configuration
OLLAMA_ENDPOINT=http://localhost:11434
LOCAL_MODEL=llama3.1:70b
DEFAULT_LLM_PROVIDER=local

# Database
POSTGRES_URL=postgresql://agent_user:agent_pass@localhost:5432/agentic_framework
REDIS_URL=redis://localhost:6379/0

# Service URLs
MCP_GATEWAY_URL=http://localhost:8080
MEMORY_SERVICE_URL=http://localhost:8002
SUBAGENT_MANAGER_URL=http://localhost:8001

# Security
JWT_SECRET_KEY=your-secret-key-here
JWT_ALGORITHM=HS256

# AWS (for production)
AWS_IP=54.167.201.176
```

#### Service-Specific Configuration

**Orchestrator (.env)**
```env
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO
MAX_WORKFLOW_TIMEOUT=3600
REQUIRE_HUMAN_APPROVAL_DEFAULT=false
```

**MCP Gateway (.env)**
```env
HOST=0.0.0.0
PORT=8080
LOG_LEVEL=INFO
```

**Memory Service (.env)**
```env
HOST=0.0.0.0
PORT=8002
CHROMA_PATH=./data/chroma
MEMORY_RETENTION_YEARS=2
```

**Subagent Manager (.env)**
```env
HOST=0.0.0.0
PORT=8001
MAX_CONCURRENT_AGENTS=5
AGENT_TIMEOUT=300
```

### Database Setup

#### Docker Compose (Recommended)
```yaml
version: '3.8'
services:
  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: agentic_framework
      POSTGRES_USER: agent_user
      POSTGRES_PASSWORD: agent_pass
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  chroma:
    image: ghcr.io/chroma-core/chroma:latest
    ports:
      - "8001:8000"
    volumes:
      - chroma_data:/chroma/chroma
    environment:
      - IS_PERSISTENT=TRUE
```

#### Initialize Database
```bash
# Run migrations
cd king-ai-v3/agentic-framework-main
alembic upgrade head

# Seed initial data
python scripts/seed_data.py
```

## 🖥️ Dashboard Operations Guide

### 1. Talk to King AI (Main Interface)
The primary way to interact with King AI v3.

**Features:**
- Natural language conversation
- Workflow creation and execution
- Real-time responses powered by Ollama
- Session persistence

**Usage Examples:**
- "Create a workflow for a SaaS business"
- "Analyze my current business portfolio"
- "Generate a marketing strategy for product X"
- "Show me the system health status"

**Status Indicators:**
- 🟢 **Online**: AI is responsive and ready
- 🟡 **Thinking**: Processing complex requests
- 🔴 **Offline**: Service unavailable

### 2. Command Center
High-level empire overview and control.

**Features:**
- Real-time business metrics
- Active workflow monitoring
- System health dashboard
- Quick action buttons

### 3. Workflow Studio
Create and manage automated workflows.

**Features:**
- Visual workflow builder
- Pre-built templates
- Execution monitoring
- Approval workflows

### 4. Approval Center
Human oversight for critical decisions.

**Features:**
- Pending approval queue
- Decision history
- Risk assessment
- Bulk actions

### 5. Business P&L Tracker
Financial monitoring and analysis.

**Features:**
- Real-time revenue tracking
- Profit/loss analysis
- Business performance metrics
- Forecasting tools

### 6. Agent Control Center
Manage specialized AI agents.

**Features:**
- Agent status monitoring
- Performance metrics
- Agent configuration
- Load balancing

### 7. Analytics Dashboard
Comprehensive business intelligence.

**Features:**
- Custom dashboards
- Data visualization
- Trend analysis
- Export capabilities

### 8. Settings & Configuration
System configuration and preferences.

**Features:**
- LLM provider settings
- Risk profile configuration
- API key management
- System preferences

## 🔧 Troubleshooting Guide

### Common Issues

#### 1. Chat Not Responding
**Symptoms:** Chat interface loads but no responses
**Solutions:**
- Check Ollama service: `curl http://localhost:11434/api/tags`
- Verify orchestrator logs: `docker logs orchestrator`
- Check network connectivity between services

#### 2. Dashboard Not Loading
**Symptoms:** Blank page or connection errors
**Solutions:**
- Check dashboard service: `curl http://localhost:3000`
- Verify proxy configuration in `vite.config.js`
- Check browser console for errors

#### 3. Database Connection Errors
**Symptoms:** Services fail to start with DB errors
**Solutions:**
- Verify PostgreSQL container: `docker ps | grep postgres`
- Check connection string in .env files
- Run migrations: `alembic upgrade head`

#### 4. LLM Model Not Available
**Symptoms:** Chat responds with errors about model
**Solutions:**
- Check Ollama models: `ollama list`
- Pull model: `ollama pull llama3.1:70b`
- Verify model name in .env files

### Service Health Checks

#### Quick Health Check Script
```bash
#!/bin/bash
echo "=== King AI v3 Health Check ==="

# Check infrastructure
echo "PostgreSQL:"; docker exec postgres pg_isready -U agent_user
echo "Redis:"; docker exec redis redis-cli ping
echo "ChromaDB:"; curl -s http://localhost:8001/api/v1/heartbeat

# Check services
echo "Orchestrator:"; curl -s http://localhost:8000/api/health
echo "MCP Gateway:"; curl -s http://localhost:8080/health
echo "Memory Service:"; curl -s http://localhost:8002/health
echo "Subagent Manager:"; curl -s http://localhost:8001/health

# Check Ollama
echo "Ollama:"; curl -s http://localhost:11434/api/tags | jq '.models | length'

echo "=== Health Check Complete ==="
```

### Log Locations

#### Docker Services
```bash
# View service logs
docker logs orchestrator
docker logs mcp-gateway
docker logs memory-service
docker logs subagent-manager

# Follow logs in real-time
docker logs -f orchestrator
```

#### Application Logs
- Orchestrator: `orchestrator/service/logs/`
- MCP Gateway: `mcp-gateway/service/logs/`
- Memory Service: `memory-service/service/logs/`
- Subagent Manager: `subagent-manager/service/logs/`

## 🔄 Updates and Maintenance

### Updating King AI v3

#### Automated Updates
```bash
# Use the control script
python scripts/control.py
# Select: [2] 🔄 Update & Restart Services
```

#### Manual Updates
```bash
# Pull latest changes
git pull origin main

# Update dependencies
pip install -r requirements.txt --upgrade
cd dashboard && npm update && cd ..

# Run migrations
alembic upgrade head

# Restart services
docker-compose restart
```

### Backup Strategy

#### Database Backup
```bash
# PostgreSQL backup
docker exec postgres pg_dump -U agent_user agentic_framework > backup_$(date +%Y%m%d).sql

# Restore
docker exec -i postgres psql -U agent_user agentic_framework < backup_file.sql
```

#### Configuration Backup
```bash
# Backup all .env files
tar -czf config_backup_$(date +%Y%m%d).tar.gz */.env
```

## 📞 Support and Resources

### Documentation
- **API Documentation**: `/api/docs` (when services are running)
- **Developer Docs**: `DEVELOPER_DOCS.md`
- **Architecture Guide**: `docs/architecture.md`

### Community
- **GitHub Issues**: Report bugs and request features
- **Discussions**: Ask questions and share experiences
- **Wiki**: Community-contributed guides and tutorials

### Professional Support
For enterprise deployments and custom integrations:
- Contact: support@king-ai-studio.me
- Enterprise SLA options available

---

**Remember**: King AI v3 is designed for autonomous operation, but human oversight ensures optimal performance. Regular monitoring and occasional human intervention will maximize your business empire's success.
```

#### Database Setup

**Option 1: Local PostgreSQL**
```bash
# Install PostgreSQL and create database
# Update .env with: DATABASE_URL=postgresql+asyncpg://king:password@localhost:5432/kingai
```

**Option 2: Docker PostgreSQL**
```bash
docker run -d \
  --name kingai-postgres \
  -e POSTGRES_USER=king \
  -e POSTGRES_PASSWORD=your-secure-password \
  -e POSTGRES_DB=kingai \
  -p 5432:5432 \
  postgres:15
```

**Redis Setup**
```bash
docker run -d --name kingai-redis -p 6379:6379 redis:7
```

#### Initialize Database
```bash
# Run database migrations
alembic upgrade head
```

#### LLM Provider Setup

Configure at least one LLM provider:

**Ollama (Local, Free)**
```bash
# Install and start Ollama
ollama pull llama3.1:8b
ollama serve
```

**Claude (Cloud, High-Quality)**
```env
ANTHROPIC_API_KEY=sk-ant-api03-xxxxx
CLAUDE_MODEL=claude-3-5-sonnet-20241022
```

**Gemini (Cloud, Fallback)**
```env
GEMINI_API_KEY=AIzaSy...
```

#### Start Services

```bash
# Terminal 1: Start the API server
python -m uvicorn src.api.main:app --reload --port 8000

# Terminal 2: Start the dashboard
cd dashboard
npm run dev
```

### Access the System

#### Production (Live System)
- **Dashboard**: https://king-ai-studio.me
- **API**: https://king-ai-studio.me/api/
- **API Docs**: https://king-ai-studio.me/api/docs
- **Health Check**: https://king-ai-studio.me/api/docs

#### Local Development
- **Dashboard**: http://localhost:5173
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **CLI**: `python cli.py`

## ⚙️ Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure:

#### Required
```env
# Database
DATABASE_URL=postgresql+asyncpg://king:password@localhost:5432/kingai
REDIS_URL=redis://localhost:6379

# At least one LLM provider
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
# OR
ANTHROPIC_API_KEY=sk-ant-api03-xxxxx
```

#### Optional Integrations
```env
# E-commerce
SHOPIFY_SHOP_URL=your-store.myshopify.com
SHOPIFY_ACCESS_TOKEN=your_token

# Payments
STRIPE_API_KEY=sk_live_...

# Analytics
GOOGLE_ANALYTICS_ID=GA_...

# Monitoring
DATADOG_API_KEY=...
```

## 🖥️ Dashboard Operations

### 1. The CEO Chat (Command Center)
This is your direct line to the Master AI.
*   **Ask**: "How is the empire doing?"
*   **Command**: "Research dropshipping trends for 2025."
*   **Analyze**: "Why is PetPal losing money?"

## 🖥️ Dashboard Operations

### 1. The CEO Chat (Command Center)
This is your direct line to the Master AI.
*   **Ask**: "How is the empire doing?"
*   **Command**: "Research dropshipping trends for 2025."
*   **Analyze**: "Why is PetPal losing money?"

**Status Indicators:**
*   🟢 **Online**: Brain is fast aynd responsive.
*   🟡 **Thinking**: Complex task in progress (Simulation/Research).
*   🔴 **Offline**: Connection lost (Check Server).

### 2. Empire Overview
A high-level view of your portfolio.
*   **Total Revenue**: Aggregated real-time income.
*   **Active Businesses**: List of all running ventures (e.g., PetPal, CodeDoc).
*   **Health**: Green/Red status for each unit based on profit margins.

---

## 🛑 Shutdown & Pause

since this runs on a Cloud Server, it runs 24/7 unless you stop it.

### To Pause the AI (Stop spending money on Autonomy)
Chat Command: "Stop autonomous mode."
*   *Effect*: The AI stops its 6-hour optimization loops but keeps the dashboard online.

### To Stop the Server (Stop paying AWS)
1.  Log into your **AWS Console**.
2.  Select the instance (`i-xxxx`).
3.  Click **Instance State** -> **Stop**.
