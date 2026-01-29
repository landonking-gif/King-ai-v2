# King AI v3 - Autonomous Business Empire

![King AI V3](https://img.shields.io/badge/Status-Operational-green) ![AI-Brain](https://img.shields.io/badge/AI_Model-DeepSeek_R1_7B-blue) ![MoltBot](https://img.shields.io/badge/Multi--Channel-MoltBot_Integrated-purple) ![License](https://img.shields.io/badge/License-Private-red)

King AI v3 is a sophisticated autonomous AI system that acts as an "AI CEO" to plan, launch, and manage digital businesses. It features a microservices architecture with specialized agents, self-evolution capabilities, and human oversight through an approval system. **Now integrated with MoltBot for multi-channel AI access via WhatsApp, Telegram, Discord, Slack, and more.**

## 🚀 Production Deployment

### Current Live Deployment
- **Domain**: https://king-ai-studio.me
- **Status**: ✅ Operational (AWS EC2 + Load Balancer)
- **Infrastructure**: AWS EC2 t3.medium, Docker containers, Nginx reverse proxy
- **SSL**: ✅ Enabled (AWS Load Balancer termination)
- **Architecture**: Microservices (Orchestrator, MCP Gateway, Memory Service, Subagent Manager)

### Access Points
- **Dashboard**: http://100.24.50.240:8000 (King AI)
- **MoltBot Control UI**: http://100.24.50.240:18789
- **API Docs**: http://100.24.50.240:8000/docs
- **API Base**: http://100.24.50.240:8000/api/
- **OpenAI-Compatible API**: http://100.24.50.240:8000/v1/chat/completions
- **Health Check**: http://100.24.50.240:8000/api/health

### System Architecture
- **Orchestrator** (port 8000): Main AI brain, workflow management, chat interface, OpenAI-compatible API
- **MCP Gateway** (port 8080): Model Context Protocol for tool integration
- **Memory Service** (port 8002): Long-term memory and vector storage
- **Subagent Manager** (port 8001): Manages specialized AI agents
- **Dashboard** (port 3000): React frontend with real-time updates
- **Ollama** (port 11434): Local LLM runtime with DeepSeek R1 7B (4.7GB)
- **MoltBot Gateway** (port 18789): Multi-channel AI interface (WhatsApp, Telegram, Discord, Slack, Signal, etc.)

## 🛠️ Quick Start

### Automated AWS Deployment (Recommended)
```bash
git clone <your-repo-url>
cd king-ai-v3/agentic-framework-main

# Run automated deployment
python scripts/control.py
# Select [3] 🤖 Automated Empire Setup (AWS Infra + GitHub + Full Setup)
# Enter your AWS EC2 IP and .pem key path
```

This automatically handles:
- AWS server setup and dependency installation
- Database configuration (PostgreSQL + Redis)
- Ollama LLM setup with llama3.1:70b model
- All microservices deployment and configuration
- Nginx reverse proxy and SSL setup

### Local Development
```bash
git clone <your-repo-url>
cd king-ai-v3/agentic-framework-main

# Start infrastructure
docker-compose up -d postgres redis chroma

# Install dependencies
pip install -r requirements.txt
cd dashboard && npm install && cd ..

# Start Ollama and pull DeepSeek R1 7B
ollama pull deepseek-r1:7b
ollama serve

# Start all services
bash /home/ubuntu/king-ai-v3/start_all_services.sh

# Or start services individually:
python orchestrator/service/main.py &
python mcp-gateway/service/main.py &
python memory-service/service/main.py &
python subagent-manager/service/main.py &
cd ../moltbot && pnpm moltbot gateway --port 18789 &

# Start dashboard
cd dashboard && npm run dev
```

### Access Local Development
- **Dashboard**: http://localhost:3000
- **Orchestrator API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## 🧠 Core Features

### Multi-Brain AI Architecture
- **Master AI Brain (Orchestrator)**: Central orchestrator that plans, delegates, and learns
- **Specialized Agents**: Research, Commerce, Finance, Content, Legal, Analytics, Banking, Code Generation
- **LLM Integration**: Ollama with llama3.1:70b model, extensible to other providers
- **Microservices Design**: Independent, scalable services for reliability

### Business Automation
- **Playbook Executor**: Runs business playbooks (dropshipping, SaaS, etc.)
- **Lifecycle Engine**: Manages businesses through ideation → scaling stages
- **Portfolio Manager**: Tracks and optimizes multiple business units
- **Real-time Monitoring**: Live dashboards and health checks

### Self-Evolution & Learning
- **Evolution Engine**: AI proposes improvements to its own code
- **Sandboxed Testing**: All changes validated before applying
- **Confidence Thresholds**: Only high-confidence improvements are proposed
- **Memory System**: Long-term learning and context retention

### Human Oversight
- **Approval System**: Critical actions require human approval
- **Risk Profiles**: Configure autonomy level (conservative/moderate/aggressive)
- **Audit Logging**: Full history of all AI actions and decisions
- **Real-time Dashboard**: Live notifications and status updates

### Advanced Capabilities
- **Chat Interface**: Natural language interaction with the AI CEO
- **Workflow Studio**: Visual workflow creation and management
- **Analytics Dashboard**: Business intelligence and reporting
- **Agent Control Center**: Monitor and manage specialized agents
- **Memory Persistence**: ChromaDB vector storage for context retention

## 📚 Documentation

| Document | Description |
|----------|-------------|
| **[USER_GUIDE.md](USER_GUIDE.md)** | Complete user guide - setup, operation, AWS deployment |
| **[AWS_DEPLOYMENT_CHECKLIST.md](AWS_DEPLOYMENT_CHECKLIST.md)** | Step-by-step AWS deployment checklist |
| **[deploy.ps1](deploy.ps1)** | Complete automated AWS deployment script |
| **[DEVELOPER_DOCS.md](DEVELOPER_DOCS.md)** | Technical documentation - architecture, API, development |
| **[scripts/control.py](scripts/control.py)** | Automated deployment and management script |
| **[docker-compose.yml](docker-compose.yml)** | Infrastructure configuration |

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐
│   Dashboard     │    │  Orchestrator   │
│   (React)       │◄──►│  (FastAPI)      │
│   Port: 3000    │    │  Port: 8000     │
└─────────────────┘    └─────────────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐    ┌─────────────────┐
│  MCP Gateway    │    │ Memory Service  │
│   Port: 8080    │    │  Port: 8002     │
└─────────────────┘    └─────────────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐    ┌─────────────────┐
│ Subagent Manager│    │   Databases     │
│   Port: 8001    │    │ PostgreSQL+Redis│
└─────────────────┘    └─────────────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐    ┌─────────────────┐
│     Ollama      │    │    ChromaDB     │
│ Port: 11434     │    │ Vector Storage  │
└─────────────────┘    └─────────────────┘
```

### Service Descriptions

- **Orchestrator**: Main AI brain handling workflows, chat, and coordination
- **MCP Gateway**: Model Context Protocol for tool and API integrations
- **Memory Service**: Long-term memory management and vector operations
- **Subagent Manager**: Manages specialized AI agents and task delegation
- **Dashboard**: React frontend with real-time WebSocket updates

## 🔧 Technology Stack

- **Backend**: Python 3.10+, FastAPI, AsyncIO
- **Frontend**: React 18, Vite, Tailwind CSS
- **Database**: PostgreSQL 16, Redis 7
- **Vector DB**: ChromaDB for embeddings
- **LLM**: Ollama with llama3.1:70b model
- **Infrastructure**: Docker, Docker Compose, Nginx
- **Monitoring**: Health checks, logging, Prometheus-ready

## 🚀 Deployment Options

### 1. Automated AWS Deployment
```bash
python scripts/control.py
# Select automated AWS deployment option
```

### 2. Manual Docker Deployment
```bash
# Start all services
docker-compose up -d

# Check health
curl http://localhost:8000/api/health
```

### 3. Local Development
```bash
# Start individual services
python orchestrator/service/main.py &
python mcp-gateway/service/main.py &
# ... etc
```

## 📊 Monitoring & Health

### Health Endpoints
- **Orchestrator**: `GET /api/health`
- **MCP Gateway**: `GET /health`
- **Memory Service**: `GET /health`
- **Subagent Manager**: `GET /health`
- **Overall System**: `GET /api/health` (via orchestrator)

### Logs
```bash
# View service logs
docker logs orchestrator
docker logs -f mcp-gateway

# Application logs in service directories
ls */service/logs/
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `python -m pytest`
5. Submit a pull request

## 📄 License

Private - All rights reserved.

## 📞 Support

- **Documentation**: See [USER_GUIDE.md](USER_GUIDE.md)
- **Issues**: GitHub Issues
- **Email**: support@king-ai-studio.me

---

**King AI v3**: Where artificial intelligence meets entrepreneurial ambition. Build, manage, and scale your business empire with AI precision.

## 🚀 Production Deployment

### Current Live Deployment
- **Domain**: https://king-ai-studio.me
- **Status**: ✅ Operational (AWS EC2 + Load Balancer)
- **SSL**: ✅ Enabled (AWS Load Balancer termination)
- **Health**: ✅ Target group healthy

### Access Points
- **Dashboard**: https://king-ai-studio.me
- **API Docs**: https://king-ai-studio.me/api/docs
- **API Base**: https://king-ai-studio.me/api/

### Infrastructure
- **Load Balancer**: AWS ALB (handles SSL termination)
- **Web Server**: Nginx (reverse proxy on port 80)
- **API Server**: FastAPI (port 8000)
- **Dashboard**: React/Vite (port 5173)
- **Database**: PostgreSQL + Redis
- **Monitoring**: Prometheus (dynamic port)

## 🛠️ Development Setup

### Prerequisites
- Python 3.10+
- PostgreSQL 14+
- Redis 6+
- Node.js 20+
- Git
- AWS CLI (for deployment)

### Automated Setup (Recommended)
```bash
git clone https://github.com/landonking-gif/King-ai-v2.git
cd king-ai-v2

# Run the automated deployment
python scripts/control.py
# Select [3] 🤖 Automated Empire Setup (AWS Infra + GitHub + Full Setup)
```

This will automatically:
- Install system prerequisites
- Configure AWS infrastructure
- Set up databases and services
- Deploy to production server
- Configure SSL and monitoring

### Local Development
```bash
git clone https://github.com/landonking-gif/King-ai-v2.git
cd king-ai-v2

# Copy configuration
cp .env.example .env
# Edit .env with your API keys

# Install Python dependencies
pip install -e .

# Install Node.js dependencies
cd dashboard && npm install

# Start services (from project root)
python scripts/control.py
# Select [1] 🚀 Full Deployment (Code + Secrets + Restart)
```

### Access Local Development
- **Dashboard**: http://localhost:5173
- **API Docs**: http://localhost:8000/docs
- **API Base**: http://localhost:8000/api/

## 📚 Documentation

| Document | Description |
|----------|-------------|
| **[USER_GUIDE.md](USER_GUIDE.md)** | Complete user guide - setup, operation, and usage |
| **[DEVELOPER_DOCS.md](DEVELOPER_DOCS.md)** | Technical documentation - architecture, API, deployment |
| **[.env.example](.env.example)** | Configuration template with all variables explained |

## 🧠 Core Features

### Multi-Brain AI Architecture
- **Master AI Brain**: Central orchestrator that plans, delegates, and learns
- **Specialized Agents**: Research, Commerce, Finance, Content, Legal, Analytics, Banking, Code Generation
- **LLM Flexibility**: Supports Ollama (local), vLLM (production), Claude (high-stakes), Gemini (fallback)

### Business Automation
- **Playbook Executor**: Runs business playbooks (dropshipping, SaaS, etc.)
- **Lifecycle Engine**: Manages businesses through 7 stages (ideation → scaling)
- **Portfolio Manager**: Tracks and optimizes multiple business units

### Self-Evolution
- **Evolution Engine**: AI proposes improvements to its own code
- **Sandboxed Testing**: All changes validated before applying
- **Confidence Thresholds**: Only high-confidence improvements are proposed

### Human Oversight
- **Approval System**: Critical actions require human approval
- **Risk Profiles**: Configure autonomy level (conservative/moderate/aggressive)
- **Audit Logging**: Full history of all AI actions and decisions

### Real-Time Dashboard
- **CEO Chat**: Natural language interface to command the AI
- **Empire Overview**: Monitor all businesses and KPIs
- **WebSocket Updates**: Live notifications and status changes
