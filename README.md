# King AI v2 - Autonomous Business Empire

![King AI V2](https://img.shields.io/badge/Status-Operational-green) ![AI-Brain](https://img.shields.io/badge/AI_Model-Multi_LLM-blue) ![License](https://img.shields.io/badge/License-Private-red)

King AI v2 is a sophisticated autonomous AI system that acts as an "AI CEO" to plan, launch, and manage digital businesses. It features a multi-brain architecture with specialized agents, self-evolution capabilities, and human oversight through an approval system.

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
