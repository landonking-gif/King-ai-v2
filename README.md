# King AI v2 - Autonomous Business Empire

![King AI V2](https://img.shields.io/badge/Status-Operational-green) ![AI-Brain](https://img.shields.io/badge/AI_Model-Multi_LLM-blue) ![License](https://img.shields.io/badge/License-Private-red)

King AI v2 is a sophisticated autonomous AI system that acts as an "AI CEO" to plan, launch, and manage digital businesses. It features a multi-brain architecture with specialized agents, self-evolution capabilities, and human oversight through an approval system.

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- PostgreSQL 14+
- Redis 6+
- Node.js 18+
- Docker (optional)

### Automated Setup (Recommended)
```bash
git clone <your-repo-url>
cd king-ai-v2

# Run the automated setup
python scripts/control.py
# Select [3] 🤖 Automated Empire Setup (AWS Infra + GitHub + Full Setup)
```

This will automatically:
- Install prerequisites
- Configure AWS infrastructure (if needed)
- Set up databases and services
- Deploy the application

### Manual Setup
```bash
git clone <your-repo-url>
cd king-ai-v2

# Copy configuration
cp .env.example .env
# Edit .env with your API keys

# Install dependencies
pip install -r requirements.txt
npm install

# Start services
docker-compose up -d
python -m uvicorn src.api.main:app --reload
```

### Access Dashboard
Open http://localhost:5173 in your browser.

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
