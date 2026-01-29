# King AI v3 - AWS Deployment Guide

This guide covers deploying King AI v3 (Agentic Framework) to a fresh AWS EC2 instance.

## Architecture Overview

King AI v3 is a standalone multi-agent orchestration platform with these components:

| Service | Port | Description |
|---------|------|-------------|
| Orchestrator | 8000 | Main API - workflow orchestration, agent management |
| Subagent Manager | 8001 | Manages subagent lifecycle and task distribution |
| Memory Service | 8002 | Multi-tier storage with /diary and /reflect endpoints |
| MCP Gateway | 8080 | External tool access via Model Context Protocol |
| Control Panel | 3000 | Web dashboard for monitoring and control |
| Ollama | 11434 | Local LLM inference (llama3.2) |

## Prerequisites

### AWS EC2 Instance
- Ubuntu 22.04 LTS
- t3.large or larger (4+ GB RAM recommended)
- 30GB+ storage
- Security group with ports: 22, 3000, 8000-8002, 8080, 11434

### Local Machine
- SSH key for EC2 access
- PowerShell (Windows) or Bash (Linux/Mac)

## Quick Deployment

### Windows (PowerShell)
```powershell
cd king-ai-v3\agentic-framework-main
.\deploy-aws.ps1 -AwsIp "YOUR_EC2_IP"
```

### Linux/Mac (Bash)
```bash
cd king-ai-v3/agentic-framework-main
chmod +x deploy-aws.sh
./deploy-aws.sh YOUR_EC2_IP
```

## Manual Deployment Steps

If the automated script fails, follow these steps:

### 1. Install System Dependencies
```bash
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3-pip
sudo apt install -y git curl wget nginx redis-server postgresql postgresql-contrib
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs
```

### 2. Install Ollama
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2:1b
```

### 3. Configure PostgreSQL
```bash
sudo -u postgres psql -c "CREATE USER kingai WITH PASSWORD 'kingai123' CREATEDB;"
sudo -u postgres psql -c "CREATE DATABASE agentic_framework OWNER kingai;"
```

### 4. Upload Framework
```bash
scp -r agentic-framework-main ubuntu@YOUR_EC2_IP:~/
```

### 5. Setup Python Environment
```bash
cd ~/agentic-framework-main
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 6. Start Services
```bash
./start_all_services.sh
```

## Configuration

### Environment Variables (.env)

The `.env` file contains all configuration. Key settings:

```env
# LLM Provider
OLLAMA_URL=http://localhost:11434
LOCAL_MODEL=llama3.2:1b

# Database
POSTGRES_URL=postgresql://kingai:kingai123@localhost:5432/agentic_framework
REDIS_URL=redis://localhost:6379/0

# API Keys (optional, for cloud LLMs)
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GEMINI_API_KEY=
```

### Transferring Credentials from King AI v2

If you have credentials in King AI v2's `.env`:
```bash
# Copy relevant keys from v2
cat ../king-ai-v2/.env | grep -E "(API_KEY|SECRET)" >> .env
```

## API Endpoints

### Orchestrator (Port 8000)
- `GET /health` - Health check
- `POST /api/chat` - Chat interface
- `GET /api/agents` - List available agents
- `POST /api/workflows/execute` - Execute workflow

### Memory Service (Port 8002)
- `GET /health` - Health check
- `POST /diary` - Create session diary entry
- `GET /diary` - List diary entries
- `POST /reflect` - Trigger pattern analysis
- `POST /memory/commit` - Store artifact
- `POST /memory/query` - Query artifacts

### Control Panel (Port 3000)
- Web dashboard at `http://YOUR_IP:3000`
- Login: admin / admin123 (change in production)

## Using Ralph Code Agent

Ralph is the autonomous coding agent. Trigger it via:

```bash
curl -X POST http://YOUR_IP:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "use ralph to create a hello world script"}'
```

## Troubleshooting

### Check Service Status
```bash
./start_all_services.sh
# Or manually:
curl http://localhost:8000/health
curl http://localhost:8002/health
```

### View Logs
```bash
tail -f /tmp/orchestrator.log
tail -f /tmp/memory-service.log
tail -f /tmp/control-panel.log
```

### Restart Services
```bash
pkill -f uvicorn
./start_all_services.sh
```

### Common Issues

1. **Port already in use**: Kill existing processes
   ```bash
   pkill -f "uvicorn.*8000"
   ```

2. **Database connection failed**: Check PostgreSQL
   ```bash
   sudo systemctl status postgresql
   sudo -u postgres psql -c "\l"
   ```

3. **Ollama not responding**: Restart Ollama
   ```bash
   sudo systemctl restart ollama
   ollama list
   ```

## Security Considerations

For production deployment:
1. Change default passwords in `.env`
2. Configure HTTPS via nginx
3. Set up firewall rules
4. Enable authentication on all endpoints
5. Use environment-specific secrets

## Upgrading

To update to a new version:
```bash
cd ~/agentic-framework-main
git pull  # or upload new files
source .venv/bin/activate
pip install -r requirements.txt
./start_all_services.sh
```
