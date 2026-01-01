# King AI v2 - Complete User Guide

Discipline. Autonomy. Success.

This comprehensive guide covers everything you need to know about King AI v2 - from initial setup to operating your autonomous business empire.

## 🟢 Getting Started

### Automated Setup (Recommended)

The easiest way to get started is using the automated setup:

1. **Open PowerShell or Terminal**
2. **Navigate to the project folder**
3. **Run the automated setup:**
   ```bash
   python scripts/control.py
   ```
4. **Select option [3] 🤖 Automated Empire Setup (AWS Infra + GitHub + Full Setup)**

This will automatically:
- Check and install prerequisites (Python, Git, AWS CLI, Terraform)
- Configure AWS credentials and infrastructure (if needed)
- Set up databases, Redis, and all services
- Deploy the application and start all services

### Manual Setup

If you prefer manual control:

#### Prerequisites
- Python 3.10+
- PostgreSQL 14+
- Redis 6+
- Node.js 18+
- Git

#### Installation Steps

```bash
# Clone the repository
git clone https://github.com/your-org/king-ai-v2.git
cd king-ai-v2

# Create Python virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
.\venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install Python dependencies
pip install -e .

# Install dashboard dependencies
cd dashboard
npm install
cd ..
```

#### Configuration

```bash
# Copy the example configuration
cp .env.example .env

# Edit .env with your settings (see Configuration section below)
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
