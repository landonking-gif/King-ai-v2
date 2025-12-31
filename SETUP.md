# King AI v2 - Complete Setup Guide

This guide walks you through setting up King AI v2, the autonomous business empire management system.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Configuration File Setup](#configuration-file-setup)
4. [Database Setup](#database-setup)
5. [LLM Provider Setup](#llm-provider-setup)
6. [Integration Setup](#integration-setup)
   - [Shopify](#shopify-e-commerce)
   - [Stripe](#stripe-payments)
   - [Plaid](#plaid-banking)
   - [Google Analytics](#google-analytics)
   - [OpenAI (DALL-E)](#openai-dall-e-images)
   - [SerpAPI](#serpapi-web-search)
7. [Monitoring Setup (Optional)](#monitoring-setup-optional)
8. [Running the Application](#running-the-application)
9. [Deployment](#deployment)
10. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Software

| Software | Version | Purpose |
|----------|---------|---------|
| Python | 3.10+ | Backend runtime |
| PostgreSQL | 14+ | Primary database |
| Redis | 6+ | Caching & task queue |
| Node.js | 18+ | Dashboard frontend |
| Docker | 20+ | Containerization (optional) |

### Recommended Hardware

- **Development**: 8GB RAM, 4 CPU cores
- **Production**: 16GB+ RAM, 8+ CPU cores, GPU for local LLM inference

---

## Quick Start

### 1. Clone and Install

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

### 2. Configure Environment

```bash
# Copy the example configuration
cp .env.example .env

# Edit .env with your settings
# See "Configuration File Setup" section below
```

### 3. Initialize Database

```bash
# Run database migrations
alembic upgrade head
```

### 4. Start Services

```bash
# Terminal 1: Start the API server
python -m uvicorn src.api.main:app --reload --port 8000

# Terminal 2: Start the dashboard
cd dashboard
npm run dev
```

### 5. Access the System

- **Dashboard**: http://localhost:5173
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **CLI**: `python cli.py`

---

## Configuration File Setup

All configuration is done through the `.env` file. Copy `.env.example` to `.env` and fill in your values.

### Configuration Reference

See the `.env.example` file for a complete template with all available options and descriptions.

### Minimum Required Configuration

At minimum, you need:

```env
# Database (required)
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/kingai

# LLM Provider (at least one required)
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
```

---

## Database Setup

### Option 1: Local PostgreSQL

```bash
# Install PostgreSQL (Ubuntu/Debian)
sudo apt install postgresql postgresql-contrib

# Create database and user
sudo -u postgres psql
CREATE DATABASE kingai;
CREATE USER king WITH PASSWORD 'your-secure-password';
GRANT ALL PRIVILEGES ON DATABASE kingai TO king;
\q

# Update .env
DATABASE_URL=postgresql+asyncpg://king:your-secure-password@localhost:5432/kingai
```

### Option 2: Docker PostgreSQL

```bash
docker run -d \
  --name kingai-postgres \
  -e POSTGRES_USER=king \
  -e POSTGRES_PASSWORD=your-secure-password \
  -e POSTGRES_DB=kingai \
  -p 5432:5432 \
  postgres:15
```

### Redis Setup

```bash
# Docker Redis
docker run -d --name kingai-redis -p 6379:6379 redis:7

# Update .env
REDIS_URL=redis://localhost:6379
```

### Run Migrations

```bash
# Initialize/update database schema
alembic upgrade head
```

---

## LLM Provider Setup

King AI supports multiple LLM providers with automatic fallback. Configure at least one.

### Option 1: Ollama (Recommended for Development)

Free, runs locally on your machine.

```bash
# Install Ollama
# Windows: Download from https://ollama.ai
# Linux:
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model
ollama pull llama3.1:8b

# Start Ollama server (runs on port 11434 by default)
ollama serve
```

**.env configuration:**
```env
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
```

### Option 2: vLLM (Recommended for Production)

High-performance inference server for production workloads.

```bash
# Install vLLM
pip install vllm

# Start vLLM server
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --port 8080
```

**.env configuration:**
```env
VLLM_URL=http://localhost:8080
VLLM_MODEL=meta-llama/Llama-3.1-70B-Instruct
```

### Option 3: Claude (Anthropic)

Cloud-based, excellent for high-stakes decisions.

1. Get API key from: https://console.anthropic.com/
2. Add to `.env`:

```env
ANTHROPIC_API_KEY=sk-ant-api03-xxxxx
CLAUDE_MODEL=claude-3-5-sonnet-20241022
```

### Option 4: Gemini (Google)

Cloud-based fallback option.

1. Get API key from: https://makersuite.google.com/app/apikey
2. Add to `.env`:

```env
GEMINI_API_KEY=AIzaSyxxxxx
```

---

## Integration Setup

### Shopify (E-Commerce)

Required for: Commerce Agent, dropshipping playbooks

#### Step 1: Create a Shopify App

1. Go to your Shopify Admin → Settings → Apps and sales channels
2. Click "Develop apps" → "Create an app"
3. Name it "King AI Integration"

#### Step 2: Configure API Scopes

In your app settings, enable these scopes:
- `read_products`, `write_products`
- `read_orders`, `write_orders`
- `read_inventory`, `write_inventory`
- `read_fulfillments`, `write_fulfillments`

#### Step 3: Get Credentials

1. Install the app on your store
2. Copy the Admin API access token

**.env configuration:**
```env
SHOPIFY_SHOP_URL=your-store.myshopify.com
SHOPIFY_ACCESS_TOKEN=shpat_xxxxx
SHOPIFY_API_VERSION=2024-10
```

---

### Stripe (Payments)

Required for: Finance Agent, payment processing

#### Step 1: Create Stripe Account

1. Sign up at https://stripe.com
2. Complete business verification

#### Step 2: Get API Keys

1. Go to Developers → API keys
2. Copy both the secret key and publishable key

#### Step 3: Set Up Webhooks (Optional but recommended)

1. Go to Developers → Webhooks
2. Add endpoint: `https://your-domain.com/api/webhooks/stripe`
3. Select events: `payment_intent.succeeded`, `customer.subscription.created`, etc.
4. Copy the webhook signing secret

**.env configuration:**
```env
STRIPE_API_KEY=sk_live_xxxxx
STRIPE_PUBLISHABLE_KEY=pk_live_xxxxx
STRIPE_WEBHOOK_SECRET=whsec_xxxxx
```

> ⚠️ Use `sk_test_` keys for development/testing!

---

### Plaid (Banking)

Required for: Banking Agent, financial account connections

#### Step 1: Create Plaid Account

1. Sign up at https://dashboard.plaid.com/signup
2. Apply for production access (if needed)

#### Step 2: Get Credentials

1. Go to Keys in the dashboard
2. Copy client_id and secret for your environment

**.env configuration:**
```env
PLAID_CLIENT_ID=your_client_id
PLAID_SECRET=your_secret
PLAID_ENV=sandbox  # sandbox, development, or production
```

---

### Google Analytics (Web Analytics)

Required for: Analytics Agent, traffic monitoring

#### Step 1: Create Service Account

1. Go to Google Cloud Console → IAM & Admin → Service Accounts
2. Create a new service account
3. Download the JSON key file

#### Step 2: Grant GA4 Access

1. In Google Analytics, go to Admin → Property Access Management
2. Add the service account email with "Viewer" role

#### Step 3: Get Property ID

1. In GA4, go to Admin → Property Settings
2. Copy the Property ID (format: `123456789`)

**.env configuration:**
```env
GA4_PROPERTY_ID=123456789
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
# OR use inline JSON:
# GA4_CREDENTIALS_JSON='{"type": "service_account", ...}'
```

---

### OpenAI / DALL-E (Images)

Required for: Content Agent image generation

1. Get API key from: https://platform.openai.com/api-keys
2. Ensure you have DALL-E access enabled

**.env configuration:**
```env
OPENAI_API_KEY=sk-xxxxx
```

---

### SerpAPI (Web Search)

Required for: Research Agent web searches

1. Sign up at https://serpapi.com
2. Get API key from dashboard

**.env configuration:**
```env
SERPAPI_KEY=xxxxx
```

---

### Pinecone (Vector Database)

Required for: Long-term memory, RAG

1. Sign up at https://www.pinecone.io
2. Create an index with dimension 1536 (OpenAI embeddings)

**.env configuration:**
```env
PINECONE_API_KEY=xxxxx
PINECONE_INDEX=king-ai
PINECONE_ENVIRONMENT=us-east-1-aws
```

---

## Monitoring Setup (Optional)

### Datadog

```env
DD_API_KEY=xxxxx
DD_APP_KEY=xxxxx
```

### Arize (ML Observability)

```env
ARIZE_API_KEY=xxxxx
ARIZE_SPACE_KEY=xxxxx
```

### LangSmith (LLM Tracing)

```env
LANGCHAIN_API_KEY=xxxxx
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=king-ai-v2
```

---

## Running the Application

### Development Mode

```bash
# Start API with hot-reload
python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Start dashboard with hot-reload
cd dashboard && npm run dev
```

### Production Mode

```bash
# API with multiple workers
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4

# Dashboard build
cd dashboard && npm run build
# Serve with nginx or similar
```

### Docker Compose

```bash
docker-compose up -d
```

### CLI Interface

```bash
python cli.py
```

---

## Deployment

### AWS EC2 Deployment

Use the included deployment script:

```bash
python scripts/control.py
```

This will:
1. Package the codebase
2. Upload to your EC2 instance
3. Install dependencies
4. Run migrations
5. Start services

### Manual Deployment

1. Set up PostgreSQL and Redis on your server
2. Clone the repository
3. Configure `.env`
4. Run `alembic upgrade head`
5. Start with systemd or supervisor

---

## Troubleshooting

### Database Connection Failed

```
Error: connection refused to localhost:5432
```

**Solution**: Ensure PostgreSQL is running:
```bash
sudo systemctl start postgresql
# or
docker start kingai-postgres
```

### Ollama Not Responding

```
Error: Connection refused to localhost:11434
```

**Solution**: Start Ollama:
```bash
ollama serve
```

### Missing API Keys

```
Warning: Shopify not configured, using mock mode
```

**Solution**: Add the required API keys to your `.env` file.

### Migration Errors

```
Error: relation "business_units" does not exist
```

**Solution**: Run migrations:
```bash
alembic upgrade head
```

### Port Already in Use

```
Error: Address already in use (8000)
```

**Solution**: Kill the existing process:
```bash
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux/Mac
lsof -i :8000
kill -9 <PID>
```

---

## Next Steps

1. ✅ Complete the configuration in `.env`
2. 🚀 Start the application
3. 📊 Access the dashboard at http://localhost:5173
4. 💬 Chat with King AI to create your first business
5. 📚 Read the [User Guide](USER_GUIDE.md) for usage instructions

---

## Support

- **Issues**: Open a GitHub issue
- **Documentation**: See `/docs` folder
- **API Reference**: http://localhost:8000/docs (when running)
