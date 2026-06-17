# King AI v2

Autonomous AI business agent that manages end-to-end business operations — from market research and supplier sourcing to sales processing and revenue optimization.

> **⚠ Security audit completed (2026-06-15). All 8 critical/high findings remediated. See `AUDIT_REMEDIATION_REPORT.md`.**

---

## What It Does

King AI v2 is a multi-agent autonomous business system with:

- **15 specialized agents** (orchestrator, content, research, finance, legal, commerce, banking, multi-modal, etc.)
- **Master AI brain** with self-evolution, plan executor, reflective loop, and rollback service
- **3-tier memory system** (recent events → summaries → long-term compressed storage)
- **12 integration clients** (Stripe, PayPal, Plaid, Shopify, DALL-E, Twilio, Whisper, Clip, etc.)
- **Risk-based approval manager** for autonomous spend decisions ($100 default auto-approve)
- **23-feature API** (chat, finance, banking, evolution, portfolio, lifecycle, monitoring)
- **Self-improvement loop** (5-min cycle) and **auto-goal planner** (30-min cycle)
- **Real-time WebSocket** dashboard with React UI

---

## Architecture

```
API (FastAPI + JWT auth)
    ↓
Agents (Orchestrator, Router, Skeptic)
    ↓
Master AI (Brain, Evolution, Planner, Rollback)
    ↓
Services (Autonomous Engine, Model Selector, Scheduler, LLM Cache)
    ↓
Integrations (Stripe, Plaid, Shopify, Anthropic, OpenAI, Ollama)
    ↓
PostgreSQL + Redis + Pinecone (vector store)
```

See `AUDIT_REMEDIATION_REPORT.md` for security architecture details.

---

## Build & Setup

### Prerequisites

- Python 3.11+ (3.14 tested)
- PostgreSQL 14+
- Redis 7+
- Node 20+ (for dashboard)
- Docker (recommended for production; required by `docker_sandbox_strict=true`)

### Installation

```bash
# Clone
git clone https://github.com/your-org/king-ai-v2.git
cd king-ai-v2

# Install Python dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt   # for tests/lint

# Configure environment
cp .env.example .env
# Edit .env with your credentials (database, LLM providers, integrations)

# Initialize database
python -m alembic upgrade head
```

### Run

```bash
# Development
python -m uvicorn src.api.main:app --reload --port 8000

# Production (Docker)
docker-compose up -d

# Production (manual, non-root)
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Verification

```bash
# Health check
curl http://localhost:8000/api/health

# Security scan
bandit -r src/ -f json -o reports/bandit.json
pip-audit -r requirements.txt
safety check -r requirements.txt

# Tests (not yet comprehensive — see "Known Limitations")
pytest --cov=src
```

---

## Configuration

All configuration is environment-driven via `.env`. See `.env.example` for the full list.

### Critical Settings

| Variable | Purpose | Default |
|----------|---------|---------|
| `DATABASE_URL` | asyncpg connection string | required |
| `REDIS_URL` | Redis connection | required |
| `JWT_SECRET` | JWT signing secret | **must override in production** |
| `ALLOWED_ORIGINS` | CORS origin allowlist (comma-separated) | localhost only |
| `RISK_PROFILE` | conservative / moderate / aggressive | moderate |
| `DOCKER_SANDBOX_STRICT` | refuse uncontained code execution | true |
| `STRIPE_WEBHOOK_SECRET` | Stripe signature verification | none |
| `SHOPIFY_WEBHOOK_SECRET` | Shopify HMAC verification | none |
| `PLAID_WEBHOOK_SECRET` | Plaid signature verification | none |

---

## Security

**Read `SECURITY.md` for the security policy and vulnerability reporting.**

Key security posture (post-audit 2026-06-15):

- ✅ CORS origins are explicit env-driven (no wildcards + credentials)
- ✅ All API routes require JWT Bearer or X-API-Key authentication
- ✅ All webhook endpoints enforce HMAC signature verification
- ✅ Evolution engine file writes require approval-gated execution context
- ✅ Sandbox refuses uncontained code execution by default
- ✅ Private keys (private.key, *.pem, *.p12, *.pfx, *.jks) gitignored
- ✅ All `hashlib.md5()` calls use `usedforsecurity=False`
- ✅ SQL queries are parameterized where the query builder accepts parameters

---

## Known Limitations

- **Test coverage < 5%** — only integration scripts at the project root; no unit tests
- **Dead code** — ~595 unused symbols identified, mostly in `src/utils/`
- **No CI/CD** — workflows exist but no automated deployment
- **Sandbox isolation** — depends on Docker availability; in non-strict mode may fall back to local subprocess (with explicit warning)
- **JWT rotation** — single shared `JWT_SECRET`; no key rotation strategy implemented

---

## License

[License information — proprietary]

---

## Acknowledgments

Built on top of OpenClaw, llm-council integration, and 89+ Claude Code plugins. Self-evolution engine inspired by Trail of Bits' evolution framework.
