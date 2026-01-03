# King AI v2 - Developer Documentation

## 1. System Architecture
King AI v2 is a hybrid "Dual-Brain" autonomous business empire. It is designed to run its heavy AI inference on a dedicated Cloud GPU while maintaining a lightweight dashboard accessible from anywhere.

### Component Overview
1.  **Backend (FastAPI)**:
    *   **Port**: 8000
    *   **Location**: `src/api`
    *   **Role**: Handles all business logic, database interactions, and AI orchestration.
2.  **Frontend (React/Vite)**:
    *   **Port**: 5173
    *   **Location**: `dashboard/`
    *   **Role**: Glassmorphic UI for the user to interact with the CEO and view empire stats.
3.  **Database (PostgreSQL & Redis)**:
    *   **Role**: PG stores persistent business data; Redis handles caching and task queues.
    *   **Deployment**: Runs via Docker Compose.
4.  **Load Balancer (Nginx)**:
    *   **Port**: 80 (internal)
    *   **Role**: Reverse proxy routing traffic to API and dashboard.
    *   **SSL Termination**: Handled by AWS Load Balancer.

### Deployment Architecture

**Production Setup (AWS):**
- **Domain**: https://king-ai-studio.me
- **Load Balancer**: AWS ALB (ports 80/443)
- **EC2 Instance**: Ubuntu 22.04
- **SSL**: AWS Certificate Manager
- **Health Checks**: `/api/docs` endpoint

**Traffic Flow:**
```
Internet → AWS ALB (443) → EC2:80 (Nginx) → API:8000 or Dashboard:5173
```

**Development URLs:**
- Dashboard: http://localhost:5173
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs

**Production URLs:**
- Dashboard: https://king-ai-studio.me
- API: https://king-ai-studio.me/api/
- API Docs: https://king-ai-studio.me/api/docs

---
As of v2.0, the system supports:
1.  **Dual-Brain Intelligence**: Automatically switches between AWS Ollama (Cost-effective) and Gemini (High-intelligence fallback).
2.  **Business Lifecycle Verification**: Can take a business idea from "Discovery" to "Active" using the `LifecycleEngine` state machine.
3.  **Autonomous Chat**: The "CEO" can answer queries about the empire ("How much revenue?", "What is PetPal doing?") by querying the Postgres database.
4.  **Remote Persistance**: All data is stored in the remote PostgreSQL instance, preserving state even if the dashboard is closed.

---

## 2. System Logic & Data Flow
The following sequence diagram illustrates how a user request (e.g., "Start a business") flows through the system.

```mermaid
sequenceDiagram
    participant User
    participant Dash as Dashboard (React)
    participant API as FastAPI
    participant Brain as MasterAI
    participant LLM as Ollama/Gemini
    participant DB as Postgres

    User->>Dash: "Start a dropshipping business"
    Dash->>API: POST /api/chat
    API->>Brain: process_input()
    
    rect rgb(20, 20, 20)
        Note over Brain,LLM: Intent Classification
        Brain->>LLM: "Classify intent of: Start dropshipping..."
        LLM-->>Brain: {"type": "command", "domain": "commerce"}
    end

    Brain->>Brain: _handle_command()
    Brain->>DB: Create Business Unit (Status: Discovery)
    DB-->>Brain: Success (ID: 123)

    rect rgb(20, 20, 20)
        Note over Brain,LLM: Generation
        Brain->>LLM: "Generate welcome message for new business..."
        LLM-->>Brain: "Dropshipping unit initialized..."
    end

    Brain-->>API: Response + Actions Taken
    API-->>Dash: JSON Response
    Dash-->>User: Displays Message & Updates Empire View
## 3. Approval System

The Approval System provides comprehensive human-in-the-loop workflow for managing actions requiring manual review.

### Features
- **Risk-Based Routing**: Automatically route requests by risk level (Low, Medium, High, Critical)
- **Auto-Approval**: Configure policies for low-risk actions
- **Multiple Action Types**: Financial, Legal, Operational, Strategic, Technical, External
- **Audit Trail**: Complete history of all decisions
- **Real-time Filtering**: Filter by risk level
- **Statistics**: Track approval rates and wait times

### Backend Usage

#### Creating an Approval Request
```python
from src.approvals.manager import ApprovalManager
from src.approvals.models import ApprovalType, RiskLevel, RiskFactor

manager = ApprovalManager()

request = await manager.create_request(
    business_id="business_123",
    action_type=ApprovalType.FINANCIAL,
    title="Server Infrastructure Upgrade",
    description="Purchase new cloud server instances",
    payload={"amount": 5000, "provider": "AWS"},
    risk_level=RiskLevel.HIGH,
    risk_factors=[
        RiskFactor(
            category="Financial",
            description="Large expenditure requiring budget approval",
            severity=RiskLevel.HIGH
        )
    ]
)
```

#### Approving a Request
```python
approved_request = await manager.approve(
    request_id=request.id,
    user_id="admin@example.com",
    notes="Approved after budget review"
)
```

#### Frontend Integration
```javascript
// Get pending approvals
const approvals = await fetch('/api/approvals/pending');

// Approve request
await fetch(`/api/approvals/${requestId}/approve`, {
  method: 'POST',
  body: JSON.stringify({ notes: "Approved" })
});
```

---

## 4. Deployment & Infrastructure

### Automated Deployment (Recommended)

The easiest way to deploy is using the automated control script:

```bash
python scripts/control.py
# Select [3] 🤖 Automated Empire Setup (AWS Infra + GitHub + Full Setup)
```

This automatically handles:
- AWS infrastructure detection and deployment
- Database and Redis setup
- Application deployment
- Environment configuration

### Manual AWS Deployment

For full control, follow these steps:

#### Prerequisites
- AWS Account with AdministratorAccess
- Terraform 1.0+
- AWS CLI configured
- SSH key pair

#### Infrastructure Components
- **VPC**: Multi-AZ with public/private subnets
- **EC2**: g4dn.xlarge instance for GPU workloads
- **RDS PostgreSQL**: db.r6g.xlarge with Multi-AZ
- **ElastiCache Redis**: cache.r6g.large
- **Application Load Balancer**: Distributes traffic
- **SQS**: Request queuing for inference
- **Auto-scaling**: Scales 2-8 GPU instances

#### Deployment Steps

1. **Create S3 Bucket for Terraform State**
```bash
aws s3 mb s3://king-ai-terraform-state --region us-east-1
aws s3api put-bucket-versioning --bucket king-ai-terraform-state --versioning-configuration Status=Enabled
```

2. **Configure Terraform Variables**
Create `infrastructure/terraform/terraform.tfvars`:
```hcl
aws_region = "us-east-1"
environment = "prod"
gpu_instance_count = 2
db_instance_class = "db.r6g.xlarge"
redis_node_type = "cache.r6g.large"
```

3. **Deploy Infrastructure**
```bash
cd infrastructure/terraform
terraform init
terraform plan
terraform apply
```

4. **Extract Endpoints**
```bash
terraform output -raw rds_endpoint
terraform output -raw redis_endpoint
terraform output -raw alb_dns_name
terraform output -raw ec2_public_ip
```

5. **Update Environment Configuration**
The automated script handles this, but manually:
```bash
# Get database password from AWS Secrets Manager
aws secretsmanager get-secret-value --secret-id king-ai/prod/db-password --query SecretString --output text

# Update .env file with AWS endpoints
DATABASE_URL=postgresql+asyncpg://kingadmin:password@rds-endpoint/kingai
REDIS_URL=redis://redis-endpoint:6379
VLLM_URL=http://alb-dns:8080
```

### Service Management

#### On AWS Instance
```bash
# Connect via SSH
ssh -i "king-ai-key.pem" ubuntu@ec2-public-ip

# Restart services
cd king-ai-v2
./start.sh

# View logs
tail -f backend.log
tail -f dashboard/frontend.log

# Check service status
sudo systemctl status king-ai-api
```

#### Local Development
```bash
# Start API server
python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Start dashboard
cd dashboard && npm run dev

# Start databases
docker-compose up -d
```

---

## 5. Environment & Configuration

### Required Variables
```env
# Database
DATABASE_URL=postgresql+asyncpg://king:password@localhost:5432/kingai
REDIS_URL=redis://localhost:6379

# AI Configuration
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
GEMINI_API_KEY=AIzaSy...

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
```

### Optional Integrations
```env
# Cloud Providers
ANTHROPIC_API_KEY=sk-ant-api03-...
VLLM_URL=http://alb-dns:8080

# Business Integrations
SHOPIFY_SHOP_URL=store.myshopify.com
SHOPIFY_ACCESS_TOKEN=token
STRIPE_API_KEY=sk_live_...

# Monitoring
DATADOG_API_KEY=...
DATADOG_APP_KEY=...
```

### Security Notes
- **Never commit .env files** to version control
- Use AWS Secrets Manager for production secrets
- Rotate API keys regularly
- Enable MFA on all accounts

---

## 6. Key Code Paths

### Core Components
*   **AI Logic**: `src/master_ai/brain.py` - Central orchestrator
*   **Response Fallback**: `src/master_ai/brain.py -> _call_llm()` - LLM routing
*   **Business Logic**: `src/business/lifecycle.py` - State machine
*   **Frontend**: `dashboard/src/App.jsx` - Main UI controller
*   **API Routes**: `src/api/` - FastAPI endpoints
*   **Database Models**: `src/database/` - SQLAlchemy models

### Agent System
*   **Base Agent**: `src/agents/base.py` - Agent framework
*   **Specialized Agents**: `src/agents/` - Domain-specific agents
*   **Capability Registry**: `src/agents/capability_registry.py` - Agent discovery

### Business Management
*   **Lifecycle Engine**: `src/business/lifecycle.py` - Business state machine
*   **Playbooks**: `config/playbooks/` - Business templates
*   **Portfolio**: `src/business/portfolio.py` - Multi-business management

### Integrations
*   **E-commerce**: `src/integrations/shopify.py`
*   **Payments**: `src/integrations/stripe.py`
*   **Banking**: `src/integrations/plaid.py`
*   **Analytics**: `src/integrations/google_analytics.py`

---

## 7. API Reference

### Core Endpoints

#### Chat & Commands
```
POST /api/chat
- Body: {"message": "Start a business"}
- Returns: AI response + executed actions
```

#### Business Management
```
GET /api/businesses - List all businesses
POST /api/businesses - Create new business
GET /api/businesses/{id} - Get business details
PUT /api/businesses/{id} - Update business
DELETE /api/businesses/{id} - Delete business
```

#### Approval System
```
GET /api/approvals/pending - Get pending approvals
POST /api/approvals/{id}/approve - Approve request
POST /api/approvals/{id}/reject - Reject request
GET /api/approvals/stats - Get approval statistics
```

#### Analytics
```
GET /api/analytics/overview - Empire overview
GET /api/analytics/business/{id} - Business-specific analytics
GET /api/analytics/revenue - Revenue metrics
```

### WebSocket
**Development:**
```
ws://localhost:8000/ws/chat - Real-time chat updates
ws://localhost:8000/ws/notifications - System notifications
```

**Production:**
```
wss://king-ai-studio.me/ws/chat - Real-time chat updates
wss://king-ai-studio.me/ws/notifications - System notifications
```

---

## 8. Testing & Quality Assurance

### Running Tests
```bash
# All tests
python -m pytest tests/ -v

# Specific test file
python -m pytest tests/test_business.py -v

# With coverage
python -m pytest tests/ --cov=src --cov-report=html
```

### Code Quality
```bash
# Linting
python -m ruff check src/

# Type checking
python -m mypy src/

# Security scanning
python -m bandit -r src/
```

### Test Categories
- **Unit Tests**: `tests/test_*.py` - Individual component tests
- **Integration Tests**: `tests/test_*_integration.py` - Multi-component tests
- **API Tests**: `tests/test_api_*.py` - Endpoint tests
- **Business Logic**: `tests/test_business_*.py` - Business rule tests

---

## 9. Troubleshooting

### Production Deployment Issues

#### Target Group Unhealthy
- **Cause**: Health check path returns 404
- **Solution**: Change health check path to `/api/docs` (returns 200)
- **AWS Console**: Target Groups → Health checks → Path: `/api/docs`

#### Load Balancer Connection Issues
- **Check Nginx**: `sudo systemctl status nginx`
- **Test local ports**: `curl http://localhost:80/api/docs`
- **Firewall**: `sudo ufw status`
- **Security Groups**: Allow inbound traffic from load balancer

#### SSL/HTTPS Issues
- **SSL Termination**: Handled by AWS Load Balancer
- **Certificate**: Check AWS Certificate Manager
- **Load Balancer Listeners**: Verify 443 → 80 forwarding

### Common Development Issues

#### "Connection to King AI Failed"
- Check if `uvicorn` is running: `ps aux | grep uvicorn`
- Verify `OLLAMA_URL` is reachable: `curl localhost:11434`
- Check API logs: `tail -f backend.log`

#### Database Connection Errors
- Ensure Docker containers are running: `docker ps`
- Check migration status: `alembic current`
- Verify connection string in `.env`

#### LLM Provider Issues
- Test Ollama: `curl http://localhost:11434/api/tags`
- Check API keys in `.env`
- Verify network connectivity to cloud providers

#### Permission Errors
- Check file permissions: `ls -la`
- Verify user has sudo access for system services
- Check AWS IAM permissions

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run with verbose output
python -m uvicorn src.api.main:app --reload --log-level debug
```

### Health Checks
```bash
# API health
curl http://localhost:8000/health

# Database connectivity
python -c "from src.database.connection import test_db; test_db()"

# Redis connectivity
python -c "from src.database.connection import test_redis; test_redis()"
```

---

## 10. Contributing

### Development Workflow
1. Create feature branch: `git checkout -b feature/new-feature`
2. Write tests for new functionality
3. Implement changes with proper error handling
4. Update documentation
5. Run full test suite: `python -m pytest tests/`
6. Submit pull request

### Code Standards
- **Python**: PEP 8 with Ruff linting
- **JavaScript**: ESLint configuration
- **Documentation**: Clear, comprehensive docs for all features
- **Testing**: 80%+ code coverage required
- **Security**: Regular dependency updates and security scans

### Architecture Decisions
- **FastAPI**: High-performance async web framework
- **SQLAlchemy**: Robust ORM with async support
- **Redis**: High-performance caching and queues
- **React**: Modern frontend framework
- **Docker**: Consistent deployment across environments

This documentation covers the complete King AI v2 system. For user-facing guides, see [USER_GUIDE.md](../USER_GUIDE.md).

---

## 11. Future Improvements Roadmap

### Critical Priority (🔴)
- **Database Connection Pool Manager**: Implement connection pooling with health checks and auto-reconnect
- **Graceful Shutdown Handler**: Handle SIGTERM/SIGINT gracefully with request draining
- **Request Correlation ID Middleware**: Generate unique request IDs for distributed tracing
- **Agent Retry Decorator**: Automatic retry for LLM calls with circuit breaker

### High Priority (🟠)
- **Background Task Queue with Redis**: Replace async tasks with proper queue and retry logic
- **Health Check Aggregator**: Aggregate all service health statuses for monitoring
- **Configuration Hot Reload**: Watch config files for changes without restart
- **Agent Response Validator**: JSON schema validation for LLM outputs
- **Conversation Memory Compressor**: Summarize old conversation turns to reduce token usage

### Medium Priority (🟡)
- **Database Migration Runner**: Run Alembic migrations programmatically
- **Metrics Prometheus Exporter**: Export custom metrics for monitoring
- **Structured Error Responses**: Consistent error response format with error codes
- **API Versioning Support**: URL-based API versioning with deprecation warnings

### Low Priority (🟢)
- **Multi-Agent Orchestrator**: Parallel agent execution with result aggregation
- **Plugin System**: Extensible agent and integration system
- **Advanced Analytics Dashboard**: Real-time business metrics and KPIs
- **Automated Testing Suite**: Comprehensive integration and E2E tests

For the complete improvement roadmap with detailed specifications, see the Future Improvements Roadmap section above.
