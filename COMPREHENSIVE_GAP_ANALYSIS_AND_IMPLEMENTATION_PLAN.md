# King AI v2 - Comprehensive Gap Analysis and Implementation Plan

## Executive Summary

After thorough analysis of the king-ai-v2 codebase against the original specification, the system is approximately **75-80% complete**. The core architecture is solid, but several critical components are missing or incomplete. This document provides a detailed gap analysis and actionable implementation plan for AI coding agents.

---

## Part 1: Current Implementation Status

### ✅ FULLY IMPLEMENTED

| Component | Location | Notes |
|-----------|----------|-------|
| Master AI Brain | `src/master_ai/brain.py` | Core orchestrator with intent classification |
| ReAct Planning | `src/master_ai/react_planner.py` | Reason-Act-Think pattern |
| Evolution Engine | `src/master_ai/evolution.py` | Self-modification proposals |
| LLM Router | `src/utils/llm_router.py` | Multi-provider routing (Ollama, vLLM, Gemini) |
| Sub-Agents | `src/agents/` | Research, Commerce, Finance, Content, Analytics, Legal, Code Generator |
| Agent Router | `src/agents/router.py` | Dynamic task delegation |
| Database Models | `src/database/models.py` | BusinessUnit, Task, EvolutionProposal |
| Approval System | `src/approvals/` | ApprovalManager with policies |
| Risk Profiles | `config/risk_profiles.yaml` | Conservative/Moderate/Aggressive |
| Playbook System | `config/playbooks/` | Dropshipping, SaaS templates |
| Business Lifecycle | `src/business/lifecycle.py` | State machine for business stages |
| Portfolio Manager | `src/business/portfolio.py` | Business portfolio management |
| KPI Monitor | `src/master_ai/kpi_monitor.py` | Redis-based metrics |
| Vector Store | `src/database/vector_store.py` | Pinecone integration |
| Web Dashboard | `dashboard/` | React + Vite frontend |
| FastAPI Backend | `src/api/` | REST API with WebSocket |
| Docker Setup | `Dockerfile`, `docker-compose.yml` | Containerization |
| Terraform | `infrastructure/terraform/` | AWS infrastructure (VPC, EC2, RDS, GPU) |
| Sandbox Testing | `src/utils/sandbox.py` | Docker-based isolated execution |
| ML Retraining | `src/master_ai/ml_retraining.py` | LoRA fine-tuning pipeline |
| Rollback Service | `src/master_ai/rollback_service.py` | Git-based rollback |
| Monitoring | `src/monitoring/` | Datadog, Arize AI integrations |
| Integrations | `src/integrations/` | Stripe, Shopify, Plaid, DALL-E, Suppliers |

### ⚠️ PARTIALLY IMPLEMENTED

| Component | Location | Missing |
|-----------|----------|---------|
| Autonomous Loop | `brain.py:436` | Not connected to scheduler/cron, no persistent state |
| Claude Fallback | `llm_router.py` | Defined but not implemented |
| vLLM Integration | Settings | vLLM URL not in settings.py |
| Test Coverage | `tests/` | 2 failing tests, limited coverage |
| CI/CD Pipeline | Missing | No GitHub Actions workflows |
| Kong API Gateway | Missing | Not implemented |
| LangSmith Tracing | Missing | Not implemented |
| Vanta/Snyk Security | Missing | Not integrated |
| Circuit Breaker | `llm_router.py` | Implemented but not fully tested |

### ❌ NOT IMPLEMENTED

| Component | Required By Spec | Priority |
|-----------|------------------|----------|
| Claude API Client | High-stakes fallback | HIGH |
| Scheduled Cron Jobs | Autonomous mode | HIGH |
| GitHub Actions CI/CD | Version control | HIGH |
| Kong API Gateway | API security | MEDIUM |
| LangSmith Integration | LLM tracing | MEDIUM |
| Vanta Compliance | Security | LOW |
| Snyk Security Scanning | Security | LOW |
| Multi-AZ ALB Configuration | High availability | MEDIUM |
| Auto-scaling Policies | Load management | MEDIUM |
| Two-Approver System | High-risk approvals | MEDIUM |
| Evolution Proposal Dashboard Tab | Dashboard UI | MEDIUM |

---

## Part 2: Bugs and Issues Found

### BUG 1: Lifecycle Transition Test Failure
**File:** `src/business/lifecycle.py`
**Issue:** SUNSET status returns `None` instead of staying at SUNSET

```python
# Current code has no transition for SUNSET
TRANSITIONS = {
    BusinessStatus.DISCOVERY: BusinessStatus.VALIDATION,
    BusinessStatus.VALIDATION: BusinessStatus.SETUP,
    # ... SUNSET is missing
}
```

**Fix Required:**
```python
# Add SUNSET -> SUNSET or handle as terminal state
def get_next_status(self, current_status: BusinessStatus) -> BusinessStatus | None:
    if current_status == BusinessStatus.SUNSET:
        return BusinessStatus.SUNSET  # Terminal state stays the same
    return self.TRANSITIONS.get(current_status)
```

### BUG 2: Portfolio Metrics Test Failure
**File:** `src/business/portfolio.py`
**Issue:** `get_total_stats()` method not properly aggregating from database

**Fix Required:** The method needs to actually query the database and aggregate results.

### BUG 3: Pydantic Deprecation Warnings
**File:** `config/settings.py`
**Issue:** Using deprecated `env=` syntax in Field()

**Fix Required:**
```python
# Change from:
database_url: str = Field(..., env="DATABASE_URL")
# To:
database_url: str = Field(..., validation_alias="DATABASE_URL")
```

### BUG 4: Evolution Engine Constructor Mismatch
**File:** `src/master_ai/brain.py:55`
**Issue:** Evolution engine initialized with `self.llm_router.ollama` but constructor expects `LLMRouter`

### BUG 5: Missing Error Handling in Autonomous Loop
**File:** `src/master_ai/brain.py:436-460`
**Issue:** Loop continues silently on errors, no alerting

---

## Part 3: Implementation Plan for AI Agents

### PHASE 1: Critical Bug Fixes (Estimated: 2 hours)

#### Task 1.1: Fix Lifecycle Transition Bug
**File to modify:** `src/business/lifecycle.py`
**Instructions:**
1. Open `src/business/lifecycle.py`
2. Find the `BasicLifecycleEngine` class
3. Modify `get_next_status` method to handle SUNSET as terminal state:

```python
def get_next_status(self, current_status: BusinessStatus) -> BusinessStatus | None:
    """
    Returns the next logical stage for a business unit.
    SUNSET is a terminal state and returns itself.
    """
    if current_status == BusinessStatus.SUNSET:
        return BusinessStatus.SUNSET
    return self.TRANSITIONS.get(current_status)
```

#### Task 1.2: Fix Portfolio Metrics
**File to modify:** `src/business/portfolio.py`
**Instructions:**
1. Find or create `get_total_stats()` method in `PortfolioManager`
2. Implement proper database aggregation:

```python
async def get_total_stats(self) -> dict:
    """Get aggregate stats across all businesses."""
    from sqlalchemy import select, func
    from src.database.models import BusinessUnit
    from src.database.connection import get_db
    
    async with get_db() as db:
        result = await db.execute(
            select(
                func.sum(BusinessUnit.total_revenue).label('total_revenue'),
                func.sum(BusinessUnit.total_expenses).label('total_expenses'),
                func.count(BusinessUnit.id).label('count')
            )
        )
        row = result.first()
        
        return {
            "total_revenue": row.total_revenue or 0,
            "total_expenses": row.total_expenses or 0,
            "total_profit": (row.total_revenue or 0) - (row.total_expenses or 0),
            "business_count": row.count or 0
        }
```

#### Task 1.3: Fix Pydantic Deprecation Warnings
**File to modify:** `config/settings.py`
**Instructions:**
1. Update all Field declarations to use `validation_alias` instead of `env`:

```python
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Literal, Optional

class Settings(BaseSettings):
    """
    Main configuration class for King AI v2.
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    # Database Settings
    database_url: str = Field(..., validation_alias="DATABASE_URL")
    redis_url: str = Field(default="redis://localhost:6379", validation_alias="REDIS_URL")
    
    # Ollama / LLM Settings
    ollama_url: str = Field(..., validation_alias="OLLAMA_URL")
    ollama_model: str = Field(default="llama3.1:8b", validation_alias="OLLAMA_MODEL")
    
    # vLLM Settings (NEW - add these)
    vllm_url: Optional[str] = Field(default=None, validation_alias="VLLM_URL")
    vllm_model: str = Field(default="meta-llama/Llama-3.1-70B-Instruct", validation_alias="VLLM_MODEL")
    
    # Pinecone Settings
    pinecone_api_key: Optional[str] = Field(default=None, validation_alias="PINECONE_API_KEY")
    pinecone_index: str = Field(default="king-ai", validation_alias="PINECONE_INDEX")
    
    # Risk & Evolution Controls
    risk_profile: Literal["conservative", "moderate", "aggressive"] = "moderate"
    max_evolutions_per_hour: int = Field(default=5)
    enable_autonomous_mode: bool = Field(default=False)
    enable_self_modification: bool = Field(default=True)
    evolution_confidence_threshold: float = Field(default=0.8)
    
    # API Server Settings
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)

settings = Settings()
```

#### Task 1.4: Fix Evolution Engine Constructor
**File to modify:** `src/master_ai/brain.py`
**Instructions:**
1. Find line ~55 where EvolutionEngine is initialized
2. Change from:
```python
self.evolution = EvolutionEngine(self.llm_router.ollama)
```
3. To:
```python
self.evolution = EvolutionEngine(self.llm_router)
```

---

### PHASE 2: Claude API Fallback Implementation (Estimated: 4 hours)

#### Task 2.1: Create Claude Client
**Create new file:** `src/utils/claude_client.py`

```python
"""
Claude API Client for high-stakes fallback inference.
Uses Anthropic's API for complex reasoning tasks.
"""

import httpx
import asyncio
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import os


@dataclass
class ClaudeConfig:
    """Configuration for Claude client."""
    api_key: str
    model: str = "claude-3-5-sonnet-20241022"
    max_tokens: int = 4096
    timeout: float = 120.0


class ClaudeClient:
    """
    Async client for Anthropic's Claude API.
    Used for high-stakes decisions requiring superior reasoning.
    """
    
    def __init__(self, api_key: str = None, model: str = "claude-3-5-sonnet-20241022"):
        """
        Initialize Claude client.
        
        Args:
            api_key: Anthropic API key (or from ANTHROPIC_API_KEY env)
            model: Claude model to use
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model
        self.base_url = "https://api.anthropic.com/v1"
        self.client = httpx.AsyncClient(
            timeout=120.0,
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
        )
    
    async def complete(
        self,
        prompt: str,
        system: str = None,
        max_tokens: int = 4096,
        temperature: float = 0.7
    ) -> str:
        """
        Generate a completion using Claude.
        
        Args:
            prompt: The user prompt
            system: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text response
        """
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not configured")
        
        messages = [{"role": "user", "content": prompt}]
        
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        if system:
            payload["system"] = system
        
        response = await self.client.post(
            f"{self.base_url}/messages",
            json=payload
        )
        response.raise_for_status()
        
        data = response.json()
        return data["content"][0]["text"]
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
    
    def is_available(self) -> bool:
        """Check if Claude is configured."""
        return bool(self.api_key)
```

#### Task 2.2: Integrate Claude into LLM Router
**File to modify:** `src/utils/llm_router.py`
**Instructions:**

1. Add import at top:
```python
from src.utils.claude_client import ClaudeClient
```

2. In `__init__`, add after gemini initialization:
```python
# Quaternary: Claude for high-stakes fallback
self.claude: Optional[ClaudeClient] = None
if os.getenv("ANTHROPIC_API_KEY"):
    self.claude = ClaudeClient()
    
# Add to health tracking
self._provider_health[ProviderType.CLAUDE] = True
self._failure_counts[ProviderType.CLAUDE] = 0
self._circuit_open[ProviderType.CLAUDE] = False
self._circuit_open_until[ProviderType.CLAUDE] = 0
```

3. Update `_route` method to use Claude for high-stakes:
```python
async def _route(self, context: TaskContext | None) -> RoutingDecision:
    """Determine the best provider for this request."""
    
    # High-stakes tasks use Claude if available
    if context and context.risk_level == "high" and context.requires_accuracy:
        if self.claude and self.claude.is_available() and self._provider_health[ProviderType.CLAUDE]:
            return RoutingDecision(
                provider=ProviderType.CLAUDE,
                reason="High-stakes task routed to Claude for accuracy"
            )
        elif self.gemini and self._provider_health[ProviderType.GEMINI]:
            return RoutingDecision(
                provider=ProviderType.GEMINI,
                reason="High-stakes task routed to Gemini (Claude unavailable)"
            )
    
    # Default to vLLM if available and healthy
    if self.vllm and self._provider_health[ProviderType.VLLM]:
        return RoutingDecision(
            provider=ProviderType.VLLM,
            reason="Production routing to vLLM"
        )
    
    # Fallback to Ollama
    return RoutingDecision(
        provider=ProviderType.OLLAMA,
        reason="Fallback to Ollama"
    )
```

4. Update `_execute` method to handle Claude:
```python
async def _execute(self, provider: ProviderType, prompt: str, system: str = None) -> str:
    """Execute inference on a specific provider."""
    if provider == ProviderType.VLLM:
        return await self.vllm.complete(prompt, system)
    elif provider == ProviderType.OLLAMA:
        return await self.ollama.complete(prompt, system)
    elif provider == ProviderType.GEMINI:
        return await self.gemini.complete(prompt, system)
    elif provider == ProviderType.CLAUDE:
        return await self.claude.complete(prompt, system)
    else:
        raise ValueError(f"Unknown provider: {provider}")
```

5. Update `_get_fallback_chain` to include Claude:
```python
def _get_fallback_chain(self, primary: ProviderType) -> List[ProviderType]:
    """Get ordered list of providers for fallback."""
    all_providers = [
        ProviderType.CLAUDE,  # Highest quality
        ProviderType.GEMINI,
        ProviderType.VLLM,
        ProviderType.OLLAMA   # Always available
    ]
    
    # Filter to available providers, primary first
    available = [primary]
    for p in all_providers:
        if p != primary and self._is_provider_available(p):
            available.append(p)
    
    return available

def _is_provider_available(self, provider: ProviderType) -> bool:
    """Check if provider is configured and available."""
    if provider == ProviderType.CLAUDE:
        return self.claude and self.claude.is_available()
    elif provider == ProviderType.GEMINI:
        return self.gemini is not None
    elif provider == ProviderType.VLLM:
        return self.vllm is not None
    elif provider == ProviderType.OLLAMA:
        return True  # Always available
    return False
```

---

### PHASE 3: CI/CD Pipeline Implementation (Estimated: 3 hours)

#### Task 3.1: Create GitHub Actions Workflow
**Create new file:** `.github/workflows/ci.yml`

```yaml
name: King AI v2 CI/CD

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  PYTHON_VERSION: '3.12'
  NODE_VERSION: '20'

jobs:
  test:
    name: Run Tests
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:16
        env:
          POSTGRES_USER: king_ai
          POSTGRES_PASSWORD: test_password
          POSTGRES_DB: king_ai_test
        ports:
          - 5432:5432
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
      
      redis:
        image: redis:7
        ports:
          - 6379:6379
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: 'pip'
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev]"
      
      - name: Run linting
        run: |
          ruff check src/ tests/
      
      - name: Run type checking
        run: |
          mypy src/ --ignore-missing-imports
      
      - name: Run tests
        env:
          DATABASE_URL: postgresql+asyncpg://king_ai:test_password@localhost:5432/king_ai_test
          REDIS_URL: redis://localhost:6379
          OLLAMA_URL: http://localhost:11434
          OLLAMA_MODEL: llama3.1:8b
        run: |
          pytest tests/ -v --tb=short --json-report --json-report-file=test-results.json
      
      - name: Upload test results
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: test-results
          path: test-results.json

  build-dashboard:
    name: Build Dashboard
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Node.js
        uses: actions/setup-node@v4
        with:
          node-version: ${{ env.NODE_VERSION }}
          cache: 'npm'
          cache-dependency-path: dashboard/package.json
      
      - name: Install dependencies
        working-directory: dashboard
        run: npm ci
      
      - name: Build
        working-directory: dashboard
        run: npm run build
      
      - name: Upload build artifacts
        uses: actions/upload-artifact@v4
        with:
          name: dashboard-build
          path: dashboard/dist

  docker-build:
    name: Build Docker Image
    runs-on: ubuntu-latest
    needs: [test, build-dashboard]
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3
      
      - name: Build Docker image
        uses: docker/build-push-action@v5
        with:
          context: .
          push: false
          tags: king-ai-v2:${{ github.sha }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

  deploy:
    name: Deploy to AWS
    runs-on: ubuntu-latest
    needs: docker-build
    if: github.ref == 'refs/heads/main'
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-east-1
      
      - name: Login to Amazon ECR
        id: login-ecr
        uses: aws-actions/amazon-ecr-login@v2
      
      - name: Build, tag, and push image to ECR
        env:
          ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
          ECR_REPOSITORY: king-ai-v2
          IMAGE_TAG: ${{ github.sha }}
        run: |
          docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG .
          docker push $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG
          docker tag $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG $ECR_REGISTRY/$ECR_REPOSITORY:latest
          docker push $ECR_REGISTRY/$ECR_REPOSITORY:latest
      
      - name: Deploy to ECS
        run: |
          aws ecs update-service --cluster king-ai-cluster --service king-ai-service --force-new-deployment
```

#### Task 3.2: Create Terraform Deploy Workflow
**Create new file:** `.github/workflows/terraform.yml`

```yaml
name: Terraform Infrastructure

on:
  push:
    branches: [main]
    paths:
      - 'infrastructure/terraform/**'
  pull_request:
    paths:
      - 'infrastructure/terraform/**'

env:
  TF_VERSION: '1.6.0'

jobs:
  terraform:
    name: Terraform Plan/Apply
    runs-on: ubuntu-latest
    
    defaults:
      run:
        working-directory: infrastructure/terraform
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Terraform
        uses: hashicorp/setup-terraform@v3
        with:
          terraform_version: ${{ env.TF_VERSION }}
      
      - name: Configure AWS Credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-east-1
      
      - name: Terraform Init
        run: terraform init
      
      - name: Terraform Plan
        run: terraform plan -out=tfplan
      
      - name: Terraform Apply
        if: github.ref == 'refs/heads/main' && github.event_name == 'push'
        run: terraform apply -auto-approve tfplan
```

---

### PHASE 4: Scheduled Autonomous Loop (Estimated: 3 hours)

#### Task 4.1: Create Scheduler Service
**Create new file:** `src/services/scheduler.py`

```python
"""
Scheduler Service - Manages cron-based autonomous operations.
Implements scheduled tasks for the Master AI brain.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import uuid

from src.utils.logging import get_logger
from config.settings import settings

logger = get_logger("scheduler")


class TaskFrequency(str, Enum):
    """Predefined task frequencies."""
    HOURLY = "hourly"
    EVERY_6_HOURS = "every_6_hours"
    DAILY = "daily"
    WEEKLY = "weekly"


@dataclass
class ScheduledTask:
    """A scheduled task definition."""
    id: str
    name: str
    frequency: TaskFrequency
    callback: Callable
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    enabled: bool = True
    error_count: int = 0
    max_errors: int = 5
    metadata: Dict[str, Any] = field(default_factory=dict)


class Scheduler:
    """
    Async scheduler for autonomous operations.
    
    Manages scheduled tasks like:
    - Autonomous optimization loop (every 6 hours)
    - KPI health checks (hourly)
    - Evolution proposals (daily)
    - Business unit reviews (daily)
    """
    
    def __init__(self):
        self._tasks: Dict[str, ScheduledTask] = {}
        self._running = False
        self._task: Optional[asyncio.Task] = None
        
        # Frequency intervals in seconds
        self._intervals = {
            TaskFrequency.HOURLY: 3600,
            TaskFrequency.EVERY_6_HOURS: 6 * 3600,
            TaskFrequency.DAILY: 24 * 3600,
            TaskFrequency.WEEKLY: 7 * 24 * 3600,
        }
    
    def register_task(
        self,
        name: str,
        callback: Callable,
        frequency: TaskFrequency,
        enabled: bool = True
    ) -> str:
        """
        Register a scheduled task.
        
        Args:
            name: Task name
            callback: Async function to call
            frequency: How often to run
            enabled: Whether task is active
            
        Returns:
            Task ID
        """
        task_id = str(uuid.uuid4())
        
        task = ScheduledTask(
            id=task_id,
            name=name,
            frequency=frequency,
            callback=callback,
            enabled=enabled,
            next_run=datetime.utcnow()  # Run immediately on first pass
        )
        
        self._tasks[task_id] = task
        logger.info(f"Registered scheduled task: {name} ({frequency.value})")
        
        return task_id
    
    def unregister_task(self, task_id: str) -> bool:
        """Unregister a task."""
        if task_id in self._tasks:
            del self._tasks[task_id]
            return True
        return False
    
    def enable_task(self, task_id: str) -> bool:
        """Enable a task."""
        if task_id in self._tasks:
            self._tasks[task_id].enabled = True
            return True
        return False
    
    def disable_task(self, task_id: str) -> bool:
        """Disable a task."""
        if task_id in self._tasks:
            self._tasks[task_id].enabled = False
            return True
        return False
    
    async def start(self):
        """Start the scheduler loop."""
        if self._running:
            logger.warning("Scheduler already running")
            return
        
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("Scheduler started")
    
    async def stop(self):
        """Stop the scheduler."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Scheduler stopped")
    
    async def _run_loop(self):
        """Main scheduler loop."""
        while self._running:
            try:
                now = datetime.utcnow()
                
                for task in self._tasks.values():
                    if not task.enabled:
                        continue
                    
                    if task.next_run and now >= task.next_run:
                        await self._execute_task(task)
                
                # Check every minute
                await asyncio.sleep(60)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scheduler error: {e}", exc_info=True)
                await asyncio.sleep(60)
    
    async def _execute_task(self, task: ScheduledTask):
        """Execute a single scheduled task."""
        logger.info(f"Executing scheduled task: {task.name}")
        
        try:
            await task.callback()
            
            # Update timing
            task.last_run = datetime.utcnow()
            task.next_run = task.last_run + timedelta(
                seconds=self._intervals[task.frequency]
            )
            task.error_count = 0
            
            logger.info(
                f"Task completed: {task.name}",
                next_run=task.next_run.isoformat()
            )
            
        except Exception as e:
            task.error_count += 1
            logger.error(
                f"Task failed: {task.name}",
                error=str(e),
                error_count=task.error_count
            )
            
            # Disable if too many errors
            if task.error_count >= task.max_errors:
                task.enabled = False
                logger.warning(f"Task disabled due to repeated failures: {task.name}")
            else:
                # Retry in 5 minutes
                task.next_run = datetime.utcnow() + timedelta(minutes=5)
    
    def get_status(self) -> Dict[str, Any]:
        """Get scheduler status."""
        return {
            "running": self._running,
            "tasks": [
                {
                    "id": t.id,
                    "name": t.name,
                    "frequency": t.frequency.value,
                    "enabled": t.enabled,
                    "last_run": t.last_run.isoformat() if t.last_run else None,
                    "next_run": t.next_run.isoformat() if t.next_run else None,
                    "error_count": t.error_count
                }
                for t in self._tasks.values()
            ]
        }


# Global scheduler instance
scheduler = Scheduler()
```

#### Task 4.2: Integrate Scheduler with API Startup
**File to modify:** `src/api/main.py`
**Instructions:**

Add to imports:
```python
from src.services.scheduler import scheduler, TaskFrequency
```

Modify lifespan function:
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles application startup and shutdown events.
    """
    global master_ai
    
    # --- Startup ---
    await init_db()
    master_ai = MasterAI()
    
    # Register scheduled tasks
    if settings.enable_autonomous_mode:
        scheduler.register_task(
            name="autonomous_optimization",
            callback=master_ai.run_autonomous_iteration,
            frequency=TaskFrequency.EVERY_6_HOURS,
            enabled=True
        )
        
        scheduler.register_task(
            name="business_health_check",
            callback=master_ai._check_business_health_scheduled,
            frequency=TaskFrequency.HOURLY,
            enabled=True
        )
        
        scheduler.register_task(
            name="evolution_consideration",
            callback=master_ai._consider_evolution_scheduled,
            frequency=TaskFrequency.DAILY,
            enabled=True
        )
        
        await scheduler.start()
    
    yield
    
    # --- Shutdown ---
    await scheduler.stop()
```

#### Task 4.3: Add Scheduler Methods to MasterAI
**File to modify:** `src/master_ai/brain.py`
**Instructions:**

Add these methods to the MasterAI class:
```python
async def run_autonomous_iteration(self):
    """Single iteration of the autonomous loop for scheduler."""
    logger.info("Running scheduled autonomous iteration")
    
    try:
        context = await self.context.build_context()
        
        # Self-improvement analysis
        await self._consider_evolution(context)
        
        # Business unit health check
        await self._check_business_health(context)
        
        logger.info("Autonomous iteration complete")
        
    except Exception as e:
        logger.error("Autonomous iteration failed", exc_info=True)
        raise

async def _check_business_health_scheduled(self):
    """Scheduled wrapper for health check."""
    context = await self.context.build_context()
    await self._check_business_health(context)

async def _consider_evolution_scheduled(self):
    """Scheduled wrapper for evolution consideration."""
    context = await self.context.build_context()
    await self._consider_evolution(context)
```

---

### PHASE 5: Dashboard Evolution Tab (Estimated: 2 hours)

#### Task 5.1: Add Evolution API Routes
**File to modify:** `src/api/routes/evolution.py`
**Instructions:**

Ensure these endpoints exist:
```python
from fastapi import APIRouter, HTTPException
from typing import List
from pydantic import BaseModel

router = APIRouter()


class EvolutionProposalResponse(BaseModel):
    id: str
    type: str
    title: str
    description: str
    status: str
    confidence_score: float
    risk_level: str
    created_at: str


@router.get("/proposals", response_model=List[EvolutionProposalResponse])
async def list_proposals(status: str = None, limit: int = 20):
    """List evolution proposals."""
    from src.database.connection import get_db
    from src.database.models import EvolutionProposal
    from sqlalchemy import select
    
    async with get_db() as db:
        query = select(EvolutionProposal).limit(limit).order_by(
            EvolutionProposal.created_at.desc()
        )
        
        if status:
            query = query.where(EvolutionProposal.status == status)
        
        result = await db.execute(query)
        proposals = result.scalars().all()
        
        return [
            EvolutionProposalResponse(
                id=p.id,
                type=p.type,
                title=p.description[:100],
                description=p.description,
                status=p.status.value,
                confidence_score=p.confidence_score,
                risk_level="medium",  # TODO: Add to model
                created_at=p.created_at.isoformat()
            )
            for p in proposals
        ]


@router.post("/proposals/{proposal_id}/approve")
async def approve_proposal(proposal_id: str, notes: str = None):
    """Approve an evolution proposal."""
    # Implementation here
    pass


@router.post("/proposals/{proposal_id}/reject")
async def reject_proposal(proposal_id: str, notes: str = None):
    """Reject an evolution proposal."""
    # Implementation here
    pass
```

#### Task 5.2: Add Evolution Tab to Dashboard
**File to modify:** `dashboard/src/App.jsx`
**Instructions:**

1. Add evolution state:
```jsx
const [evolutions, setEvolutions] = useState([]);
```

2. Add evolution fetch in useEffect:
```jsx
const evoRes = await fetch(`${API_BASE}/evolution/proposals`);
if (evoRes.ok) {
  setEvolutions(await evoRes.json());
}
```

3. Add evolution tab button in nav:
```jsx
<button
  className={`btn-nav ${activeTab === 'evolution' ? 'active' : ''}`}
  onClick={() => setActiveTab('evolution')}
>
  🧬 Evolution
</button>
```

4. Add evolution tab content:
```jsx
{activeTab === 'evolution' && (
  <div className="content-fade-in">
    <div className="card glass">
      <h3 style={{ marginBottom: '20px' }}>Evolution Proposals</h3>
      {evolutions.length === 0 ? (
        <p style={{ color: 'var(--text-dim)' }}>No evolution proposals pending.</p>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
          {evolutions.map(evo => (
            <div key={evo.id} className="glass" style={{ padding: '16px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                <div>
                  <strong>{evo.title}</strong>
                  <div style={{ fontSize: '0.8rem', color: 'var(--text-dim)', marginTop: '4px' }}>
                    {evo.type} • Confidence: {(evo.confidence_score * 100).toFixed(1)}%
                  </div>
                  <p style={{ marginTop: '8px', fontSize: '0.9rem' }}>{evo.description}</p>
                </div>
                <div style={{ display: 'flex', gap: '8px' }}>
                  {evo.status === 'pending' && (
                    <>
                      <button 
                        className="btn-primary"
                        onClick={() => handleApproveEvolution(evo.id)}
                        style={{ padding: '8px 16px', fontSize: '0.8rem' }}
                      >
                        ✓ Approve
                      </button>
                      <button 
                        className="btn-secondary"
                        onClick={() => handleRejectEvolution(evo.id)}
                        style={{ padding: '8px 16px', fontSize: '0.8rem' }}
                      >
                        ✗ Reject
                      </button>
                    </>
                  )}
                  {evo.status !== 'pending' && (
                    <span className={`status-badge ${evo.status}`}>
                      {evo.status}
                    </span>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  </div>
)}
```

5. Add handler functions:
```jsx
const handleApproveEvolution = async (id) => {
  await fetch(`${API_BASE}/evolution/proposals/${id}/approve`, { method: 'POST' });
  // Refresh evolutions
  const evoRes = await fetch(`${API_BASE}/evolution/proposals`);
  if (evoRes.ok) setEvolutions(await evoRes.json());
};

const handleRejectEvolution = async (id) => {
  await fetch(`${API_BASE}/evolution/proposals/${id}/reject`, { method: 'POST' });
  // Refresh evolutions  
  const evoRes = await fetch(`${API_BASE}/evolution/proposals`);
  if (evoRes.ok) setEvolutions(await evoRes.json());
};
```

---

### PHASE 6: LangSmith Integration (Estimated: 2 hours)

#### Task 6.1: Create LangSmith Client
**Create new file:** `src/utils/langsmith_client.py`

```python
"""
LangSmith Integration - LLM tracing and evaluation.
Provides observability for all LLM interactions.
"""

import os
from typing import Optional, Dict, Any, Callable
from datetime import datetime
from contextlib import asynccontextmanager
from functools import wraps
import uuid

from src.utils.logging import get_logger

logger = get_logger("langsmith")

# Check if LangSmith is available
try:
    from langsmith import Client, traceable
    from langsmith.run_helpers import get_current_run_tree, trace
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    logger.info("LangSmith not installed")


class LangSmithTracer:
    """
    LangSmith tracing integration for King AI.
    
    Features:
    - Trace all LLM calls
    - Track token usage and latency
    - Enable feedback collection
    - Support for custom metadata
    """
    
    def __init__(self):
        self.enabled = LANGSMITH_AVAILABLE and os.getenv("LANGCHAIN_API_KEY")
        
        if self.enabled:
            self.client = Client()
            self.project_name = os.getenv("LANGCHAIN_PROJECT", "king-ai-v2")
            logger.info(f"LangSmith tracing enabled for project: {self.project_name}")
        else:
            self.client = None
            logger.info("LangSmith tracing disabled")
    
    @asynccontextmanager
    async def trace_llm_call(
        self,
        name: str,
        inputs: Dict[str, Any],
        metadata: Dict[str, Any] = None
    ):
        """
        Context manager for tracing LLM calls.
        
        Usage:
            async with tracer.trace_llm_call("classify_intent", {"prompt": prompt}) as run:
                result = await llm.complete(prompt)
                run.end(outputs={"result": result})
        """
        if not self.enabled:
            yield DummyRun()
            return
        
        run_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        try:
            run = self.client.create_run(
                name=name,
                run_type="llm",
                inputs=inputs,
                extra=metadata or {},
                project_name=self.project_name
            )
            
            yield run
            
        except Exception as e:
            logger.error(f"LangSmith trace error: {e}")
            yield DummyRun()
    
    def trace(self, name: str = None, metadata: Dict[str, Any] = None):
        """
        Decorator for tracing functions.
        
        Usage:
            @tracer.trace("my_function")
            async def my_function(input):
                return result
        """
        def decorator(func: Callable):
            if not self.enabled:
                return func
            
            @wraps(func)
            async def wrapper(*args, **kwargs):
                trace_name = name or func.__name__
                
                with trace(
                    name=trace_name,
                    project_name=self.project_name,
                    extra=metadata or {}
                ):
                    return await func(*args, **kwargs)
            
            return wrapper
        return decorator
    
    def log_feedback(
        self,
        run_id: str,
        score: float,
        comment: str = None,
        feedback_type: str = "user"
    ):
        """Log user feedback for a run."""
        if not self.enabled:
            return
        
        try:
            self.client.create_feedback(
                run_id=run_id,
                key=feedback_type,
                score=score,
                comment=comment
            )
        except Exception as e:
            logger.error(f"Failed to log feedback: {e}")


class DummyRun:
    """Dummy run for when LangSmith is disabled."""
    
    def end(self, **kwargs):
        pass


# Global instance
langsmith_tracer = LangSmithTracer()
```

#### Task 6.2: Integrate LangSmith with LLM Router
**File to modify:** `src/utils/llm_router.py`
**Instructions:**

Add import:
```python
from src.utils.langsmith_client import langsmith_tracer
```

Modify `complete` method:
```python
async def complete(
    self,
    prompt: str,
    system: str | None = None,
    context: TaskContext | None = None
) -> str:
    """Route and execute an inference request with tracing."""
    
    async with langsmith_tracer.trace_llm_call(
        name="llm_router.complete",
        inputs={"prompt": prompt[:500], "system": system[:200] if system else None},
        metadata={
            "task_type": context.task_type if context else "unknown",
            "risk_level": context.risk_level if context else "unknown"
        }
    ) as run:
        decision = await self._route(context)
        providers = self._get_fallback_chain(decision.provider)
        
        last_error = None
        for provider in providers:
            if self._is_circuit_open(provider):
                continue
            
            try:
                start = time.time()
                result = await self._execute(provider, prompt, system)
                latency = (time.time() - start) * 1000
                
                self._record_success(provider)
                
                run.end(outputs={
                    "result": result[:500],
                    "provider": provider.value,
                    "latency_ms": latency
                })
                
                return result
                
            except Exception as e:
                last_error = e
                self._record_failure(provider)
                continue
        
        raise RuntimeError(f"All providers failed. Last error: {last_error}")
```

---

### PHASE 7: Enhanced Two-Approver System (Estimated: 2 hours)

#### Task 7.1: Update Approval Models
**File to modify:** `src/approvals/models.py`
**Instructions:**

Add two-approver support:
```python
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime


@dataclass
class ApprovalVote:
    """A single approval vote."""
    user_id: str
    decision: str  # "approve" or "reject"
    timestamp: datetime
    notes: Optional[str] = None


@dataclass 
class ApprovalRequest:
    """Extended approval request with multi-approver support."""
    id: str
    business_id: str
    action_type: ApprovalType
    title: str
    description: str
    risk_level: RiskLevel
    risk_factors: List[RiskFactor]
    payload: dict
    status: ApprovalStatus = ApprovalStatus.PENDING
    
    # Multi-approver fields
    required_approvers: int = 1
    votes: List[ApprovalVote] = field(default_factory=list)
    
    # Existing fields
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    reviewed_at: Optional[datetime] = None
    reviewed_by: Optional[str] = None
    review_notes: Optional[str] = None
    
    @property
    def approval_count(self) -> int:
        return sum(1 for v in self.votes if v.decision == "approve")
    
    @property
    def rejection_count(self) -> int:
        return sum(1 for v in self.votes if v.decision == "reject")
    
    @property
    def is_fully_approved(self) -> bool:
        return self.approval_count >= self.required_approvers
    
    def add_vote(self, user_id: str, decision: str, notes: str = None) -> bool:
        """Add a vote, returns True if this completes the approval."""
        # Check if user already voted
        if any(v.user_id == user_id for v in self.votes):
            return False
        
        self.votes.append(ApprovalVote(
            user_id=user_id,
            decision=decision,
            timestamp=datetime.utcnow(),
            notes=notes
        ))
        
        if decision == "reject":
            self.status = ApprovalStatus.REJECTED
            return False
        
        if self.is_fully_approved:
            self.status = ApprovalStatus.APPROVED
            return True
        
        return False
```

#### Task 7.2: Update Approval Manager
**File to modify:** `src/approvals/manager.py`
**Instructions:**

Update `create_request` to set `required_approvers`:
```python
async def create_request(
    self,
    business_id: str,
    action_type: ApprovalType,
    title: str,
    description: str,
    payload: dict,
    risk_level: RiskLevel = RiskLevel.MEDIUM,
    # ...existing params
) -> ApprovalRequest:
    """Create a new approval request."""
    
    # Find applicable policy
    policy = self._find_policy(action_type, risk_level)
    
    # Determine required approvers based on policy
    required_approvers = 1
    if policy and policy.require_two_approvers:
        required_approvers = 2
    
    # High-risk always requires 2
    if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
        required_approvers = max(required_approvers, 2)
    
    request = ApprovalRequest(
        id=str(uuid.uuid4()),
        business_id=business_id,
        action_type=action_type,
        title=title,
        description=description,
        risk_level=risk_level,
        payload=payload,
        required_approvers=required_approvers,
        # ... rest of fields
    )
    
    # ... rest of method
```

Update `approve` method:
```python
async def approve(
    self,
    request_id: str,
    user_id: str,
    notes: str = None,
) -> Optional[ApprovalRequest]:
    """Add an approval vote."""
    request = self._requests.get(request_id)
    if not request or request.status not in [ApprovalStatus.PENDING, ApprovalStatus.PARTIAL]:
        return None

    if request.is_expired:
        request.status = ApprovalStatus.EXPIRED
        return None

    # Add vote
    fully_approved = request.add_vote(user_id, "approve", notes)
    
    if not fully_approved and request.status == ApprovalStatus.PENDING:
        request.status = ApprovalStatus.PARTIAL
    
    # ... existing hook logic
    
    return request
```

---

## Part 4: Test Execution Checklist

After implementing all phases, run these verification steps:

### Step 1: Run All Tests
```bash
pytest tests/ -v --tb=short
```
Expected: All tests should pass.

### Step 2: Verify API Starts
```bash
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```
Expected: API should start without errors.

### Step 3: Test LLM Router with Claude
```python
from src.utils.llm_router import LLMRouter, TaskContext

router = LLMRouter()
context = TaskContext(
    task_type="legal",
    risk_level="high",
    requires_accuracy=True,
    token_estimate=1000,
    priority="high"
)
result = await router.complete("Test prompt", context=context)
print(result)  # Should route to Claude if configured
```

### Step 4: Verify Dashboard
```bash
cd dashboard && npm run dev
```
Navigate to http://localhost:5173 and verify:
- Empire tab loads
- Chat works
- Approvals tab shows queue
- Evolution tab shows proposals

### Step 5: Test Scheduler
```python
from src.services.scheduler import scheduler
await scheduler.start()
print(scheduler.get_status())  # Should show registered tasks
```

---

## Part 5: Priority Order for Implementation

| Priority | Phase | Estimated Time | Impact |
|----------|-------|----------------|--------|
| 1 | PHASE 1: Bug Fixes | 2 hours | Critical - fixes failing tests |
| 2 | PHASE 2: Claude Integration | 4 hours | High - enables high-stakes fallback |
| 3 | PHASE 3: CI/CD Pipeline | 3 hours | High - enables automated deployment |
| 4 | PHASE 4: Scheduler | 3 hours | High - enables true autonomy |
| 5 | PHASE 5: Evolution Dashboard | 2 hours | Medium - UI improvement |
| 6 | PHASE 6: LangSmith | 2 hours | Medium - observability |
| 7 | PHASE 7: Two-Approver | 2 hours | Medium - security enhancement |

**Total Estimated Implementation Time: ~18 hours**

---

## Appendix A: File Creation Summary

| File | Action | Phase |
|------|--------|-------|
| `src/utils/claude_client.py` | CREATE | 2 |
| `.github/workflows/ci.yml` | CREATE | 3 |
| `.github/workflows/terraform.yml` | CREATE | 3 |
| `src/services/scheduler.py` | CREATE | 4 |
| `src/utils/langsmith_client.py` | CREATE | 6 |

## Appendix B: File Modification Summary

| File | Changes | Phase |
|------|---------|-------|
| `src/business/lifecycle.py` | Fix SUNSET handling | 1 |
| `src/business/portfolio.py` | Add get_total_stats | 1 |
| `config/settings.py` | Fix Pydantic deprecations, add vLLM | 1 |
| `src/master_ai/brain.py` | Fix EvolutionEngine init, add scheduler methods | 1, 4 |
| `src/utils/llm_router.py` | Add Claude, LangSmith integration | 2, 6 |
| `src/api/main.py` | Add scheduler startup | 4 |
| `src/api/routes/evolution.py` | Add proposal endpoints | 5 |
| `dashboard/src/App.jsx` | Add evolution tab | 5 |
| `src/approvals/models.py` | Add multi-approver | 7 |
| `src/approvals/manager.py` | Update approval logic | 7 |

---

*Document generated: December 31, 2025*
*Analysis performed on king-ai-v2 codebase*
