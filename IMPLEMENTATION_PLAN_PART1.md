# King AI v2 - Implementation Plan Part 1 of 4
## Infrastructure Layer & Core System Hardening

**Target Timeline:** Week 1-2
**Objective:** Build production-ready infrastructure foundation with AWS deployment, high-availability databases, and enterprise monitoring.

---

## Table of Contents
1. [Current State Analysis](#current-state-analysis)
2. [Gap Analysis](#gap-analysis)
3. [Implementation Tasks](#implementation-tasks)
4. [File-by-File Instructions](#file-by-file-instructions)
5. [Testing Requirements](#testing-requirements)
6. [Acceptance Criteria](#acceptance-criteria)

---

## Current State Analysis

### What Exists
| Component | Status | Location |
|-----------|--------|----------|
| Basic Docker setup | ✅ Exists | `Dockerfile`, `docker-compose.yml` |
| PostgreSQL config | ✅ Basic | `docker-compose.yml` |
| Redis config | ✅ Basic | `docker-compose.yml` |
| Ollama client | ✅ Exists | `src/utils/ollama_client.py` |
| Pinecone vector store | ✅ Basic | `src/database/vector_store.py` |
| Settings management | ✅ Basic | `config/settings.py` |

### What's Missing
| Component | Priority | Specification Requirement |
|-----------|----------|---------------------------|
| AWS Terraform scripts | 🔴 Critical | Multi-AZ VPC with private subnets |
| GPU instance auto-scaling | 🔴 Critical | p5.48xlarge with 2-8 instance scaling |
| vLLM integration | 🔴 Critical | High-throughput inference batching |
| Kong API Gateway | 🟡 High | JWT auth, rate limiting |
| Datadog monitoring | 🟡 High | Metrics, alerts, APM |
| Arize AI integration | 🟡 High | ML model drift detection |
| Circuit breakers | 🟡 High | Fault tolerance patterns |
| Health check endpoints | 🟢 Medium | Comprehensive system health |

---

## Gap Analysis

### Infrastructure Gaps

**1. No AWS Infrastructure as Code**
- Current: Local Docker only
- Required: Terraform scripts for full AWS deployment
- Impact: Cannot deploy to production

**2. No Auto-Scaling for LLM Inference**
- Current: Single Ollama instance
- Required: Auto-scaling GPU cluster (2-8 instances based on queue depth >100)
- Impact: Cannot handle production load

**3. No vLLM for Production Throughput**
- Current: Ollama only (good for dev, limited throughput)
- Required: vLLM integration for batched inference
- Impact: 10-50x slower than required

**4. No API Gateway**
- Current: Direct FastAPI exposure
- Required: Kong with JWT auth, rate limiting
- Impact: Security vulnerability, no rate limiting

**5. No Enterprise Monitoring**
- Current: Basic health endpoint
- Required: Datadog APM, custom metrics, alerts
- Impact: No visibility into production issues

---

## Implementation Tasks

### Task 1.1: Create AWS Terraform Infrastructure
**Priority:** 🔴 Critical
**Estimated Time:** 8 hours
**Dependencies:** None

Create the following directory structure:
```
infrastructure/
├── terraform/
│   ├── main.tf
│   ├── variables.tf
│   ├── outputs.tf
│   ├── vpc.tf
│   ├── ec2.tf
│   ├── rds.tf
│   ├── elasticache.tf
│   ├── alb.tf
│   └── autoscaling.tf
```

#### File: `infrastructure/terraform/variables.tf`
```hcl
variable "aws_region" {
  description = "AWS region for deployment"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "prod"
}

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "availability_zones" {
  description = "List of availability zones"
  type        = list(string)
  default     = ["us-east-1a", "us-east-1b", "us-east-1c"]
}

variable "gpu_instance_type" {
  description = "EC2 instance type for GPU inference"
  type        = string
  default     = "p5.48xlarge"
}

variable "gpu_min_instances" {
  description = "Minimum number of GPU instances"
  type        = number
  default     = 2
}

variable "gpu_max_instances" {
  description = "Maximum number of GPU instances"
  type        = number
  default     = 8
}

variable "db_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.r6g.xlarge"
}

variable "redis_node_type" {
  description = "ElastiCache node type"
  type        = string
  default     = "cache.r6g.large"
}
```

#### File: `infrastructure/terraform/vpc.tf`
```hcl
# VPC
resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name        = "king-ai-${var.environment}-vpc"
    Environment = var.environment
  }
}

# Internet Gateway
resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id

  tags = {
    Name = "king-ai-${var.environment}-igw"
  }
}

# Public Subnets (for ALB)
resource "aws_subnet" "public" {
  count                   = length(var.availability_zones)
  vpc_id                  = aws_vpc.main.id
  cidr_block              = cidrsubnet(var.vpc_cidr, 4, count.index)
  availability_zone       = var.availability_zones[count.index]
  map_public_ip_on_launch = true

  tags = {
    Name = "king-ai-${var.environment}-public-${count.index + 1}"
    Type = "public"
  }
}

# Private Subnets (for EC2, RDS, ElastiCache)
resource "aws_subnet" "private" {
  count             = length(var.availability_zones)
  vpc_id            = aws_vpc.main.id
  cidr_block        = cidrsubnet(var.vpc_cidr, 4, count.index + length(var.availability_zones))
  availability_zone = var.availability_zones[count.index]

  tags = {
    Name = "king-ai-${var.environment}-private-${count.index + 1}"
    Type = "private"
  }
}

# NAT Gateway (one per AZ for HA)
resource "aws_eip" "nat" {
  count  = length(var.availability_zones)
  domain = "vpc"

  tags = {
    Name = "king-ai-${var.environment}-nat-eip-${count.index + 1}"
  }
}

resource "aws_nat_gateway" "main" {
  count         = length(var.availability_zones)
  allocation_id = aws_eip.nat[count.index].id
  subnet_id     = aws_subnet.public[count.index].id

  tags = {
    Name = "king-ai-${var.environment}-nat-${count.index + 1}"
  }
}

# Route Tables
resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }

  tags = {
    Name = "king-ai-${var.environment}-public-rt"
  }
}

resource "aws_route_table" "private" {
  count  = length(var.availability_zones)
  vpc_id = aws_vpc.main.id

  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.main[count.index].id
  }

  tags = {
    Name = "king-ai-${var.environment}-private-rt-${count.index + 1}"
  }
}

# Route Table Associations
resource "aws_route_table_association" "public" {
  count          = length(var.availability_zones)
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table_association" "private" {
  count          = length(var.availability_zones)
  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private[count.index].id
}
```

#### File: `infrastructure/terraform/autoscaling.tf`
```hcl
# Launch Template for GPU Instances
resource "aws_launch_template" "gpu_inference" {
  name_prefix   = "king-ai-gpu-"
  image_id      = data.aws_ami.deep_learning.id
  instance_type = var.gpu_instance_type

  vpc_security_group_ids = [aws_security_group.gpu_inference.id]

  iam_instance_profile {
    name = aws_iam_instance_profile.gpu_inference.name
  }

  user_data = base64encode(<<-EOF
    #!/bin/bash
    set -e
    
    # Install Ollama
    curl -fsSL https://ollama.com/install.sh | sh
    
    # Install vLLM
    pip install vllm
    
    # Pull the model
    ollama pull llama3.1:70b
    
    # Start Ollama service
    systemctl enable ollama
    systemctl start ollama
    
    # Start vLLM server for high-throughput
    python -m vllm.entrypoints.openai.api_server \
      --model meta-llama/Llama-3.1-70B-Instruct \
      --tensor-parallel-size 8 \
      --port 8080 &
    
    # Signal ready to ALB
    curl -X PUT "http://169.254.169.254/latest/api/token" \
      -H "X-aws-ec2-metadata-token-ttl-seconds: 21600"
  EOF
  )

  block_device_mappings {
    device_name = "/dev/sda1"
    ebs {
      volume_size           = 500
      volume_type           = "gp3"
      delete_on_termination = true
      encrypted             = true
    }
  }

  monitoring {
    enabled = true
  }

  tag_specifications {
    resource_type = "instance"
    tags = {
      Name        = "king-ai-gpu-inference"
      Environment = var.environment
      Role        = "llm-inference"
    }
  }
}

# Auto Scaling Group
resource "aws_autoscaling_group" "gpu_inference" {
  name                = "king-ai-gpu-asg"
  vpc_zone_identifier = aws_subnet.private[*].id
  target_group_arns   = [aws_lb_target_group.inference.arn]
  health_check_type   = "ELB"
  
  min_size         = var.gpu_min_instances
  max_size         = var.gpu_max_instances
  desired_capacity = var.gpu_min_instances

  launch_template {
    id      = aws_launch_template.gpu_inference.id
    version = "$Latest"
  }

  tag {
    key                 = "Name"
    value               = "king-ai-gpu-inference"
    propagate_at_launch = true
  }
}

# Scaling Policy - Scale based on SQS queue depth
resource "aws_autoscaling_policy" "scale_up" {
  name                   = "king-ai-scale-up"
  scaling_adjustment     = 1
  adjustment_type        = "ChangeInCapacity"
  cooldown               = 300
  autoscaling_group_name = aws_autoscaling_group.gpu_inference.name
}

resource "aws_autoscaling_policy" "scale_down" {
  name                   = "king-ai-scale-down"
  scaling_adjustment     = -1
  adjustment_type        = "ChangeInCapacity"
  cooldown               = 600
  autoscaling_group_name = aws_autoscaling_group.gpu_inference.name
}

# CloudWatch Alarm for Queue Depth > 100
resource "aws_cloudwatch_metric_alarm" "queue_depth_high" {
  alarm_name          = "king-ai-queue-depth-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "ApproximateNumberOfMessagesVisible"
  namespace           = "AWS/SQS"
  period              = 60
  statistic           = "Average"
  threshold           = 100
  alarm_description   = "Scale up when inference queue depth exceeds 100"
  alarm_actions       = [aws_autoscaling_policy.scale_up.arn]

  dimensions = {
    QueueName = aws_sqs_queue.inference_queue.name
  }
}

resource "aws_cloudwatch_metric_alarm" "queue_depth_low" {
  alarm_name          = "king-ai-queue-depth-low"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = 5
  metric_name         = "ApproximateNumberOfMessagesVisible"
  namespace           = "AWS/SQS"
  period              = 60
  statistic           = "Average"
  threshold           = 10
  alarm_description   = "Scale down when inference queue depth is low"
  alarm_actions       = [aws_autoscaling_policy.scale_down.arn]

  dimensions = {
    QueueName = aws_sqs_queue.inference_queue.name
  }
}

# SQS Queue for inference requests
resource "aws_sqs_queue" "inference_queue" {
  name                       = "king-ai-inference-queue"
  delay_seconds              = 0
  max_message_size           = 262144
  message_retention_seconds  = 86400
  receive_wait_time_seconds  = 10
  visibility_timeout_seconds = 300

  tags = {
    Environment = var.environment
  }
}
```

---

### Task 1.2: Implement vLLM Client for Production Inference
**Priority:** 🔴 Critical
**Estimated Time:** 4 hours
**Dependencies:** Task 1.1

#### File: `src/utils/vllm_client.py` (CREATE NEW FILE)
```python
"""
vLLM Client for high-throughput production inference.
Provides batched inference with OpenAI-compatible API.
"""

import httpx
import asyncio
from typing import AsyncIterator, List, Dict, Any
from dataclasses import dataclass
import json


@dataclass
class InferenceRequest:
    """Represents a single inference request."""
    prompt: str
    max_tokens: int = 4096
    temperature: float = 0.7
    request_id: str = None


class VLLMClient:
    """
    Async client for vLLM's OpenAI-compatible API.
    Supports batched requests for high throughput.
    """
    
    def __init__(
        self,
        base_url: str,
        model: str = "meta-llama/Llama-3.1-70B-Instruct",
        max_concurrent: int = 10
    ):
        """
        Initialize the vLLM client.
        
        Args:
            base_url: vLLM server URL (e.g., http://inference-alb.internal:8080)
            model: Model identifier for the API
            max_concurrent: Maximum concurrent requests
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.client = httpx.AsyncClient(timeout=300.0)
        
    async def complete(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7
    ) -> str:
        """
        Generate a completion using vLLM.
        
        Args:
            prompt: The user prompt
            system: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text response
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        async with self.semaphore:
            response = await self.client.post(
                f"{self.base_url}/v1/chat/completions",
                json={
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stream": False
                }
            )
            response.raise_for_status()
            
        data = response.json()
        return data["choices"][0]["message"]["content"]
    
    async def batch_complete(
        self,
        requests: List[InferenceRequest]
    ) -> List[Dict[str, Any]]:
        """
        Process multiple inference requests concurrently.
        
        Args:
            requests: List of InferenceRequest objects
            
        Returns:
            List of results with request_id and response
        """
        async def process_single(req: InferenceRequest) -> Dict[str, Any]:
            try:
                response = await self.complete(
                    prompt=req.prompt,
                    max_tokens=req.max_tokens,
                    temperature=req.temperature
                )
                return {
                    "request_id": req.request_id,
                    "success": True,
                    "response": response
                }
            except Exception as e:
                return {
                    "request_id": req.request_id,
                    "success": False,
                    "error": str(e)
                }
        
        tasks = [process_single(req) for req in requests]
        return await asyncio.gather(*tasks)
    
    async def complete_stream(
        self,
        prompt: str,
        system: str | None = None
    ) -> AsyncIterator[str]:
        """
        Stream a completion token by token.
        
        Args:
            prompt: The user prompt
            system: Optional system prompt
            
        Yields:
            Generated tokens as they arrive
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        async with self.client.stream(
            "POST",
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": messages,
                "max_tokens": 4096,
                "stream": True
            }
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        delta = chunk["choices"][0].get("delta", {})
                        if "content" in delta:
                            yield delta["content"]
                    except json.JSONDecodeError:
                        continue
    
    async def health_check(self) -> bool:
        """Check if vLLM server is healthy."""
        try:
            response = await self.client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except Exception:
            return False
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        response = await self.client.get(f"{self.base_url}/v1/models")
        response.raise_for_status()
        return response.json()
```

---

### Task 1.3: Implement Hybrid LLM Router with Fallback
**Priority:** 🔴 Critical
**Estimated Time:** 4 hours
**Dependencies:** Task 1.2

#### File: `src/utils/llm_router.py` (CREATE NEW FILE)
```python
"""
LLM Router - Intelligent routing between inference providers.
Implements hybrid routing with fallback for reliability.
"""

import asyncio
from enum import Enum
from typing import Optional, Callable, Any
from dataclasses import dataclass
import time

from src.utils.ollama_client import OllamaClient
from src.utils.vllm_client import VLLMClient
from src.utils.gemini_client import GeminiClient
from config.settings import settings


class ProviderType(Enum):
    """Available LLM providers."""
    VLLM = "vllm"           # High-throughput production
    OLLAMA = "ollama"       # Development / fallback
    GEMINI = "gemini"       # Cloud fallback
    CLAUDE = "claude"       # High-stakes fallback (future)


@dataclass
class RoutingDecision:
    """Result of routing decision."""
    provider: ProviderType
    reason: str
    latency_ms: float = 0


@dataclass
class TaskContext:
    """Context for routing decisions."""
    task_type: str          # "research", "finance", "legal", etc.
    risk_level: str         # "low", "medium", "high"
    requires_accuracy: bool # High-stakes decision?
    token_estimate: int     # Estimated tokens needed
    priority: str           # "normal", "high", "critical"


class LLMRouter:
    """
    Routes inference requests to the optimal provider based on:
    - Task risk level (high-stakes -> more accurate provider)
    - Provider health status
    - Current load and latency
    - Cost optimization
    """
    
    def __init__(self):
        """Initialize all available providers."""
        # Primary: vLLM for production throughput
        self.vllm: Optional[VLLMClient] = None
        if hasattr(settings, 'vllm_url') and settings.vllm_url:
            self.vllm = VLLMClient(
                base_url=settings.vllm_url,
                model=settings.vllm_model if hasattr(settings, 'vllm_model') else "meta-llama/Llama-3.1-70B-Instruct"
            )
        
        # Secondary: Ollama for dev/fallback
        self.ollama = OllamaClient(
            base_url=settings.ollama_url,
            model=settings.ollama_model
        )
        
        # Tertiary: Gemini for cloud fallback
        self.gemini: Optional[GeminiClient] = None
        import os
        if os.getenv("GEMINI_API_KEY"):
            self.gemini = GeminiClient(api_key=os.getenv("GEMINI_API_KEY"))
        
        # Health tracking
        self._provider_health = {
            ProviderType.VLLM: True,
            ProviderType.OLLAMA: True,
            ProviderType.GEMINI: True,
        }
        self._last_health_check = 0
        self._health_check_interval = 30  # seconds
        
        # Circuit breaker state
        self._failure_counts = {p: 0 for p in ProviderType}
        self._circuit_open = {p: False for p in ProviderType}
        self._circuit_open_until = {p: 0 for p in ProviderType}
        self._failure_threshold = 3
        self._circuit_timeout = 60  # seconds
    
    async def complete(
        self,
        prompt: str,
        system: str | None = None,
        context: TaskContext | None = None
    ) -> str:
        """
        Route and execute an inference request.
        
        Args:
            prompt: The user prompt
            system: Optional system prompt
            context: Task context for routing decisions
            
        Returns:
            Generated response from selected provider
        """
        # Determine routing
        decision = await self._route(context)
        
        # Execute with fallback chain
        providers = self._get_fallback_chain(decision.provider)
        
        last_error = None
        for provider in providers:
            if self._is_circuit_open(provider):
                continue
                
            try:
                start = time.time()
                result = await self._execute(provider, prompt, system)
                latency = (time.time() - start) * 1000
                
                # Record success
                self._record_success(provider)
                
                return result
                
            except Exception as e:
                last_error = e
                self._record_failure(provider)
                continue
        
        raise RuntimeError(f"All providers failed. Last error: {last_error}")
    
    async def _route(self, context: TaskContext | None) -> RoutingDecision:
        """Determine the best provider for this request."""
        
        # Default to vLLM if available and healthy
        if self.vllm and self._provider_health[ProviderType.VLLM]:
            # High-stakes tasks might need more accurate provider
            if context and context.risk_level == "high" and context.requires_accuracy:
                if self.gemini and self._provider_health[ProviderType.GEMINI]:
                    return RoutingDecision(
                        provider=ProviderType.GEMINI,
                        reason="High-stakes task routed to Gemini for accuracy"
                    )
            
            return RoutingDecision(
                provider=ProviderType.VLLM,
                reason="Primary production provider"
            )
        
        # Fallback to Ollama
        if self._provider_health[ProviderType.OLLAMA]:
            return RoutingDecision(
                provider=ProviderType.OLLAMA,
                reason="Fallback to Ollama (vLLM unavailable)"
            )
        
        # Last resort: Gemini
        if self.gemini and self._provider_health[ProviderType.GEMINI]:
            return RoutingDecision(
                provider=ProviderType.GEMINI,
                reason="Cloud fallback (all local providers down)"
            )
        
        raise RuntimeError("No healthy providers available")
    
    def _get_fallback_chain(self, primary: ProviderType) -> list[ProviderType]:
        """Get ordered fallback chain starting from primary."""
        chain = [primary]
        
        all_providers = [ProviderType.VLLM, ProviderType.OLLAMA, ProviderType.GEMINI]
        for p in all_providers:
            if p not in chain:
                chain.append(p)
        
        return chain
    
    async def _execute(
        self,
        provider: ProviderType,
        prompt: str,
        system: str | None
    ) -> str:
        """Execute inference on specific provider."""
        if provider == ProviderType.VLLM and self.vllm:
            return await self.vllm.complete(prompt, system)
        elif provider == ProviderType.OLLAMA:
            return await self.ollama.complete(prompt, system)
        elif provider == ProviderType.GEMINI and self.gemini:
            return await self.gemini.complete(prompt, system)
        else:
            raise ValueError(f"Provider {provider} not available")
    
    def _is_circuit_open(self, provider: ProviderType) -> bool:
        """Check if circuit breaker is open for provider."""
        if not self._circuit_open[provider]:
            return False
        
        # Check if timeout has passed
        if time.time() > self._circuit_open_until[provider]:
            self._circuit_open[provider] = False
            self._failure_counts[provider] = 0
            return False
        
        return True
    
    def _record_success(self, provider: ProviderType):
        """Record successful request."""
        self._failure_counts[provider] = 0
        self._circuit_open[provider] = False
        self._provider_health[provider] = True
    
    def _record_failure(self, provider: ProviderType):
        """Record failed request and potentially open circuit."""
        self._failure_counts[provider] += 1
        
        if self._failure_counts[provider] >= self._failure_threshold:
            self._circuit_open[provider] = True
            self._circuit_open_until[provider] = time.time() + self._circuit_timeout
            self._provider_health[provider] = False
    
    async def health_check_all(self) -> dict[ProviderType, bool]:
        """Check health of all providers."""
        results = {}
        
        if self.vllm:
            results[ProviderType.VLLM] = await self.vllm.health_check()
        else:
            results[ProviderType.VLLM] = False
            
        results[ProviderType.OLLAMA] = await self.ollama.health_check()
        
        if self.gemini:
            try:
                # Simple test
                await self.gemini.complete("test", None)
                results[ProviderType.GEMINI] = True
            except:
                results[ProviderType.GEMINI] = False
        else:
            results[ProviderType.GEMINI] = False
        
        self._provider_health = results
        return results
```

---

### Task 1.4: Add Datadog Monitoring Integration
**Priority:** 🟡 High
**Estimated Time:** 4 hours
**Dependencies:** None

#### File: `src/utils/monitoring.py` (CREATE NEW FILE)
```python
"""
Datadog monitoring integration.
Provides metrics, APM, and alerting capabilities.
"""

import os
import time
import functools
from typing import Callable, Any, Dict, Optional
from contextlib import contextmanager
from dataclasses import dataclass

# Conditional import for Datadog
try:
    from ddtrace import tracer, patch_all
    from datadog import initialize, statsd
    DATADOG_AVAILABLE = True
except ImportError:
    DATADOG_AVAILABLE = False
    tracer = None
    statsd = None


@dataclass
class MetricTags:
    """Standard tags for metrics."""
    environment: str
    service: str = "king-ai"
    version: str = "2.0.0"


class DatadogMonitor:
    """
    Centralized Datadog monitoring client.
    Provides metrics, tracing, and custom events.
    """
    
    def __init__(self):
        """Initialize Datadog if API key is available."""
        self.enabled = False
        self.tags = MetricTags(
            environment=os.getenv("ENVIRONMENT", "development")
        )
        
        dd_api_key = os.getenv("DD_API_KEY")
        if DATADOG_AVAILABLE and dd_api_key:
            initialize(
                api_key=dd_api_key,
                app_key=os.getenv("DD_APP_KEY"),
            )
            patch_all()  # Auto-instrument common libraries
            self.enabled = True
    
    def _get_tags(self, extra_tags: Dict[str, str] = None) -> list[str]:
        """Build tag list for metrics."""
        tags = [
            f"env:{self.tags.environment}",
            f"service:{self.tags.service}",
            f"version:{self.tags.version}",
        ]
        if extra_tags:
            tags.extend([f"{k}:{v}" for k, v in extra_tags.items()])
        return tags
    
    def increment(
        self,
        metric: str,
        value: int = 1,
        tags: Dict[str, str] = None
    ):
        """Increment a counter metric."""
        if self.enabled and statsd:
            statsd.increment(
                f"king_ai.{metric}",
                value=value,
                tags=self._get_tags(tags)
            )
    
    def gauge(
        self,
        metric: str,
        value: float,
        tags: Dict[str, str] = None
    ):
        """Set a gauge metric."""
        if self.enabled and statsd:
            statsd.gauge(
                f"king_ai.{metric}",
                value=value,
                tags=self._get_tags(tags)
            )
    
    def histogram(
        self,
        metric: str,
        value: float,
        tags: Dict[str, str] = None
    ):
        """Record a histogram value (timing, size, etc.)."""
        if self.enabled and statsd:
            statsd.histogram(
                f"king_ai.{metric}",
                value=value,
                tags=self._get_tags(tags)
            )
    
    def timing(
        self,
        metric: str,
        value_ms: float,
        tags: Dict[str, str] = None
    ):
        """Record a timing metric in milliseconds."""
        self.histogram(metric, value_ms, tags)
    
    @contextmanager
    def timed(self, metric: str, tags: Dict[str, str] = None):
        """Context manager for timing a block of code."""
        start = time.time()
        try:
            yield
        finally:
            elapsed_ms = (time.time() - start) * 1000
            self.timing(metric, elapsed_ms, tags)
    
    def trace(
        self,
        operation_name: str,
        service: str = None,
        resource: str = None
    ):
        """Decorator for tracing a function."""
        def decorator(func: Callable) -> Callable:
            if not self.enabled or not tracer:
                return func
            
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                with tracer.trace(
                    operation_name,
                    service=service or self.tags.service,
                    resource=resource or func.__name__
                ):
                    return await func(*args, **kwargs)
            
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                with tracer.trace(
                    operation_name,
                    service=service or self.tags.service,
                    resource=resource or func.__name__
                ):
                    return func(*args, **kwargs)
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            return sync_wrapper
        
        return decorator
    
    def event(
        self,
        title: str,
        text: str,
        alert_type: str = "info",
        tags: Dict[str, str] = None
    ):
        """Send a custom event to Datadog."""
        if self.enabled and statsd:
            statsd.event(
                title=title,
                text=text,
                alert_type=alert_type,
                tags=self._get_tags(tags)
            )


# Import asyncio for the decorator
import asyncio

# Global monitor instance
monitor = DatadogMonitor()


# Convenience decorators
def trace_llm(func: Callable) -> Callable:
    """Trace LLM inference calls."""
    return monitor.trace("llm.inference", resource=func.__name__)(func)


def trace_agent(agent_name: str):
    """Trace agent execution."""
    def decorator(func: Callable) -> Callable:
        return monitor.trace(
            "agent.execute",
            resource=agent_name
        )(func)
    return decorator


def trace_db(func: Callable) -> Callable:
    """Trace database operations."""
    return monitor.trace("db.query", resource=func.__name__)(func)
```

---

### Task 1.5: Update Settings for New Infrastructure
**Priority:** 🔴 Critical
**Estimated Time:** 2 hours
**Dependencies:** Tasks 1.2, 1.3, 1.4

#### File: `config/settings.py` (MODIFY EXISTING FILE)
Replace the entire contents with:

```python
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Literal, Optional

class Settings(BaseSettings):
    """
    Main configuration class for King AI v2.
    Loads values from environment variables or a .env file.
    """
    
    # --- Database Settings ---
    database_url: str = Field(..., env="DATABASE_URL")
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    
    # --- LLM Provider Settings ---
    # Primary: vLLM for production
    vllm_url: Optional[str] = Field(default=None, env="VLLM_URL")
    vllm_model: str = Field(
        default="meta-llama/Llama-3.1-70B-Instruct",
        env="VLLM_MODEL"
    )
    
    # Secondary: Ollama for development/fallback
    ollama_url: str = Field(default="http://localhost:11434", env="OLLAMA_URL")
    ollama_model: str = Field(default="llama3.1:8b", env="OLLAMA_MODEL")
    
    # Tertiary: Cloud fallbacks (API keys via env)
    # GEMINI_API_KEY loaded directly from env
    # ANTHROPIC_API_KEY for future Claude integration
    
    # --- Vector Store Settings ---
    pinecone_api_key: Optional[str] = Field(default=None, env="PINECONE_API_KEY")
    pinecone_index: str = Field(default="king-ai", env="PINECONE_INDEX")
    pinecone_environment: str = Field(default="us-east-1", env="PINECONE_ENV")
    
    # --- Risk & Evolution Controls ---
    risk_profile: Literal["conservative", "moderate", "aggressive"] = Field(
        default="moderate",
        env="RISK_PROFILE"
    )
    max_evolutions_per_hour: int = Field(default=5, env="MAX_EVOLUTIONS_PER_HOUR")
    evolution_confidence_threshold: float = Field(
        default=0.8,
        env="EVOLUTION_CONFIDENCE_THRESHOLD"
    )
    
    # --- API Server Settings ---
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_workers: int = Field(default=4, env="API_WORKERS")
    
    # --- AWS Settings ---
    aws_region: str = Field(default="us-east-1", env="AWS_REGION")
    sqs_inference_queue: Optional[str] = Field(default=None, env="SQS_INFERENCE_QUEUE")
    s3_artifacts_bucket: Optional[str] = Field(default=None, env="S3_ARTIFACTS_BUCKET")
    
    # --- Monitoring Settings ---
    datadog_enabled: bool = Field(default=False, env="DD_ENABLED")
    datadog_api_key: Optional[str] = Field(default=None, env="DD_API_KEY")
    datadog_app_key: Optional[str] = Field(default=None, env="DD_APP_KEY")
    
    # --- Security Settings ---
    jwt_secret: str = Field(default="change-me-in-production", env="JWT_SECRET")
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    api_rate_limit: int = Field(default=100, env="API_RATE_LIMIT")  # requests per minute
    
    # --- Feature Flags ---
    enable_autonomous_mode: bool = Field(default=False, env="ENABLE_AUTONOMOUS_MODE")
    enable_self_modification: bool = Field(default=True, env="ENABLE_SELF_MODIFICATION")
    enable_vllm: bool = Field(default=False, env="ENABLE_VLLM")
    
    class Config:
        env_file = ".env"
        extra = "ignore"


# Singleton instance
settings = Settings()
```

---

### Task 1.6: Create Comprehensive Health Check Endpoint
**Priority:** 🟢 Medium
**Estimated Time:** 2 hours
**Dependencies:** Tasks 1.3, 1.4

#### File: `src/api/routes/health.py` (CREATE NEW FILE)
```python
"""
Health Check Routes - Comprehensive system health monitoring.
"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict, Any, Optional
from datetime import datetime
import asyncio

from config.settings import settings
from src.utils.llm_router import LLMRouter, ProviderType
from src.database.connection import get_db
from sqlalchemy import text

router = APIRouter()


class ProviderHealth(BaseModel):
    """Health status for a single provider."""
    name: str
    healthy: bool
    latency_ms: Optional[float] = None
    error: Optional[str] = None


class ComponentHealth(BaseModel):
    """Health status for a system component."""
    name: str
    healthy: bool
    latency_ms: Optional[float] = None
    details: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class SystemHealth(BaseModel):
    """Complete system health report."""
    status: str  # "healthy", "degraded", "unhealthy"
    timestamp: str
    version: str
    environment: str
    components: Dict[str, ComponentHealth]
    llm_providers: Dict[str, ProviderHealth]


@router.get("/", response_model=SystemHealth)
async def full_health_check():
    """
    Comprehensive health check for all system components.
    Returns detailed status for monitoring and alerting.
    """
    import time
    
    components = {}
    llm_providers = {}
    
    # Check Database
    try:
        start = time.time()
        async with get_db() as db:
            await db.execute(text("SELECT 1"))
        latency = (time.time() - start) * 1000
        components["database"] = ComponentHealth(
            name="PostgreSQL",
            healthy=True,
            latency_ms=latency
        )
    except Exception as e:
        components["database"] = ComponentHealth(
            name="PostgreSQL",
            healthy=False,
            error=str(e)
        )
    
    # Check Redis
    try:
        import redis.asyncio as redis
        start = time.time()
        r = redis.from_url(settings.redis_url)
        await r.ping()
        latency = (time.time() - start) * 1000
        components["redis"] = ComponentHealth(
            name="Redis",
            healthy=True,
            latency_ms=latency
        )
    except Exception as e:
        components["redis"] = ComponentHealth(
            name="Redis",
            healthy=False,
            error=str(e)
        )
    
    # Check LLM Providers
    llm_router = LLMRouter()
    provider_health = await llm_router.health_check_all()
    
    for provider, healthy in provider_health.items():
        llm_providers[provider.value] = ProviderHealth(
            name=provider.value,
            healthy=healthy
        )
    
    # Check Vector Store
    try:
        from src.database.vector_store import VectorStore
        vs = VectorStore()
        if vs.index:
            components["vector_store"] = ComponentHealth(
                name="Pinecone",
                healthy=True,
                details={"index": settings.pinecone_index}
            )
        else:
            components["vector_store"] = ComponentHealth(
                name="Pinecone",
                healthy=False,
                error="Not configured"
            )
    except Exception as e:
        components["vector_store"] = ComponentHealth(
            name="Pinecone",
            healthy=False,
            error=str(e)
        )
    
    # Determine overall status
    all_healthy = all(c.healthy for c in components.values())
    any_llm_healthy = any(p.healthy for p in llm_providers.values())
    
    if all_healthy and any_llm_healthy:
        status = "healthy"
    elif components.get("database", ComponentHealth(name="db", healthy=False)).healthy and any_llm_healthy:
        status = "degraded"
    else:
        status = "unhealthy"
    
    return SystemHealth(
        status=status,
        timestamp=datetime.now().isoformat(),
        version="2.0.0",
        environment=settings.risk_profile,
        components=components,
        llm_providers=llm_providers
    )


@router.get("/ready")
async def readiness_check():
    """
    Kubernetes readiness probe.
    Returns 200 if the service can accept traffic.
    """
    try:
        async with get_db() as db:
            await db.execute(text("SELECT 1"))
        return {"ready": True}
    except:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="Service not ready")


@router.get("/live")
async def liveness_check():
    """
    Kubernetes liveness probe.
    Returns 200 if the service is alive.
    """
    return {"alive": True}
```

---

## Testing Requirements

### Unit Tests to Create

#### File: `tests/test_infrastructure.py` (CREATE NEW FILE)
```python
"""
Infrastructure component tests.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from src.utils.vllm_client import VLLMClient, InferenceRequest
from src.utils.llm_router import LLMRouter, ProviderType, TaskContext
from src.utils.monitoring import DatadogMonitor


class TestVLLMClient:
    """Tests for vLLM client."""
    
    @pytest.fixture
    def client(self):
        return VLLMClient(base_url="http://localhost:8080", model="test-model")
    
    @pytest.mark.asyncio
    async def test_complete_success(self, client):
        """Test successful completion."""
        with patch.object(client.client, 'post', new_callable=AsyncMock) as mock_post:
            mock_post.return_value.json.return_value = {
                "choices": [{"message": {"content": "Hello!"}}]
            }
            mock_post.return_value.raise_for_status = MagicMock()
            
            result = await client.complete("Hello")
            assert result == "Hello!"
    
    @pytest.mark.asyncio
    async def test_batch_complete(self, client):
        """Test batch completion."""
        requests = [
            InferenceRequest(prompt="Test 1", request_id="1"),
            InferenceRequest(prompt="Test 2", request_id="2"),
        ]
        
        with patch.object(client, 'complete', new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = "Response"
            
            results = await client.batch_complete(requests)
            assert len(results) == 2
            assert all(r["success"] for r in results)


class TestLLMRouter:
    """Tests for LLM routing logic."""
    
    @pytest.fixture
    def router(self):
        with patch('src.utils.llm_router.settings') as mock_settings:
            mock_settings.ollama_url = "http://localhost:11434"
            mock_settings.ollama_model = "llama3.1:8b"
            mock_settings.vllm_url = None
            return LLMRouter()
    
    @pytest.mark.asyncio
    async def test_fallback_chain(self, router):
        """Test fallback when primary fails."""
        router._provider_health[ProviderType.VLLM] = False
        
        with patch.object(router.ollama, 'complete', new_callable=AsyncMock) as mock_ollama:
            mock_ollama.return_value = "Ollama response"
            
            result = await router.complete("Test")
            assert result == "Ollama response"
    
    def test_circuit_breaker(self, router):
        """Test circuit breaker opens after failures."""
        provider = ProviderType.OLLAMA
        
        for _ in range(router._failure_threshold):
            router._record_failure(provider)
        
        assert router._is_circuit_open(provider)


class TestDatadogMonitor:
    """Tests for monitoring integration."""
    
    def test_tags_format(self):
        """Test metric tags are properly formatted."""
        monitor = DatadogMonitor()
        tags = monitor._get_tags({"custom": "tag"})
        
        assert any("env:" in t for t in tags)
        assert any("service:" in t for t in tags)
        assert "custom:tag" in tags
```

---

## Acceptance Criteria

### Part 1 Completion Checklist

- [ ] **Terraform Infrastructure**
  - [ ] VPC with multi-AZ subnets created
  - [ ] Auto-scaling group for GPU instances configured
  - [ ] RDS PostgreSQL provisioned
  - [ ] ElastiCache Redis provisioned
  - [ ] ALB with health checks configured
  - [ ] SQS queue for inference requests created

- [ ] **vLLM Integration**
  - [ ] VLLMClient class implemented and tested
  - [ ] Batch inference support working
  - [ ] Streaming support implemented
  - [ ] Health check endpoint working

- [ ] **LLM Router**
  - [ ] Multi-provider routing implemented
  - [ ] Circuit breaker pattern working
  - [ ] Fallback chain tested
  - [ ] High-stakes routing to accurate provider working

- [ ] **Monitoring**
  - [ ] Datadog integration configured
  - [ ] Custom metrics being recorded
  - [ ] APM tracing working
  - [ ] Alerts configured in Datadog

- [ ] **Health Checks**
  - [ ] Comprehensive health endpoint working
  - [ ] All components checked (DB, Redis, LLM, Vector store)
  - [ ] Kubernetes probes implemented

- [ ] **Settings Updated**
  - [ ] All new configuration options added
  - [ ] Environment variables documented
  - [ ] Feature flags implemented

---

## Next Steps

After completing Part 1, proceed to:
- **Part 2:** Master AI Layer Enhancement (Self-modification, ML retraining)
- **Part 3:** Sub-Agents & Tools Layer (External integrations, API connections)
- **Part 4:** Dashboard & Human Oversight (React UI, Approval workflows)

---

## Environment Variables Template

Create/update `.env.example`:
```bash
# Database
DATABASE_URL=postgresql+asyncpg://king:password@localhost:5432/kingai
REDIS_URL=redis://localhost:6379

# LLM Providers
VLLM_URL=http://inference-alb.internal:8080
VLLM_MODEL=meta-llama/Llama-3.1-70B-Instruct
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
GEMINI_API_KEY=your-gemini-key

# Vector Store
PINECONE_API_KEY=your-pinecone-key
PINECONE_INDEX=king-ai
PINECONE_ENV=us-east-1

# AWS
AWS_REGION=us-east-1
SQS_INFERENCE_QUEUE=king-ai-inference-queue
S3_ARTIFACTS_BUCKET=king-ai-artifacts

# Monitoring
DD_ENABLED=true
DD_API_KEY=your-datadog-key
DD_APP_KEY=your-datadog-app-key

# Security
JWT_SECRET=your-super-secret-key-change-in-production
API_RATE_LIMIT=100

# Feature Flags
ENABLE_AUTONOMOUS_MODE=false
ENABLE_SELF_MODIFICATION=true
ENABLE_VLLM=true
RISK_PROFILE=moderate
```

---

*End of Part 1 - Infrastructure Layer & Core System Hardening*
