# King AI v2 Implementation Plan - Part 1 of 5

## Executive Summary

This implementation plan addresses gaps, bugs, and missing features identified in the King AI v2 codebase. The plan is divided into 5 parts:

| Part | Focus Area | Estimated Effort |
|------|------------|------------------|
| **Part 1** | Infrastructure Layer Fixes | 2-3 days |
| **Part 2** | Master AI & Evolution Engine Fixes | 3-4 days |
| **Part 3** | Sub-Agents & Router Enhancements | 2-3 days |
| **Part 4** | Business Units & Playbook Enhancements | 2-3 days |
| **Part 5** | Human Oversight, Risk & Dashboard Fixes | 2-3 days |

---

## Part 1: Infrastructure Layer Fixes

### Overview

The infrastructure layer is ~85% complete but has critical bugs that will cause Terraform failures and security vulnerabilities. This part focuses on fixing those issues.

---

## Task 1.1: Add Missing Terraform Variable `gpu_max_instances`

**Priority:** 🔴 CRITICAL  
**File:** `infrastructure/terraform/variables.tf`  
**Issue:** Variable `gpu_max_instances` is referenced in `autoscaling.tf` but not defined

### Instructions

1. Open `infrastructure/terraform/variables.tf`
2. Add the following variable definition after the `gpu_min_instances` variable (around line 40):

```hcl
variable "gpu_max_instances" {
  description = "Maximum number of GPU instances for auto-scaling"
  type        = number
  default     = 8
}
```

### Verification
```bash
cd infrastructure/terraform
terraform validate
```

---

## Task 1.2: Remove Duplicate `data.aws_ami.deep_learning` Resource

**Priority:** 🔴 CRITICAL  
**Files:** 
- `infrastructure/terraform/ec2.tf`
- `infrastructure/terraform/gpu_instances.tf`

**Issue:** Same data source defined in both files, causing Terraform error

### Instructions

1. Open `infrastructure/terraform/ec2.tf`
2. Find and DELETE the following block (approximately lines 1-10):

```hcl
# Deep Learning AMI for GPU instances
data "aws_ami" "deep_learning" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["Deep Learning AMI (Ubuntu 22.04) *"]
  }
}
```

3. Keep the data source ONLY in `gpu_instances.tf`

### Verification
```bash
cd infrastructure/terraform
terraform validate
```

---

## Task 1.3: Create Missing Ollama Setup Script

**Priority:** 🔴 CRITICAL  
**File to Create:** `infrastructure/terraform/scripts/ollama_setup.sh`  
**Issue:** Referenced in `gpu_instances.tf` but file doesn't exist

### Instructions

1. Create directory: `infrastructure/terraform/scripts/`
2. Create file `ollama_setup.sh` with the following content:

```bash
#!/bin/bash
set -e

# Ollama and vLLM Setup Script for King AI v2
# This script runs on GPU instance boot to configure the inference stack

LOG_FILE="/var/log/ollama_setup.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== Starting Ollama Setup $(date) ==="

# Update system
apt-get update -y
apt-get upgrade -y

# Install dependencies
apt-get install -y curl wget git python3-pip python3-venv nvidia-cuda-toolkit

# Verify NVIDIA drivers
echo "Checking NVIDIA drivers..."
nvidia-smi || {
    echo "ERROR: NVIDIA drivers not detected"
    exit 1
}

# Install Ollama
echo "Installing Ollama..."
curl -fsSL https://ollama.com/install.sh | sh

# Configure Ollama to listen on all interfaces (for internal VPC access)
mkdir -p /etc/systemd/system/ollama.service.d
cat > /etc/systemd/system/ollama.service.d/override.conf << EOF
[Service]
Environment="OLLAMA_HOST=0.0.0.0:11434"
Environment="OLLAMA_ORIGINS=*"
EOF

# Start Ollama service
systemctl daemon-reload
systemctl enable ollama
systemctl start ollama

# Wait for Ollama to be ready
echo "Waiting for Ollama to start..."
sleep 10

# Pull the configured model
OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.1:70b}"
echo "Pulling model: $OLLAMA_MODEL"
ollama pull "$OLLAMA_MODEL"

# Install vLLM for high-throughput batching
echo "Installing vLLM..."
pip3 install vllm

# Create vLLM service
VLLM_MODEL="${VLLM_MODEL:-meta-llama/Meta-Llama-3.1-70B-Instruct}"
cat > /etc/systemd/system/vllm.service << EOF
[Unit]
Description=vLLM Inference Server
After=network.target

[Service]
Type=simple
User=root
Environment="HF_TOKEN=${HF_TOKEN}"
ExecStart=/usr/local/bin/python3 -m vllm.entrypoints.openai.api_server \
    --model $VLLM_MODEL \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 8 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.9
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable vllm
systemctl start vllm

# Health check endpoint
cat > /usr/local/bin/health_check.sh << 'EOF'
#!/bin/bash
# Check Ollama
curl -s http://localhost:11434/api/tags > /dev/null && echo "Ollama: OK" || echo "Ollama: FAILED"
# Check vLLM
curl -s http://localhost:8000/health > /dev/null && echo "vLLM: OK" || echo "vLLM: FAILED"
EOF
chmod +x /usr/local/bin/health_check.sh

echo "=== Ollama Setup Complete $(date) ==="
```

3. Make the script executable in your local environment (for version control):
```bash
chmod +x infrastructure/terraform/scripts/ollama_setup.sh
```

### Verification
- File exists at `infrastructure/terraform/scripts/ollama_setup.sh`
- Run `terraform validate` - should pass

---

## Task 1.4: Fix ALB to Be Internal (Security)

**Priority:** 🟡 HIGH  
**File:** `infrastructure/terraform/alb.tf`  
**Issue:** ALB is public (`internal = false`) but requirements specify internal API access

### Instructions

1. Open `infrastructure/terraform/alb.tf`
2. Find the `aws_lb` resource (around line 1-15)
3. Change `internal = false` to `internal = true`

**Before:**
```hcl
resource "aws_lb" "main" {
  name               = "king-ai-${var.environment}-alb"
  internal           = false
  load_balancer_type = "application"
```

**After:**
```hcl
resource "aws_lb" "main" {
  name               = "king-ai-${var.environment}-alb"
  internal           = true
  load_balancer_type = "application"
```

### Verification
```bash
terraform plan | grep -A5 "aws_lb.main"
# Should show internal = true
```

---

## Task 1.5: Restrict SSH Access (Security Vulnerability)

**Priority:** 🔴 CRITICAL - SECURITY  
**File:** `infrastructure/terraform/ec2.tf`  
**Issue:** SSH port 22 open to `0.0.0.0/0` (entire internet)

### Instructions

1. Open `infrastructure/terraform/ec2.tf`
2. Find the security group ingress rule for port 22
3. Replace `0.0.0.0/0` with a VPN or bastion CIDR

**Before:**
```hcl
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "SSH access"
  }
```

**After:**
```hcl
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.vpn_cidr]  # Restrict to VPN only
    description = "SSH access from VPN"
  }
```

4. Add the VPN CIDR variable to `variables.tf`:

```hcl
variable "vpn_cidr" {
  description = "CIDR block for VPN access (SSH, admin)"
  type        = string
  default     = "10.100.0.0/16"  # Replace with actual VPN CIDR
}
```

---

## Task 1.6: Fix Model Version Inconsistency

**Priority:** 🟡 MEDIUM  
**Files:** 
- `infrastructure/terraform/variables.tf`
- `config/settings.py`

**Issue:** Terraform defaults to `llama2:13b` but settings.py uses `llama3.1:70b`

### Instructions

1. Open `infrastructure/terraform/variables.tf`
2. Update the default model versions:

**Before:**
```hcl
variable "ollama_model" {
  description = "Ollama model to deploy"
  type        = string
  default     = "llama2:13b"
}

variable "vllm_model" {
  description = "vLLM model to deploy"
  type        = string
  default     = "meta-llama/Llama-2-13b-hf"
}
```

**After:**
```hcl
variable "ollama_model" {
  description = "Ollama model to deploy"
  type        = string
  default     = "llama3.1:70b"
}

variable "vllm_model" {
  description = "vLLM model to deploy"
  type        = string
  default     = "meta-llama/Meta-Llama-3.1-70B-Instruct"
}
```

---

## Task 1.7: Enable HTTPS on ALB

**Priority:** 🟡 HIGH  
**File:** `infrastructure/terraform/alb.tf`  
**Issue:** HTTPS listener is commented out

### Instructions

1. Open `infrastructure/terraform/alb.tf`
2. Uncomment and configure the HTTPS listener
3. Add ACM certificate variable

Add to `variables.tf`:
```hcl
variable "acm_certificate_arn" {
  description = "ARN of ACM certificate for HTTPS"
  type        = string
  default     = ""  # Set in terraform.tfvars
}
```

Add to `alb.tf` after the HTTP listener:
```hcl
resource "aws_lb_listener" "https" {
  count             = var.acm_certificate_arn != "" ? 1 : 0
  load_balancer_arn = aws_lb.main.arn
  port              = "443"
  protocol          = "HTTPS"
  ssl_policy        = "ELBSecurityPolicy-TLS13-1-2-2021-06"
  certificate_arn   = var.acm_certificate_arn

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.main.arn
  }
}

# Redirect HTTP to HTTPS when certificate is configured
resource "aws_lb_listener_rule" "redirect_http_to_https" {
  count        = var.acm_certificate_arn != "" ? 1 : 0
  listener_arn = aws_lb_listener.http.arn
  priority     = 1

  action {
    type = "redirect"
    redirect {
      port        = "443"
      protocol    = "HTTPS"
      status_code = "HTTP_301"
    }
  }

  condition {
    path_pattern {
      values = ["/*"]
    }
  }
}
```

---

## Task 1.8: Add Kong API Gateway (Missing Component)

**Priority:** 🔴 CRITICAL  
**Files to Create:** `infrastructure/terraform/kong.tf`  
**Issue:** Kong API Gateway with JWT auth is completely missing

### Instructions

1. Create file `infrastructure/terraform/kong.tf`:

```hcl
# Kong API Gateway for King AI v2
# Provides API gateway with JWT authentication

# Kong ECS Task Definition
resource "aws_ecs_task_definition" "kong" {
  family                   = "king-ai-${var.environment}-kong"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = 1024
  memory                   = 2048
  execution_role_arn       = aws_iam_role.ecs_execution.arn
  task_role_arn           = aws_iam_role.ecs_task.arn

  container_definitions = jsonencode([
    {
      name  = "kong"
      image = "kong:3.4"
      
      environment = [
        {
          name  = "KONG_DATABASE"
          value = "off"
        },
        {
          name  = "KONG_PROXY_ACCESS_LOG"
          value = "/dev/stdout"
        },
        {
          name  = "KONG_ADMIN_ACCESS_LOG"
          value = "/dev/stdout"
        },
        {
          name  = "KONG_PROXY_ERROR_LOG"
          value = "/dev/stderr"
        },
        {
          name  = "KONG_ADMIN_ERROR_LOG"
          value = "/dev/stderr"
        },
        {
          name  = "KONG_ADMIN_LISTEN"
          value = "0.0.0.0:8001"
        },
        {
          name  = "KONG_DECLARATIVE_CONFIG"
          value = "/kong/kong.yml"
        }
      ]

      portMappings = [
        {
          containerPort = 8000
          hostPort      = 8000
          protocol      = "tcp"
        },
        {
          containerPort = 8001
          hostPort      = 8001
          protocol      = "tcp"
        }
      ]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = "/ecs/king-ai-kong"
          awslogs-region        = var.aws_region
          awslogs-stream-prefix = "kong"
        }
      }

      mountPoints = [
        {
          sourceVolume  = "kong-config"
          containerPath = "/kong"
          readOnly      = true
        }
      ]
    }
  ])

  volume {
    name = "kong-config"
    
    efs_volume_configuration {
      file_system_id = aws_efs_file_system.kong_config.id
      root_directory = "/"
    }
  }

  tags = {
    Name        = "king-ai-${var.environment}-kong"
    Environment = var.environment
  }
}

# EFS for Kong configuration
resource "aws_efs_file_system" "kong_config" {
  creation_token = "king-ai-${var.environment}-kong-config"
  encrypted      = true

  tags = {
    Name        = "king-ai-${var.environment}-kong-config"
    Environment = var.environment
  }
}

# Kong ECS Service
resource "aws_ecs_service" "kong" {
  name            = "king-ai-${var.environment}-kong"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.kong.arn
  desired_count   = 2
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = aws_subnet.private[*].id
    security_groups  = [aws_security_group.kong.id]
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.kong.arn
    container_name   = "kong"
    container_port   = 8000
  }

  depends_on = [aws_lb_listener.http]
}

# Kong Security Group
resource "aws_security_group" "kong" {
  name        = "king-ai-${var.environment}-kong-sg"
  description = "Security group for Kong API Gateway"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port       = 8000
    to_port         = 8000
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
    description     = "Kong proxy from ALB"
  }

  ingress {
    from_port   = 8001
    to_port     = 8001
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
    description = "Kong admin API (internal only)"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name        = "king-ai-${var.environment}-kong-sg"
    Environment = var.environment
  }
}

# Kong Target Group
resource "aws_lb_target_group" "kong" {
  name        = "king-ai-${var.environment}-kong-tg"
  port        = 8000
  protocol    = "HTTP"
  vpc_id      = aws_vpc.main.id
  target_type = "ip"

  health_check {
    enabled             = true
    healthy_threshold   = 2
    interval            = 30
    matcher             = "200"
    path                = "/status"
    port                = "traffic-port"
    protocol            = "HTTP"
    timeout             = 5
    unhealthy_threshold = 3
  }

  tags = {
    Name        = "king-ai-${var.environment}-kong-tg"
    Environment = var.environment
  }
}

# ECS Cluster (if not already defined)
resource "aws_ecs_cluster" "main" {
  name = "king-ai-${var.environment}-cluster"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }

  tags = {
    Name        = "king-ai-${var.environment}-cluster"
    Environment = var.environment
  }
}

# IAM Role for ECS Execution
resource "aws_iam_role" "ecs_execution" {
  name = "king-ai-${var.environment}-ecs-execution"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "ecs_execution" {
  role       = aws_iam_role.ecs_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# IAM Role for ECS Task
resource "aws_iam_role" "ecs_task" {
  name = "king-ai-${var.environment}-ecs-task"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })
}
```

2. Create Kong configuration file `infrastructure/terraform/files/kong.yml`:

```yaml
_format_version: "3.0"
_transform: true

services:
  - name: king-ai-api
    url: http://king-ai-api:8080
    routes:
      - name: api-route
        paths:
          - /api
        strip_path: false
    plugins:
      - name: jwt
        config:
          claims_to_verify:
            - exp
          key_claim_name: iss
      - name: rate-limiting
        config:
          minute: 100
          policy: local
      - name: cors
        config:
          origins:
            - "*"
          methods:
            - GET
            - POST
            - PUT
            - DELETE
            - OPTIONS
          headers:
            - Authorization
            - Content-Type
          credentials: true
          max_age: 3600

  - name: ollama-inference
    url: http://ollama-cluster:11434
    routes:
      - name: inference-route
        paths:
          - /infer
    plugins:
      - name: jwt
      - name: rate-limiting
        config:
          minute: 50
          policy: local

consumers:
  - username: king-ai-service
    jwt_secrets:
      - algorithm: HS256
        key: king-ai-api
        secret: ${JWT_SECRET}
```

---

## Task 1.9: Add Datadog Terraform Provider (Missing)

**Priority:** 🟡 MEDIUM  
**Files to Create:** `infrastructure/terraform/datadog.tf`  
**Issue:** Datadog integration exists in app code but no Terraform monitoring

### Instructions

1. Add Datadog provider to `infrastructure/terraform/main.tf`:

```hcl
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    datadog = {
      source  = "DataDog/datadog"
      version = "~> 3.30"
    }
  }
}

provider "datadog" {
  api_key = var.datadog_api_key
  app_key = var.datadog_app_key
}
```

2. Add variables to `variables.tf`:

```hcl
variable "datadog_api_key" {
  description = "Datadog API key"
  type        = string
  sensitive   = true
  default     = ""
}

variable "datadog_app_key" {
  description = "Datadog Application key"
  type        = string
  sensitive   = true
  default     = ""
}
```

3. Create `infrastructure/terraform/datadog.tf`:

```hcl
# Datadog Monitoring for King AI v2

# GPU Instance Monitoring
resource "datadog_monitor" "gpu_utilization" {
  count   = var.datadog_api_key != "" ? 1 : 0
  name    = "King AI - GPU Utilization High"
  type    = "metric alert"
  message = "GPU utilization is above 90% on {{host.name}}. @slack-king-ai-alerts"

  query = "avg(last_5m):avg:nvidia.gpu.utilization{environment:${var.environment}} by {host} > 90"

  monitor_thresholds {
    critical = 90
    warning  = 80
  }

  tags = ["environment:${var.environment}", "service:king-ai", "team:ai-platform"]
}

# Inference Queue Depth
resource "datadog_monitor" "queue_depth" {
  count   = var.datadog_api_key != "" ? 1 : 0
  name    = "King AI - Inference Queue Backing Up"
  type    = "metric alert"
  message = "Inference queue depth is high. Consider scaling GPU instances. @pagerduty-king-ai"

  query = "avg(last_5m):avg:aws.sqs.approximate_number_of_messages_visible{queuename:king-ai-inference-queue,environment:${var.environment}} > 100"

  monitor_thresholds {
    critical = 100
    warning  = 50
  }

  tags = ["environment:${var.environment}", "service:king-ai"]
}

# API Latency
resource "datadog_monitor" "api_latency" {
  count   = var.datadog_api_key != "" ? 1 : 0
  name    = "King AI - API Latency High"
  type    = "metric alert"
  message = "API p95 latency is above 5 seconds. @slack-king-ai-alerts"

  query = "avg(last_5m):avg:trace.fastapi.request.duration.by.resource_service.95p{service:king-ai-api,environment:${var.environment}} > 5"

  monitor_thresholds {
    critical = 5
    warning  = 3
  }

  tags = ["environment:${var.environment}", "service:king-ai"]
}

# Error Rate
resource "datadog_monitor" "error_rate" {
  count   = var.datadog_api_key != "" ? 1 : 0
  name    = "King AI - Error Rate Elevated"
  type    = "metric alert"
  message = "Error rate is above 5%. Check logs for details. @slack-king-ai-alerts"

  query = "sum(last_5m):sum:trace.fastapi.request.errors{service:king-ai-api,environment:${var.environment}}.as_count() / sum:trace.fastapi.request.hits{service:king-ai-api,environment:${var.environment}}.as_count() * 100 > 5"

  monitor_thresholds {
    critical = 5
    warning  = 2
  }

  tags = ["environment:${var.environment}", "service:king-ai"]
}

# Dashboard
resource "datadog_dashboard" "king_ai" {
  count       = var.datadog_api_key != "" ? 1 : 0
  title       = "King AI v2 - Operations Dashboard"
  description = "Main operational dashboard for King AI autonomous business system"
  layout_type = "ordered"

  widget {
    group_definition {
      title = "GPU Inference Cluster"

      widget {
        timeseries_definition {
          title = "GPU Utilization"
          request {
            q            = "avg:nvidia.gpu.utilization{environment:${var.environment}} by {host}"
            display_type = "line"
          }
        }
      }

      widget {
        timeseries_definition {
          title = "Inference Queue Depth"
          request {
            q            = "avg:aws.sqs.approximate_number_of_messages_visible{queuename:king-ai-inference-queue}"
            display_type = "bars"
          }
        }
      }
    }
  }

  widget {
    group_definition {
      title = "API Performance"

      widget {
        timeseries_definition {
          title = "Request Latency (p95)"
          request {
            q            = "avg:trace.fastapi.request.duration.by.resource_service.95p{service:king-ai-api}"
            display_type = "line"
          }
        }
      }

      widget {
        timeseries_definition {
          title = "Requests/sec"
          request {
            q            = "sum:trace.fastapi.request.hits{service:king-ai-api}.as_rate()"
            display_type = "line"
          }
        }
      }
    }
  }
}
```

---

## Task 1.10: Make ElastiCache Highly Available

**Priority:** 🟡 MEDIUM  
**File:** `infrastructure/terraform/elasticache.tf`  
**Issue:** Only 1 Redis node - not production-ready

### Instructions

1. Open `infrastructure/terraform/elasticache.tf`
2. Update the configuration for HA:

**Before:**
```hcl
resource "aws_elasticache_cluster" "redis" {
  cluster_id           = "king-ai-${var.environment}-redis"
  engine               = "redis"
  node_type            = var.redis_node_type
  num_cache_nodes      = 1
  ...
}
```

**After (use Replication Group for HA):**
```hcl
resource "aws_elasticache_replication_group" "redis" {
  replication_group_id       = "king-ai-${var.environment}-redis"
  description                = "King AI Redis cluster with HA"
  node_type                  = var.redis_node_type
  port                       = 6379
  parameter_group_name       = aws_elasticache_parameter_group.redis.name
  subnet_group_name          = aws_elasticache_subnet_group.redis.name
  security_group_ids         = [aws_security_group.redis.id]
  
  # HA Configuration
  automatic_failover_enabled = true
  multi_az_enabled          = true
  num_cache_clusters        = 3  # 1 primary + 2 replicas
  
  # Maintenance
  maintenance_window        = "sun:05:00-sun:06:00"
  snapshot_retention_limit  = 7
  snapshot_window           = "03:00-04:00"
  
  # Encryption
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  
  engine_version            = "7.0"
  
  tags = {
    Name        = "king-ai-${var.environment}-redis"
    Environment = var.environment
  }
}

# Update outputs to use replication group
output "redis_primary_endpoint" {
  value = aws_elasticache_replication_group.redis.primary_endpoint_address
}

output "redis_reader_endpoint" {
  value = aws_elasticache_replication_group.redis.reader_endpoint_address
}
```

---

## Part 1 Verification Checklist

After completing all tasks, verify:

- [ ] `terraform validate` passes without errors
- [ ] `terraform plan` shows expected changes
- [ ] No duplicate resource errors
- [ ] SSH access restricted to VPN CIDR
- [ ] ALB is internal
- [ ] Model versions consistent (llama3.1:70b)
- [ ] Kong API Gateway resources created
- [ ] Datadog monitors configured
- [ ] Redis is HA with 3 nodes
- [ ] Ollama setup script exists

---

## Next Steps

Proceed to **IMPLEMENTATION_PLAN_PART2.md** for Master AI & Evolution Engine fixes.
