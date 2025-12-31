# Datadog Monitoring for King AI v2

# Note: These resources are only created when Datadog credentials are provided
# Set datadog_api_key and datadog_app_key in terraform.tfvars

# GPU Instance Monitoring
resource "datadog_monitor" "gpu_utilization" {
  count   = var.datadog_api_key != "" ? 1 : 0
  name    = "King AI - GPU Utilization High"
  type    = "metric alert"
  message = <<-EOT
    GPU utilization is above 90% on {{host.name}}.
    
    This may indicate:
    - High inference demand
    - Need to scale up GPU instances
    - Potential model optimization needed
    
    @slack-king-ai-alerts
  EOT

  query = "avg(last_5m):avg:nvidia.gpu.utilization{environment:${var.environment}} by {host} > 90"

  monitor_thresholds {
    critical = 90
    warning  = 80
  }

  notify_no_data    = false
  renotify_interval = 60

  tags = ["environment:${var.environment}", "service:king-ai", "team:ai-platform"]
}

# Inference Queue Depth
resource "datadog_monitor" "queue_depth" {
  count   = var.datadog_api_key != "" ? 1 : 0
  name    = "King AI - Inference Queue Backing Up"
  type    = "metric alert"
  message = <<-EOT
    Inference queue depth is high (>100 messages).
    
    Actions:
    1. Check GPU instance health
    2. Consider scaling GPU cluster
    3. Review inference request patterns
    
    @pagerduty-king-ai
  EOT

  query = "avg(last_5m):avg:aws.sqs.approximate_number_of_messages_visible{queuename:king-ai-inference-queue,environment:${var.environment}} > 100"

  monitor_thresholds {
    critical = 100
    warning  = 50
  }

  notify_no_data    = true
  no_data_timeframe = 10

  tags = ["environment:${var.environment}", "service:king-ai", "team:ai-platform"]
}

# API Latency
resource "datadog_monitor" "api_latency" {
  count   = var.datadog_api_key != "" ? 1 : 0
  name    = "King AI - API Latency High"
  type    = "metric alert"
  message = <<-EOT
    API p95 latency is above 5 seconds.
    
    Check:
    - Database connection pool
    - Redis cache hit rate
    - LLM inference times
    
    @slack-king-ai-alerts
  EOT

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
  message = <<-EOT
    Error rate is above 5%.
    
    Investigate:
    - Recent deployments
    - External service failures
    - Database connectivity
    
    @slack-king-ai-alerts @pagerduty-king-ai
  EOT

  query = "sum(last_5m):sum:trace.fastapi.request.errors{service:king-ai-api,environment:${var.environment}}.as_count() / sum:trace.fastapi.request.hits{service:king-ai-api,environment:${var.environment}}.as_count() * 100 > 5"

  monitor_thresholds {
    critical = 5
    warning  = 2
  }

  tags = ["environment:${var.environment}", "service:king-ai"]
}

# LLM Circuit Breaker
resource "datadog_monitor" "circuit_breaker" {
  count   = var.datadog_api_key != "" ? 1 : 0
  name    = "King AI - LLM Circuit Breaker Open"
  type    = "metric alert"
  message = <<-EOT
    LLM provider circuit breaker is open.
    
    This means the primary LLM provider is experiencing failures.
    System is using fallback providers.
    
    @pagerduty-king-ai
  EOT

  query = "avg(last_5m):avg:king_ai.llm.circuit_breaker.open{environment:${var.environment}} by {provider} > 0"

  monitor_thresholds {
    critical = 0
  }

  notify_no_data = false

  tags = ["environment:${var.environment}", "service:king-ai", "component:llm"]
}

# Business Unit Health
resource "datadog_monitor" "business_health" {
  count   = var.datadog_api_key != "" ? 1 : 0
  name    = "King AI - Business Unit Unhealthy"
  type    = "metric alert"
  message = <<-EOT
    A business unit has been in unhealthy state.
    
    Review:
    - Business unit KPIs
    - Recent operations
    - Approval queue
    
    @slack-king-ai-business
  EOT

  query = "avg(last_15m):avg:king_ai.business.health_score{environment:${var.environment}} by {business_id} < 0.5"

  monitor_thresholds {
    critical = 0.3
    warning  = 0.5
  }

  tags = ["environment:${var.environment}", "service:king-ai", "component:business"]
}

# Evolution Proposal Failure
resource "datadog_monitor" "evolution_failures" {
  count   = var.datadog_api_key != "" ? 1 : 0
  name    = "King AI - Evolution Proposals Failing"
  type    = "metric alert"
  message = <<-EOT
    Evolution proposals are failing at a high rate.
    
    This may indicate:
    - Code quality issues
    - Test failures
    - Sandbox problems
    
    @slack-king-ai-evolution
  EOT

  query = "sum(last_1h):sum:king_ai.evolution.proposal.failed{environment:${var.environment}}.as_count() > 3"

  monitor_thresholds {
    critical = 3
    warning  = 1
  }

  tags = ["environment:${var.environment}", "service:king-ai", "component:evolution"]
}

# Dashboard
resource "datadog_dashboard" "king_ai" {
  count       = var.datadog_api_key != "" ? 1 : 0
  title       = "King AI v2 - Operations Dashboard"
  description = "Main operational dashboard for King AI autonomous business system"
  layout_type = "ordered"

  widget {
    group_definition {
      title            = "System Overview"
      layout_type      = "ordered"
      background_color = "vivid_blue"

      widget {
        query_value_definition {
          title = "Active Business Units"
          request {
            q          = "sum:king_ai.business.active{environment:${var.environment}}"
            aggregator = "last"
          }
          precision = 0
        }
      }

      widget {
        query_value_definition {
          title = "Total Revenue (24h)"
          request {
            q          = "sum:king_ai.business.revenue{environment:${var.environment}}.rollup(sum, 86400)"
            aggregator = "last"
          }
          precision = 2
          custom_unit = "$"
        }
      }

      widget {
        query_value_definition {
          title = "Pending Approvals"
          request {
            q          = "sum:king_ai.approvals.pending{environment:${var.environment}}"
            aggregator = "last"
          }
          precision = 0
        }
      }
    }
  }

  widget {
    group_definition {
      title       = "GPU Inference Cluster"
      layout_type = "ordered"

      widget {
        timeseries_definition {
          title = "GPU Utilization"
          request {
            q            = "avg:nvidia.gpu.utilization{environment:${var.environment}} by {host}"
            display_type = "line"
          }
          yaxis {
            max = "100"
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

      widget {
        timeseries_definition {
          title = "GPU Memory Usage"
          request {
            q            = "avg:nvidia.gpu.memory.used{environment:${var.environment}} by {host}"
            display_type = "area"
          }
        }
      }
    }
  }

  widget {
    group_definition {
      title       = "API Performance"
      layout_type = "ordered"

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

      widget {
        timeseries_definition {
          title = "Error Rate"
          request {
            q            = "100 * sum:trace.fastapi.request.errors{service:king-ai-api}.as_count() / sum:trace.fastapi.request.hits{service:king-ai-api}.as_count()"
            display_type = "line"
          }
          yaxis {
            max = "10"
          }
        }
      }
    }
  }

  widget {
    group_definition {
      title       = "LLM Routing"
      layout_type = "ordered"

      widget {
        timeseries_definition {
          title = "LLM Provider Usage"
          request {
            q            = "sum:king_ai.llm.requests{environment:${var.environment}} by {provider}.as_count()"
            display_type = "bars"
          }
        }
      }

      widget {
        timeseries_definition {
          title = "LLM Latency by Provider"
          request {
            q            = "avg:king_ai.llm.latency{environment:${var.environment}} by {provider}"
            display_type = "line"
          }
        }
      }
    }
  }

  tags = ["environment:${var.environment}", "service:king-ai"]
}
