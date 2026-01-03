# Alerting Thresholds and Monitoring Configuration

## Overview

This document defines alerting thresholds, monitoring rules, and incident response triggers for the Mother-Harness platform.

**Last Updated:** December 22, 2024
**Review Frequency:** Quarterly or after major incidents

---

## Alert Severity Levels

| Severity | Description | Response Time | Notification Method |
|----------|-------------|---------------|---------------------|
| **P1 - Critical** | System down, data loss, security breach | Immediate | PagerDuty + SMS + Phone |
| **P2 - High** | Degraded performance, elevated errors | 15 minutes | PagerDuty + Slack |
| **P3 - Medium** | Warning conditions, approaching thresholds | 1 hour | Slack + Email |
| **P4 - Low** | Informational, trends | 24 hours | Email |

---

## System Health Alerts

### Service Availability

#### P1: Service Down
**Metric:** Health check failure
**Threshold:** 2 consecutive failures (60 seconds apart)
**Query:**
```promql
up{job="orchestrator"} == 0 for 1m
up{job="docling"} == 0 for 1m
up{job="dashboard"} == 0 for 1m
```

**Action:**
1. Check service logs
2. Verify infrastructure (Docker, network)
3. Attempt service restart
4. Escalate if restart fails

---

#### P2: Service Health Degraded
**Metric:** Health check reports "degraded"
**Threshold:** Any occurrence for >5 minutes
**Query:**
```promql
health_status{status="degraded"} for 5m
```

**Action:**
1. Check dependent services (Redis, Ollama)
2. Review error logs
3. Monitor for escalation to P1

---

#### P2: Redis Connection Failure
**Metric:** Redis connection errors
**Threshold:** >5 errors in 1 minute
**Query:**
```promql
rate(redis_connection_errors_total[1m]) > 5
```

**Action:**
1. Check Redis service status
2. Verify network connectivity
3. Check Redis memory/CPU
4. Review connection pool settings

---

### Response Time Alerts

#### P2: High Latency
**Metric:** API response time P95
**Threshold:** >2000ms for 5 minutes
**Query:**
```promql
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 2
```

**Action:**
1. Check system resources (CPU, memory)
2. Review slow queries in logs
3. Check Redis performance
4. Verify external service latency (Ollama, n8n)

---

#### P3: Elevated Latency
**Metric:** API response time P95
**Threshold:** >1000ms for 10 minutes
**Query:**
```promql
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[10m])) > 1
```

**Action:**
1. Monitor for escalation
2. Check recent deployments
3. Review traffic patterns

---

## Error Rate Alerts

### Application Errors

#### P1: Critical Error Rate
**Metric:** 5xx error percentage
**Threshold:** >10% of requests for 5 minutes
**Query:**
```promql
(rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m])) > 0.10
```

**Action:**
1. **IMMEDIATE**: Check if data corruption is occurring
2. Review error logs for root cause
3. Consider emergency rollback
4. Escalate to engineering

---

#### P2: Elevated Error Rate
**Metric:** 5xx error percentage
**Threshold:** >5% of requests for 5 minutes
**Query:**
```promql
(rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m])) > 0.05
```

**Action:**
1. Identify error pattern (specific endpoint?)
2. Check recent changes
3. Review service dependencies
4. Prepare for potential rollback

---

#### P3: Increased Error Rate
**Metric:** 5xx error percentage
**Threshold:** >1% of requests for 15 minutes
**Query:**
```promql
(rate(http_requests_total{status=~"5.."}[15m]) / rate(http_requests_total[15m])) > 0.01
```

**Action:**
1. Monitor trend
2. Review logs for patterns
3. Document findings

---

### Task Execution Errors

#### P2: High Task Failure Rate
**Metric:** Failed task executions
**Threshold:** >20% failures in 10 minutes
**Query:**
```promql
(rate(task_execution_total{status="failed"}[10m]) / rate(task_execution_total[10m])) > 0.20
```

**Action:**
1. Check agent availability
2. Verify Ollama/n8n connectivity
3. Review failed task logs
4. Check for pattern (specific agent type?)

---

#### P3: Elevated Task Failure Rate
**Metric:** Failed task executions
**Threshold:** >10% failures in 15 minutes
**Query:**
```promql
(rate(task_execution_total{status="failed"}[15m]) / rate(task_execution_total[15m])) > 0.10
```

**Action:**
1. Monitor for escalation
2. Identify failure patterns
3. Check model performance

---

## Security Alerts

### Authentication Failures

#### P1: Authentication Attack Suspected
**Metric:** Failed authentication attempts
**Threshold:** >20 failures from single IP in 5 minutes
**Query:**
```promql
rate(auth_failures_total[5m]) by (ip_address) > 20
```

**Action:**
1. **IMMEDIATE**: Block IP address
2. Review authentication logs
3. Check for credential stuffing
4. Notify security team

---

#### P2: Elevated Auth Failures
**Metric:** Failed authentication attempts
**Threshold:** >5 failures from single IP in 5 minutes
**Query:**
```promql
rate(auth_failures_total[5m]) by (ip_address) > 5
```

**Action:**
1. Monitor for escalation
2. Log IP for analysis
3. Consider rate limiting

---

### Authorization Violations

#### P1: Unauthorized Access Attempt
**Metric:** 403 Forbidden responses
**Threshold:** Any occurrence
**Query:**
```promql
increase(http_requests_total{status="403"}[1m]) > 0
```

**Action:**
1. Review request details (user, endpoint, action)
2. Check if legitimate user misconfiguration
3. Investigate for privilege escalation attempt
4. Document incident

---

#### P2: Approval Bypass Attempt
**Metric:** Rejected approval attempts
**Threshold:** >3 rejections from single user in 15 minutes
**Query:**
```promql
rate(approval_response_total{status="rejected"}[15m]) by (user_id) > 3
```

**Action:**
1. Review rejected approvals
2. Contact user to verify legitimate activity
3. Check for automation/scripting

---

## Resource Utilization Alerts

### Memory Usage

#### P1: Critical Memory Usage
**Metric:** Container memory usage
**Threshold:** >90% of limit for 2 minutes
**Query:**
```promql
(container_memory_usage_bytes / container_spec_memory_limit_bytes) > 0.90 for 2m
```

**Action:**
1. **IMMEDIATE**: Identify memory leak
2. Consider emergency restart
3. Review memory limits
4. Check for runaway processes

---

#### P2: High Memory Usage
**Metric:** Container memory usage
**Threshold:** >80% of limit for 5 minutes
**Query:**
```promql
(container_memory_usage_bytes / container_spec_memory_limit_bytes) > 0.80 for 5m
```

**Action:**
1. Monitor for escalation
2. Identify memory-intensive operations
3. Consider increasing limits
4. Review for memory leaks

---

### CPU Usage

#### P2: High CPU Usage
**Metric:** Container CPU usage
**Threshold:** >80% for 10 minutes
**Query:**
```promql
(rate(container_cpu_usage_seconds_total[5m]) / container_spec_cpu_quota) > 0.80 for 10m
```

**Action:**
1. Identify CPU-intensive operations
2. Check for infinite loops
3. Review concurrent task limits
4. Consider scaling

---

### Disk Usage

#### P1: Critical Disk Space
**Metric:** Disk usage
**Threshold:** >95% full
**Query:**
```promql
(node_filesystem_avail_bytes / node_filesystem_size_bytes) < 0.05
```

**Action:**
1. **IMMEDIATE**: Clean up logs
2. Check backup sizes
3. Clear temporary files
4. Expand storage if needed

---

#### P2: High Disk Usage
**Metric:** Disk usage
**Threshold:** >85% full
**Query:**
```promql
(node_filesystem_avail_bytes / node_filesystem_size_bytes) < 0.15
```

**Action:**
1. Review disk usage trends
2. Plan storage expansion
3. Implement log rotation
4. Review data retention policy

---

## Database Alerts (Redis)

### Redis Memory

#### P1: Redis Memory Critical
**Metric:** Redis memory usage
**Threshold:** >95% of maxmemory
**Query:**
```promql
(redis_memory_used_bytes / redis_memory_max_bytes) > 0.95
```

**Action:**
1. **IMMEDIATE**: Check eviction policy
2. Identify large keys
3. Consider increasing memory limit
4. Review data retention

---

#### P2: Redis Memory High
**Metric:** Redis memory usage
**Threshold:** >80% of maxmemory
**Query:**
```promql
(redis_memory_used_bytes / redis_memory_max_bytes) > 0.80
```

**Action:**
1. Monitor growth trend
2. Review key expiration settings
3. Plan capacity increase

---

### Redis Operations

#### P2: Slow Redis Commands
**Metric:** Slow log entries
**Threshold:** >10 slow commands in 5 minutes
**Query:**
```promql
rate(redis_slowlog_length[5m]) > 10
```

**Action:**
1. Review slow log for patterns
2. Optimize problematic queries
3. Check for large key operations
4. Consider read replicas

---

#### P2: High Redis Connections
**Metric:** Active connections
**Threshold:** >80% of max connections
**Query:**
```promql
(redis_connected_clients / redis_config_maxclients) > 0.80
```

**Action:**
1. Check for connection leaks
2. Review connection pool settings
3. Increase max connections if needed
4. Identify services with excessive connections

---

## Business Metrics Alerts

### Task Volume

#### P3: Unusual Task Volume
**Metric:** Tasks created per hour
**Threshold:** >200% of 7-day average OR <50% of 7-day average
**Query:**
```promql
rate(task_created_total[1h]) > (avg_over_time(rate(task_created_total[1h])[7d]) * 2)
OR
rate(task_created_total[1h]) < (avg_over_time(rate(task_created_total[1h])[7d]) * 0.5)
```

**Action:**
1. Check for legitimate usage spike
2. Verify no service disruption
3. Monitor for abuse
4. Trend analysis

---

### Approval Backlog

#### P2: Approval Backlog Growing
**Metric:** Pending approvals
**Threshold:** >50 pending for >1 hour
**Query:**
```promql
approval_pending_total > 50 for 1h
```

**Action:**
1. Notify approvers
2. Check approval notification system
3. Review if approvals can be auto-approved
4. Consider staffing levels

---

### Cost Budget

#### P2: Model Cost Budget Alert
**Metric:** Daily model API cost
**Threshold:** >80% of daily budget
**Query:**
```promql
sum(model_api_cost_total) > (daily_budget * 0.80)
```

**Action:**
1. Review high-cost operations
2. Check for unusual usage
3. Consider increasing budget
4. Optimize model selection

---

## Integration Alerts

### External Service Health

#### P2: Ollama Unavailable
**Metric:** Ollama health check
**Threshold:** Failed for >2 minutes
**Query:**
```promql
up{job="ollama"} == 0 for 2m
```

**Action:**
1. Check Ollama service status
2. Verify network connectivity
3. Fall back to cloud models
4. Restart if needed

---

#### P3: n8n Unavailable
**Metric:** n8n health check
**Threshold:** Failed for >5 minutes
**Query:**
```promql
up{job="n8n"} == 0 for 5m
```

**Action:**
1. Check n8n service status
2. Fall back to local agent execution
3. Review workflows

---

#### P2: Docling Service Degraded
**Metric:** Document processing failures
**Threshold:** >20% failures in 10 minutes
**Query:**
```promql
(rate(document_processing_total{status="failed"}[10m]) / rate(document_processing_total[10m])) > 0.20
```

**Action:**
1. Check Docling service logs
2. Verify file system access
3. Review failed document types
4. Check memory/CPU usage

---

## Backup and Recovery Alerts

### Backup Failures

#### P1: Backup Failed
**Metric:** Backup completion status
**Threshold:** Any failure
**Query:**
```promql
backup_status{status="failed"} == 1
```

**Action:**
1. **IMMEDIATE**: Retry backup
2. Check disk space
3. Verify Redis connectivity
4. Review backup logs
5. Escalate if retry fails

---

#### P2: Backup Not Running
**Metric:** Last successful backup time
**Threshold:** >36 hours (should run daily)
**Query:**
```promql
(time() - backup_last_success_timestamp) > 129600
```

**Action:**
1. Check backup cron job
2. Verify backup script permissions
3. Manually trigger backup
4. Review automation setup

---

## Alert Configuration Examples

### Prometheus AlertManager

```yaml
groups:
  - name: mother-harness-critical
    interval: 30s
    rules:
      - alert: ServiceDown
        expr: up{job=~"orchestrator|docling|dashboard"} == 0
        for: 1m
        labels:
          severity: P1
        annotations:
          summary: "Service {{ $labels.job }} is down"
          description: "{{ $labels.job }} has been down for 1 minute"

      - alert: HighErrorRate
        expr: (rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m])) > 0.10
        for: 5m
        labels:
          severity: P1
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value | humanizePercentage }}"

      - alert: MemoryCritical
        expr: (container_memory_usage_bytes / container_spec_memory_limit_bytes) > 0.90
        for: 2m
        labels:
          severity: P1
        annotations:
          summary: "Critical memory usage in {{ $labels.container }}"
          description: "Memory usage is {{ $value | humanizePercentage }}"
```

---

## Alert Routing

### Notification Channels

| Severity | Slack | Email | PagerDuty | SMS |
|----------|-------|-------|-----------|-----|
| P1 | ✅ #incidents | ✅ on-call@  | ✅ On-call | ✅ |
| P2 | ✅ #alerts | ✅ eng@      | ✅ On-call | ❌ |
| P3 | ✅ #monitoring | ✅ eng@ | ❌ | ❌ |
| P4 | ❌ | ✅ eng@ | ❌ | ❌ |

### Alert Grouping

- Group by: `alertname`, `service`, `severity`
- Group wait: 30s
- Group interval: 5m
- Repeat interval: 4h (P1/P2), 12h (P3), 24h (P4)

---

## Alert Tuning Guidelines

### Reducing False Positives

1. **Start conservative** - Set wider thresholds initially
2. **Monitor for 2 weeks** - Observe baseline behavior
3. **Adjust gradually** - Tighten by 10-20% at a time
4. **Consider time of day** - Different thresholds for peak vs off-peak
5. **Seasonal adjustments** - Review quarterly

### Key Metrics to Track

- Alert volume per day
- Mean time to acknowledge (MTTA)
- Mean time to resolve (MTTR)
- False positive rate
- Alert fatigue score (alerts per on-call shift)

### Target Metrics

- MTTA: <5 minutes (P1), <15 minutes (P2)
- MTTR: <30 minutes (P1), <2 hours (P2)
- False positive rate: <10%
- Alert fatigue: <15 alerts per shift

---

## Testing Alerts

### Monthly Alert Testing

```bash
# Test critical service down alert
docker-compose stop orchestrator
# Wait for alert, verify received
docker-compose start orchestrator

# Test high error rate (requires test traffic generator)
# Simulate 500 errors

# Test memory alert
# Use stress-ng or similar tool

# Test authentication failure alert
# Attempt failed logins from test account
```

### Runbook for Each Alert

Each alert should have:
1. **Trigger condition** - Exact threshold
2. **Impact** - What does this mean for users?
3. **Investigation steps** - How to diagnose
4. **Resolution steps** - How to fix
5. **Escalation path** - Who to call if stuck

---

## Dashboard Integration

### Required Dashboards

1. **System Overview** - All services health at a glance
2. **Error Rates** - 5xx errors, task failures, auth failures
3. **Performance** - Latency P50/P95/P99, throughput
4. **Resources** - CPU, memory, disk, Redis
5. **Business Metrics** - Tasks, approvals, costs
6. **Security** - Auth attempts, authorization violations

### Alert Annotations

Add alert firing/resolving as annotations on dashboards for correlation

---

## Review and Maintenance

### Monthly Review

- Review alert volume and false positive rate
- Update thresholds based on new baselines
- Add alerts for new features
- Remove obsolete alerts

### Quarterly Review

- Full alert audit
- Update runbooks
- Test all P1 alerts
- Review escalation procedures

---

*Last Updated: December 22, 2024*
*Alert Configuration Version: 1.0*
*Next Review: January 22, 2025*
