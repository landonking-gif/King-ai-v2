# Load Testing Guide

## Overview

This guide covers load testing procedures for Mother-Harness to validate performance under concurrent load, particularly the requirement for handling 10 concurrent task executions.

## Test Requirements

### Performance Targets

- **Concurrent Tasks**: 10 simultaneous task executions
- **Success Rate**: ≥90% tasks complete successfully
- **Response Time P95**: <2 seconds
- **Response Time P99**: <5 seconds
- **Error Rate**: <1%
- **System Stability**: No crashes or memory leaks

## Test Approaches

### 1. Integration Tests (Recommended for CI/CD)

Located in [tests/load/concurrent-tasks.test.ts](../tests/load/concurrent-tasks.test.ts)

**Advantages:**
- No additional dependencies
- Runs in CI/CD pipeline
- Detailed assertions
- Easy to debug

**Run Tests:**
```bash
cd tests/load
TEST_JWT_TOKEN=your-token npm test
```

**Test Scenarios:**
1. Health check under load (100 concurrent requests)
2. Create 10 tasks concurrently
3. Retrieve all created tasks
4. Execute 10 tasks concurrently
5. Monitor concurrent execution to completion
6. Verify system health post-load
7. Measure response times (P95, P99)
8. Rapid create/retrieve cycles

### 2. Artillery Load Testing (Recommended for Production)

Located in [tests/load/](../tests/load/)

**Advantages:**
- More realistic load patterns
- Better metrics and reporting
- Gradual ramp-up
- Sustained load testing

**Setup:**
```bash
cd tests/load
npm install
```

**Run Load Tests:**
```bash
# 10 concurrent tasks scenario
export TEST_JWT_TOKEN=your-token
npm run test:load

# Stress test (find breaking point)
npm run test:stress

# Quick smoke test
npm run test:quick
```

**Generate Report:**
```bash
npm run report latest-report.json
```

## Test Scenarios

### Scenario 1: Concurrent Task Creation

**Objective**: Validate 10 tasks can be created simultaneously

**Steps:**
1. Create 10 HTTP POST requests to `/api/tasks`
2. Execute all requests in parallel
3. Verify all return 200 OK
4. Measure total time and average per request

**Expected Results:**
- All 10 tasks created successfully
- Average creation time < 200ms
- No rate limiting errors

### Scenario 2: Concurrent Task Execution

**Objective**: Validate 10 tasks can execute simultaneously

**Steps:**
1. Create 10 tasks with simple queries
2. Execute all 10 tasks in parallel via `/api/tasks/{id}/execute`
3. Monitor execution progress
4. Wait for completion or timeout (3 minutes)

**Expected Results:**
- ≥9 out of 10 tasks complete successfully
- Total execution time < 3 minutes
- No resource exhaustion errors

### Scenario 3: Mixed Operations

**Objective**: Simulate realistic user behavior

**Steps:**
1. 70% task creation and execution
2. 20% project and library management
3. 10% approval workflow
4. 5% health checks

**Expected Results:**
- Error rate < 1%
- P95 response time < 2 seconds
- System remains healthy throughout

### Scenario 4: Stress Testing

**Objective**: Find system breaking point

**Steps:**
1. Gradually increase load from 5 to 50 users/second
2. Monitor response times and error rates
3. Identify degradation point
4. Verify graceful degradation

**Expected Results:**
- System handles at least 20 concurrent users
- Graceful degradation (429/503 responses, not crashes)
- Quick recovery after load reduction

## Running Load Tests

### Prerequisites

```bash
# Set authentication token
export TEST_JWT_TOKEN=your-jwt-token

# Ensure services are running
docker-compose ps

# Verify health
curl http://localhost:8000/health
```

### Integration Test Approach

```bash
# Navigate to test directory
cd tests/load

# Install dependencies (if using Artillery)
npm install

# Run integration tests with vitest
npx vitest run concurrent-tasks.test.ts

# Run with verbose output
npx vitest run concurrent-tasks.test.ts --reporter=verbose
```

### Artillery Approach

```bash
# Run 10 concurrent tasks scenario
artillery run scenario-concurrent-tasks.yml

# Run stress test
artillery run scenario-stress.yml

# Quick smoke test
artillery quick --count 10 --num 50 http://localhost:8000/health

# Generate HTML report
artillery run scenario-concurrent-tasks.yml --output report.json
artillery report report.json
```

## Monitoring During Tests

### Key Metrics to Monitor

1. **Response Times**
   - Average
   - P50, P95, P99
   - Max

2. **Throughput**
   - Requests per second
   - Tasks completed per minute

3. **Error Rates**
   - HTTP errors (4xx, 5xx)
   - Application errors
   - Timeouts

4. **System Resources**
   - CPU usage
   - Memory usage
   - Redis connections
   - Network I/O

### Monitoring Commands

```bash
# Monitor Docker resources
docker stats

# Monitor Redis
redis-cli INFO stats
redis-cli INFO memory

# Monitor application logs
docker-compose logs -f orchestrator

# Check connection counts
redis-cli CLIENT LIST | wc -l
```

### Redis Monitoring

```bash
# Monitor commands per second
redis-cli INFO stats | grep instantaneous_ops_per_sec

# Check memory usage
redis-cli INFO memory | grep used_memory_human

# Monitor connected clients
redis-cli CLIENT LIST
```

## Interpreting Results

### Success Criteria

✅ **Pass**: All criteria met
- 10 concurrent tasks execute successfully
- Error rate < 1%
- P95 < 2 seconds
- System remains stable

⚠️ **Warning**: Some concerns
- 8-9 tasks complete (80-90% success rate)
- Error rate 1-5%
- P95 2-4 seconds
- Temporary resource spikes

❌ **Fail**: Does not meet requirements
- < 8 tasks complete
- Error rate > 5%
- P95 > 4 seconds
- System crashes or hangs

### Example Good Results

```
=== Load Test Summary ===
Tasks created: 10
Tasks executed: 10
Completed: 10/10
Failed: 0/10

=== Response Time Metrics ===
Average: 145.32ms
P95: 320ms
P99: 450ms
Min: 87ms
Max: 502ms

=== System Health ===
Status: healthy
Redis: connected
Memory: 2.3GB / 32GB
CPU: 45%
```

### Example Warning Results

```
=== Load Test Summary ===
Tasks created: 10
Tasks executed: 10
Completed: 8/10
Failed: 2/10

=== Response Time Metrics ===
Average: 1250ms
P95: 2400ms
P99: 3100ms

⚠️ Action Required:
- Investigate failed tasks
- Review timeout settings
- Check resource constraints
```

## Troubleshooting

### Common Issues

#### Issue: High Failure Rate

**Symptoms:**
- Many tasks failing with errors
- 500/503 responses

**Diagnosis:**
```bash
# Check application logs
docker-compose logs orchestrator | grep ERROR

# Check Redis health
redis-cli PING
redis-cli INFO stats
```

**Solutions:**
- Increase Redis max connections
- Scale orchestrator replicas
- Increase timeout values
- Check n8n workflow availability

#### Issue: Slow Response Times

**Symptoms:**
- P95 > 2 seconds
- Timeouts occurring

**Diagnosis:**
```bash
# Check Redis latency
redis-cli --latency

# Monitor slow queries
redis-cli SLOWLOG GET 10

# Check CPU usage
docker stats
```

**Solutions:**
- Add Redis read replicas
- Optimize database queries
- Enable Redis persistence tuning
- Add caching layer

#### Issue: Memory Leaks

**Symptoms:**
- Memory usage climbing continuously
- Out of memory errors

**Diagnosis:**
```bash
# Monitor memory over time
watch -n 5 'docker stats --no-stream'

# Check Redis memory
redis-cli INFO memory
```

**Solutions:**
- Review Redis eviction policy
- Set maxmemory limit
- Check for unclosed connections
- Profile application memory

## Performance Optimization

### Redis Tuning

```bash
# Increase max connections
redis-cli CONFIG SET maxclients 10000

# Enable RDB+AOF persistence
redis-cli CONFIG SET save "900 1 300 10 60 10000"
redis-cli CONFIG SET appendonly yes

# Optimize memory
redis-cli CONFIG SET maxmemory 8gb
redis-cli CONFIG SET maxmemory-policy allkeys-lru
```

### Application Tuning

```javascript
// Increase connection pool size
const redis = new Redis({
  maxRetriesPerRequest: 3,
  enableReadyCheck: true,
  connectTimeout: 10000,
  lazyConnect: false,
  maxmemoryPolicy: 'allkeys-lru',
  enableOfflineQueue: true,
  db: 0,
});
```

### Docker Resource Limits

```yaml
# docker-compose.yml
services:
  orchestrator:
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: '4'
        reservations:
          memory: 4G
          cpus: '2'
```

## Continuous Load Testing

### CI/CD Integration

```yaml
# .github/workflows/load-test.yml
name: Load Tests
on:
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM
  workflow_dispatch:

jobs:
  load-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Start services
        run: docker-compose up -d
      - name: Wait for health
        run: |
          timeout 60 bash -c 'until curl -f http://localhost:8000/health; do sleep 2; done'
      - name: Run load tests
        run: |
          cd tests/load
          npm install
          npx vitest run concurrent-tasks.test.ts
      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: load-test-results
          path: tests/load/results/
```

### Monitoring Trends

Track metrics over time:
- Response time trends
- Error rate trends
- Resource usage trends
- Capacity planning

## Best Practices

1. **Test Regularly**: Run load tests before each release
2. **Test Realistically**: Use production-like data and scenarios
3. **Monitor Everything**: Collect metrics during tests
4. **Test Incrementally**: Gradually increase load
5. **Test Recovery**: Verify system recovers after load
6. **Document Results**: Keep history of test results
7. **Automate**: Integrate into CI/CD pipeline
8. **Test Failover**: Validate redundancy works under load

## Appendix

### Quick Reference Commands

```bash
# Run integration tests
cd tests/load && npx vitest run

# Run Artillery tests
cd tests/load && npm run test:load

# Monitor during test
docker stats
redis-cli MONITOR

# Generate report
artillery report results.json

# Check system health
curl http://localhost:8000/health
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TEST_JWT_TOKEN` | Authentication token | (required) |
| `ORCHESTRATOR_URL` | Orchestrator URL | http://localhost:8000 |
| `CONCURRENT_TASKS` | Number of tasks | 10 |

### Related Documentation

- [API Reference](api-reference.md)
- [Deployment Checklist](../DEPLOYMENT_CHECKLIST.md)
- [Monitoring Guide](monitoring.md)
