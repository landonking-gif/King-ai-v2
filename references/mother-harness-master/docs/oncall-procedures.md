# On-Call and Escalation Procedures

## Overview

This document defines the on-call rotation, escalation procedures, and incident response protocols for the Mother-Harness platform.

**Last Updated:** December 22, 2024
**Review Frequency:** Quarterly

---

## On-Call Rotation

### Rotation Schedule

**Primary On-Call:**
- Duration: 1 week (Monday 9:00 AM - Monday 9:00 AM)
- Coverage: 24/7
- Handoff: Monday mornings at 9:00 AM
- Rotation: Weekly rotation through engineering team

**Secondary On-Call:**
- Duration: 1 week (aligned with primary)
- Coverage: Backup for primary, handles escalations
- Minimum: 2 engineers on rotation at all times

**Tertiary Escalation:**
- Engineering Manager / Tech Lead
- Available for critical (P1) incidents only

### On-Call Responsibilities

**Primary On-Call Engineer:**
- Monitor PagerDuty for P1/P2 alerts
- Respond to incidents within SLA
- Triage and resolve issues
- Document incidents in post-mortem
- Communicate status updates to stakeholders

**Secondary On-Call Engineer:**
- Support primary on-call
- Handle escalations
- Provide subject matter expertise
- Take over if primary is unavailable

**Engineering Manager:**
- Final escalation point
- Decision authority for major changes
- Stakeholder communication
- Resource allocation during incidents

---

## Response Time SLAs

| Severity | Acknowledgment | First Update | Resolution Target |
|----------|----------------|--------------|-------------------|
| **P1 - Critical** | 5 minutes | 15 minutes | 1 hour |
| **P2 - High** | 15 minutes | 30 minutes | 4 hours |
| **P3 - Medium** | 1 hour | 2 hours | 1 business day |
| **P4 - Low** | 24 hours | N/A | Best effort |

**Acknowledgment:** Acknowledge alert in PagerDuty
**First Update:** Post initial findings in incident channel
**Resolution:** Issue resolved or mitigated

---

## Incident Severity Definitions

### P1 - Critical

**Criteria:**
- Complete service outage (all users affected)
- Data loss or corruption
- Security breach or suspected compromise
- Payment/billing system failure
- Critical integration failure (no workaround)

**Examples:**
- Orchestrator service completely down
- Redis data corruption detected
- Unauthorized access to production systems
- All task executions failing

**Impact:** Business-critical, all users affected
**Response:** Immediate all-hands response

---

### P2 - High

**Criteria:**
- Partial service outage (>10% users affected)
- Severe performance degradation (>50% slowdown)
- Elevated error rates (>5%)
- Critical feature unavailable
- Integration failure with workaround available

**Examples:**
- High API error rate (>5%)
- Single service degraded
- Approval workflow broken
- Document processing failing

**Impact:** Major functionality impaired
**Response:** Primary on-call responds immediately

---

### P3 - Medium

**Criteria:**
- Minor service degradation (<10% users affected)
- Moderate performance issues
- Non-critical feature unavailable
- Elevated resource usage (approaching limits)

**Examples:**
- Slow response times (P95 >1s)
- Memory usage >80%
- Single agent type unavailable
- Dashboard loading slowly

**Impact:** Degraded experience, workaround available
**Response:** Addressed during business hours

---

### P4 - Low

**Criteria:**
- Cosmetic issues
- Minor bugs
- Monitoring alerts (informational)
- Feature requests

**Examples:**
- UI text incorrect
- Log noise
- Monitoring gap identified

**Impact:** Minimal or no user impact
**Response:** Scheduled for next sprint

---

## Escalation Paths

### Standard Escalation Flow

```
P2/P3 Alert
    ↓
Primary On-Call Engineer
    ↓ (No resolution in 30min or needs help)
Secondary On-Call Engineer
    ↓ (No resolution in 1hr or P1)
Engineering Manager
    ↓ (Major incident)
VP Engineering / CTO
```

### Immediate Escalation (P1)

```
P1 Alert
    ↓
Primary On-Call (Acknowledge immediately)
    ↓ (Simultaneously notify)
Secondary On-Call + Engineering Manager
    ↓ (If needed)
War Room / All Hands
```

### Security Incident Escalation

```
Security Alert
    ↓
Primary On-Call + Security Team
    ↓ (Simultaneously)
Engineering Manager + CISO
    ↓ (If breach confirmed)
Legal + PR + C-Level
```

---

## Incident Response Procedures

### Phase 1: Triage (0-5 minutes)

**Objective:** Assess severity and impact

**Actions:**
1. **Acknowledge** alert in PagerDuty (stop paging)
2. **Check dashboards** for system overview
3. **Determine severity** (P1-P4)
4. **Create incident channel** in Slack: `#incident-YYYYMMDD-description`
5. **Post initial status**: "Investigating [issue]"

**Tools:**
- PagerDuty
- Monitoring dashboards
- Slack

---

### Phase 2: Investigation (5-15 minutes)

**Objective:** Identify root cause

**Actions:**
1. **Review metrics** in Grafana/monitoring
2. **Check logs** for errors
   ```bash
   docker-compose logs --tail=100 orchestrator
   docker-compose logs --tail=100 docling
   ```
3. **Verify dependencies**
   ```bash
   curl http://localhost:8000/health
   curl http://localhost:8080/health
   redis-cli PING
   ```
4. **Post findings** in incident channel
5. **Escalate if needed** (See escalation paths)

**Key Log Locations:**
- Orchestrator: `docker-compose logs orchestrator`
- Docling: `docker-compose logs docling`
- Redis: `docker-compose logs redis-stack`
- Dashboard: `docker-compose logs dashboard`

---

### Phase 3: Mitigation (15-60 minutes)

**Objective:** Restore service (workaround OK)

**Common Mitigation Actions:**

**Service Down:**
```bash
# Restart service
docker-compose restart orchestrator

# Or restart all
docker-compose down && docker-compose up -d

# Check logs for errors
docker-compose logs -f orchestrator
```

**High Error Rate:**
```bash
# Check Redis connectivity
redis-cli PING

# Check external services
curl http://core2:11434/api/tags  # Ollama
curl http://n8n:5678/healthz      # n8n

# Restart if hung
docker-compose restart orchestrator
```

**Memory Issues:**
```bash
# Check memory usage
docker stats

# If Redis memory high
redis-cli MEMORY PURGE

# Restart service to clear memory
docker-compose restart orchestrator
```

**High CPU:**
```bash
# Check running tasks
curl http://localhost:8000/api/tasks

# Check for runaway processes
docker exec orchestrator ps aux

# Kill specific task if needed
# (requires task termination endpoint)
```

---

### Phase 4: Recovery Verification (Post-fix)

**Objective:** Confirm issue is resolved

**Actions:**
1. **Monitor metrics** for 15 minutes
   - Error rate back to normal (<1%)
   - Latency within SLA (P95 <1s)
   - Resource usage stable
2. **Test critical paths**
   ```bash
   # Health check
   curl http://localhost:8000/health

   # Create test task
   curl -X POST http://localhost:8000/api/tasks \
     -H "Authorization: Bearer $TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"query": "Test query"}'
   ```
3. **Post resolution** in incident channel
4. **Close PagerDuty incident**
5. **Update status page** (if public-facing)

---

### Phase 5: Post-Mortem (Within 48 hours)

**Objective:** Learn and prevent recurrence

**Required for:** All P1 incidents, optional for P2

**Template:**
```markdown
# Incident Post-Mortem: [Title]

**Date:** YYYY-MM-DD
**Duration:** [start] - [end] ([total time])
**Severity:** P1/P2/P3
**On-Call:** [Primary], [Secondary]

## Impact
- Users affected: [number/percentage]
- Services impacted: [list]
- Data loss: Yes/No
- Downtime: [duration]

## Timeline
- HH:MM - Alert triggered
- HH:MM - Acknowledged
- HH:MM - Root cause identified
- HH:MM - Mitigation applied
- HH:MM - Service restored
- HH:MM - Incident closed

## Root Cause
[Detailed explanation]

## Resolution
[What was done to fix it]

## Action Items
- [ ] [Action 1] - Owner: [name], Due: [date]
- [ ] [Action 2] - Owner: [name], Due: [date]
- [ ] [Action 3] - Owner: [name], Due: [date]

## Lessons Learned
[What we learned and how to prevent in future]
```

**Distribution:** Post to #engineering, file in `docs/postmortems/`

---

## Communication Protocols

### Internal Communication

**Slack Channels:**
- `#incidents` - Active incident coordination (P1/P2)
- `#alerts` - Automated alerts (P2/P3)
- `#monitoring` - Informational alerts (P3/P4)
- `#engineering` - Post-mortems and updates

**Incident Channel Naming:**
- Format: `#incident-YYYYMMDD-short-description`
- Example: `#incident-20241222-api-errors`
- Archive after post-mortem completed

**Update Frequency:**
- P1: Every 15 minutes until resolved
- P2: Every 30 minutes until resolved
- P3: Hourly or as needed

---

### External Communication

**Status Page:**
- Update for all P1 incidents
- Update for P2 if user-facing
- Templates:
  - "Investigating: We are investigating [issue]"
  - "Identified: We have identified the issue and are working on a fix"
  - "Monitoring: A fix has been applied, monitoring for stability"
  - "Resolved: The issue has been resolved"

**Customer Communication:**
- P1: Email all affected users after resolution
- P2: Email if downtime >1 hour
- Include: Issue summary, impact, resolution, prevention

---

## On-Call Handoff

### Handoff Checklist

**Outgoing On-Call:**
1. Summarize week's incidents (P1/P2)
2. Note any ongoing issues
3. Share context on recent deployments
4. Review upcoming scheduled maintenance
5. Transfer any open incidents

**Incoming On-Call:**
1. Review last week's incidents
2. Check for any pending issues
3. Verify PagerDuty assignment
4. Test alert routing (send test page)
5. Review recent changes/deployments

**Handoff Meeting:**
- When: Every Monday 9:00 AM
- Duration: 15 minutes
- Attendees: Outgoing + Incoming + EM (optional)

---

## On-Call Tools Access

### Required Access

- **PagerDuty** - Alert receiving and incident management
- **Grafana** - Monitoring dashboards
- **Slack** - Communication
- **AWS Console** / **Server SSH** - Infrastructure access
- **Docker** - Container management
- **Redis CLI** - Database access
- **GitHub** - Code repository

### Credentials

- Stored in 1Password team vault
- Shared with on-call engineers only
- Rotate quarterly

---

## Escalation Contact Information

| Role | Primary Contact | Backup | Method |
|------|----------------|--------|---------|
| **Primary On-Call** | Per rotation | N/A | PagerDuty |
| **Secondary On-Call** | Per rotation | N/A | PagerDuty |
| **Engineering Manager** | [Name] | [Backup] | PagerDuty + Phone |
| **VP Engineering** | [Name] | [Backup] | Phone (P1 only) |
| **Security Team** | [Name] | [Email] | PagerDuty + Email |
| **Infrastructure** | [Name] | [Email] | Slack + Email |

**Emergency Contact List:** Stored in PagerDuty and 1Password

---

## War Room Procedures

### When to Declare War Room

- Any P1 incident lasting >30 minutes
- Multiple services affected
- Data loss suspected
- Security breach confirmed
- Decision needed on rollback or major change

### War Room Setup

1. **Zoom/Meet Link** - Posted in incident channel
2. **Roles:**
   - **Incident Commander** - Engineering Manager or Senior Engineer
   - **Ops Lead** - Primary on-call
   - **Communications Lead** - Posts updates to status page
   - **SMEs** - Subject matter experts as needed
3. **Ground Rules:**
   - Incident Commander has final say
   - One person talks at a time
   - Updates every 15 minutes
   - Screen share for visibility

---

## Common Scenarios and Playbooks

### Scenario 1: Orchestrator Service Down

**Symptoms:**
- Health check failing
- All API requests timing out
- Users cannot access system

**Immediate Actions:**
1. Check if container is running: `docker-compose ps`
2. Check logs: `docker-compose logs --tail=50 orchestrator`
3. Restart service: `docker-compose restart orchestrator`
4. If restart fails, check dependencies: Redis, disk space
5. Escalate if not resolved in 10 minutes

**See:** [Troubleshooting Runbook - Service Down](troubleshooting-runbook.md#service-down)

---

### Scenario 2: High Error Rate

**Symptoms:**
- >5% of API requests returning 500 errors
- Elevated failure rate in metrics

**Immediate Actions:**
1. Check error logs for pattern
2. Identify failing endpoint(s)
3. Check Redis connectivity
4. Check external service health (Ollama, n8n)
5. Consider rollback if after recent deployment

**See:** [Troubleshooting Runbook - High Error Rate](troubleshooting-runbook.md#high-error-rate)

---

### Scenario 3: Memory Leak

**Symptoms:**
- Memory usage climbing steadily
- Container restart needed frequently
- OOM kills in logs

**Immediate Actions:**
1. Restart affected service (immediate relief)
2. Monitor memory growth rate
3. Check for unclosed connections
4. Review recent code changes
5. Enable memory profiling
6. Escalate to engineering for code fix

**See:** [Troubleshooting Runbook - Memory Leak](troubleshooting-runbook.md#memory-leak)

---

### Scenario 4: Redis Down

**Symptoms:**
- All services reporting Redis connection errors
- Task data unavailable
- High latency on all endpoints

**Immediate Actions:**
1. Check Redis container: `docker-compose ps redis-stack`
2. Try connecting: `redis-cli PING`
3. Check Redis logs: `docker-compose logs redis-stack`
4. If down, restart: `docker-compose restart redis-stack`
5. Verify data integrity after restart
6. Check if backup restore needed

**See:** [Troubleshooting Runbook - Redis Issues](troubleshooting-runbook.md#redis-down)

---

## Runbook References

Detailed troubleshooting steps in:
- [Troubleshooting Runbook](troubleshooting-runbook.md)
- [Deployment Checklist](../DEPLOYMENT_CHECKLIST.md)
- [Backup and Restore](backup-restore.md)
- [Security Baseline](security-baseline.md)

---

## On-Call Best Practices

### Do's

✅ **Acknowledge alerts immediately** (even if you need more time)
✅ **Document everything** in incident channel
✅ **Ask for help early** - Don't struggle alone
✅ **Communicate status** regularly
✅ **Test fixes in staging** if possible
✅ **Write clear post-mortems**
✅ **Update runbooks** with new learnings

### Don'ts

❌ **Don't ignore alerts** - Acknowledge even if false positive
❌ **Don't make changes without backup** - Always have rollback plan
❌ **Don't skip post-mortems** - Even for "simple" fixes
❌ **Don't work on production while impaired** - Escalate if you can't focus
❌ **Don't delete data without confirmation** - Always double-check
❌ **Don't deploy during incidents** - Stabilize first

---

## Scheduled Maintenance Windows

**Preferred Window:** Sundays 2:00 AM - 6:00 AM (lowest traffic)

**Process:**
1. Schedule at least 1 week in advance
2. Post in #engineering for awareness
3. Update status page with planned maintenance
4. Notify affected users (if necessary)
5. Have rollback plan ready
6. Monitor post-deployment for 1 hour

**During On-Call:**
- Maintenance should not conflict with on-call shifts
- If urgent maintenance needed, coordinate with on-call engineer
- Engineering Manager approval required for off-schedule maintenance

---

## Review and Improvement

### Weekly Review (Monday Handoff)

- Incident count and trends
- False positive alerts (tune thresholds)
- New alerts needed
- Runbook gaps

### Monthly Review (First Monday)

- P1/P2 incident analysis
- Post-mortem action item status
- Escalation procedure effectiveness
- Tool and access issues

### Quarterly Review

- Full on-call process audit
- SLA adherence review
- Training needs assessment
- Runbook updates

---

## Training and Onboarding

### New On-Call Engineer Checklist

- [ ] Complete access setup (all tools)
- [ ] Shadow experienced on-call for 1 week
- [ ] Read all runbooks and procedures
- [ ] Practice common scenarios in staging
- [ ] Test alert routing (receive test page)
- [ ] Verify contact information
- [ ] Attend handoff meeting
- [ ] Schedule first rotation (2-4 weeks out)

### Quarterly Training

- Fire drill exercises
- New feature walkthroughs
- Security incident simulation
- Tool updates and changes

---

*Last Updated: December 22, 2024*
*On-Call Procedures Version: 1.0*
*Next Review: January 22, 2025*
