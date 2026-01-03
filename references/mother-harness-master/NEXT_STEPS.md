# Next Steps - Home Environment

**Created:** December 29, 2025  
**Context:** Code-only improvements completed remotely. This document outlines what needs to be done when back at the home development environment.

---

## 🔴 Immediate Actions (Do First)

### 1. Build All Packages
```bash
# Build in dependency order
pnpm --filter @mother-harness/shared build
pnpm --filter @mother-harness/agents build
pnpm --filter @mother-harness/orchestrator build
pnpm --filter @mother-harness/dashboard build
```

### 2. Run Full Test Suite
```bash
# Runs Vitest in non-watch mode across all packages
pnpm -r test -- --run
```

**Expected Results:**
- ✅ shared: 34 tests passing
- ✅ agents: 1 test passing  
- ⚠️ orchestrator: Some tests may fail until agents package is built

### 3. Start Docker Stack
```bash
# Docker Compose v2+ syntax (use 'docker-compose' for v1)
docker compose up -d
```

Verify all services start:
- [ ] Redis Stack (port 6379)
- [ ] Orchestrator (port 3000)
- [ ] Dashboard (port 3001)
- [ ] Docling (port 3002)

---

## 🟡 TypeScript Technical Debt (13 Occurrences across 9 Files)

All errors are `exactOptionalPropertyTypes` violations. Fix by adding `| undefined` to optional property types:

| File | Line | Property Issue |
|------|------|----------------|
| `src/robustness/budget-guard.ts` | 162, 168, 174, 180 | `warning: string \| undefined` not assignable to `warning?: string` |
| `src/scheduler/scheduler.ts` | 81, 153 | Multiple optional properties |
| `src/scheduler/focus-mode.ts` | 103 | `ends_at` property |
| `src/security/audit.ts` | 42 | `resource`, `metadata`, `status`, `ip`, `user_agent` |
| `src/pkm/decision-journal.ts` | 75 | `project_id`, `task_id` |
| `src/pkm/manager.ts` | 63 | `source_task_id` |
| `src/templates/library.ts` | 77 | `target_agents` |
| `src/enforcement/contract-enforcer.ts` | 216 | `last_error` |
| `src/tools/tool-executor.ts` | 33 | `error` property |

**Fix Pattern:**
```typescript
// Change interface from:
interface Example {
  optional?: string;
}

// To:
interface Example {
  optional?: string | undefined;
}
```

---

## 🟢 Integration Testing Checklist

### Redis Operations
- [ ] Verify Redis Stack modules loaded (JSON, Search)
- [ ] Test `FT.CREATE` index creation
- [ ] Test `JSON.SET` / `JSON.GET` operations
- [ ] Verify vector similarity search works

### End-to-End Flow
```bash
# Create a test run via API
curl -X POST http://localhost:3000/api/runs \
  -H "Content-Type: application/json" \
  -d '{"user_id": "test-user", "query": "Research TypeScript best practices"}'
```

- [ ] Run creates successfully
- [ ] Task planning generates steps
- [ ] Researcher agent executes
- [ ] Critic review completes
- [ ] Memory tiers populate

### Approval Workflow
- [ ] High-risk steps pause for approval
- [ ] Dashboard shows pending approvals
- [ ] Approval/rejection flows work
- [ ] Timeout handling works

---

## 📋 Pre-Launch Checklist

### Security
- [ ] Run `pnpm audit` for vulnerabilities
- [ ] Scan for secrets: `npx secretlint "**/*"` or review with `git secrets --scan`
- [ ] Check `.env.example` has no real values
- [ ] Verify Redis AUTH is configured for production

### Performance
- [ ] Run basic load test (10 concurrent runs)
- [ ] Check memory usage under load
- [ ] Verify Redis connection pooling

### Documentation
- [ ] API endpoints documented
- [ ] Deployment runbook complete
- [ ] Troubleshooting guide exists

---

## 📁 Files Modified This Session

**Shared Package:**
- `services/shared/src/redis/client.ts` - Fixed ioredis imports
- `services/shared/src/redis/indexes.ts` - Fixed type imports
- `services/shared/src/testing/test-utils.ts` - jest → vitest mocks
- `services/shared/src/tools/executor.ts` - Removed unused import
- `services/shared/src/enforcement/contract-enforcer.ts` - Removed unused import
- `services/shared/src/validation/schemas.test.ts` - Fixed ApprovalSchema test

**Orchestrator Package:**
- `services/orchestrator/src/orchestrator.ts` - Added missing imports/members
- `services/orchestrator/src/orchestrator.test.ts` - Added mocks, created tests
- `services/orchestrator/src/planner.unit.test.ts` - Fixed type assertions
- `services/orchestrator/src/approval-workflow.ts` - Created workflow guide

**Documentation:**
- `docs/progress-report-2025-12-29.md` - Session progress
- `docs/launch-readiness.md` - Updated status

---

## 🎯 Priority Order

1. **Build packages** → unblocks all testing
2. **Start Docker** → enables integration tests  
3. **Run integration tests** → validates runtime behavior
4. **Fix TypeScript errors** → enables clean builds
5. **Security audit** → pre-launch requirement
6. **Load testing** → production readiness

---

*Delete this file after completing the checklist.*
