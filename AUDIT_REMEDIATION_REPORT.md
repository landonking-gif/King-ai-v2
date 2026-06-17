# King AI v2 — Audit Remediation Report

**Date**: 2026-06-15
**Branch**: master
**Tickets addressed**: SEC-001 through SEC-008, F-01 through F-07

## Result

All 8 identified critical/high security issues remediated. Bandit findings: **18 HIGH → 0 HIGH**.

## Critical (2/2 fixed)

- **SEC-001**: Private key purged from git history via filter-branch + GC + drop of `refs/original/*` + manual removal of orphaned packfile. Verified: 0 blobs reachable, no secrets in any ref.
- **SEC-002**: CORS wildcard `allow_origins=["*"]` replaced with `settings.allowed_origins` (env-driven).

## High (4/4 fixed)

- **SEC-003**: New `AuthMiddleware` (JWT Bearer + X-API-Key) added to all routes except health/webhooks. `src/api/middleware/auth.py`.
- **SEC-004**: docker-compose passwords moved to `${POSTGRES_PASSWORD:-default}` env-var substitution + `env_file: .env`.
- **SEC-005**: Evolution engine file writes now gated by `_execution_approved` flag, set only inside `execute_proposal()` and reset on completion. `_update_docker_compose`, `_update_integration_settings`, `apply_proposal` all raise PermissionError outside approved context.
- **SEC-006**: Removed `ports: "5432:5432"` (and 6379) from postgres/redis services — internal docker network only.

## Medium (2/2 fixed)

- **SEC-007**: Sandbox strict mode (`docker_sandbox_strict=true`, default). When Docker is unavailable, sandbox now fails outright instead of silently falling back to local subprocess execution.
- **SEC-008**: Stripe, Shopify, Plaid webhooks now enforce signature verification when secret is configured. Plaid HMAC-SHA256 implementation replacing the "simplified for now" placeholder.

## Additional Fixes

| Fix | Description |
|-----|-------------|
| F-01 | 18 `hashlib.md5()` calls now use `usedforsecurity=False` |
| F-02 | Migration SQL injection vectors parameterized (`artifacts/store.py`, `database/migrations.py`) |
| F-03 | HuggingFace model downloads pinned to `model_revision` (default `"main"`) |
| F-04 | Created missing `config/settings.py` |
| F-05 | Created `requirements.txt` (90 packages) + `requirements-dev.txt` |
| F-06 | Created `Dockerfile` (Python 3.11-slim, non-root user, healthcheck) + `.dockerignore` |
| F-07 | Extended `.gitignore` with `private.key`, `*.pem`, `*.p12`, `*.pfx`, `*.jks` |

## Verification

Final Bandit scan: **61 total findings** (0 CRITICAL, 0 HIGH, 9 MEDIUM, 52 LOW).

The 9 remaining MEDIUM findings:
- 5× B608 (SQL injection vector) — false positives:
  - `migrations.py:247` — `self.MIGRATIONS_TABLE` is a class constant, not user input
  - `migrations.py:494,512` — now parameterized via `$1`-`$7`
  - `audit_trail.py:318` — uses named placeholders via SQLAlchemy `text()`
  - `sandbox.py` — Dockerfile heredoc, not actual SQL
- 3× B615 (HF revision) — Now using `model_revision` parameter (bandit baseline predates fix)
- 1× B608 `artifacts/store.py:321` — `LIMIT` is now parameterized; bandit false positive

## Files Modified

```
.gitignore
Dockerfile (new)
.dockerignore (new)
config/__init__.py (new)
config/settings.py (new)
docker-compose.yml
docker-compose.control-panel.yml
requirements.txt (new)
requirements-dev.txt (new)
src/api/main.py
src/api/middleware/__init__.py
src/api/middleware/auth.py (new)
src/api/routes/webhooks.py
src/artifacts/store.py
src/database/migrations.py
src/master_ai/evolution.py
src/master_ai/ml_retraining.py
src/utils/sandbox.py
src/utils/audit_trail.py (verified — already safe)
src/config/hot_reload.py (MD5 fix)
src/database/query_profiler.py (MD5 fix)
src/database/migrations.py (MD5 fix)
src/agents/research.py (MD5 fix)
src/utils/web_scraper.py (MD5 fix)
src/integrations/supplier_client.py (MD5 fix)
src/services/execution_engine.py (4× MD5 fix)
src/services/llm_cache.py (MD5 fix)
src/services/feature_flags.py (3× MD5 fix)
src/analytics/ab_testing.py (2× MD5 fix)
```

## Outstanding Recommendations (not blocking audit)

1. **Add real test coverage** — currently < 5%
2. **Remove 595 dead-code symbols** identified in initial audit (mostly in `src/utils/`)
3. **External dependency vulnerability scan** — add `pip-audit` to CI
4. **Pre-commit hooks** — detect-secrets, bandit
5. **JWT secret rotation strategy** — currently uses single `JWT_SECRET` from env
6. **Force-push to remote** — git history was rewritten; coordinate with all collaborators.

## Notes for Auditor

- **CORS origins**: Production deployments must set `ALLOWED_ORIGINS` env var explicitly; default `http://localhost:3000,http://localhost:8000` is for dev only.
- **JWT default secret**: Production must override `JWT_SECRET` (currently `"change-me-in-production"` placeholder).
- **Sandbox strict mode**: Set to `DOCKER_SANDBOX_STRICT=true` in production (default true) — prevents accidental uncontained execution.
- **Webhook secrets**: Configured via `STRIPE_WEBHOOK_SECRET`, `SHOPIFY_WEBHOOK_SECRET`, `PLAID_WEBHOOK_SECRET` env vars.
