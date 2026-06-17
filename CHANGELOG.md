# Changelog

All notable changes to King AI v2 are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Security — 2026-06-15

Full security audit completed. All 8 critical/high findings remediated.

#### Critical
- **SEC-001**: Purged `private.key` and `request.csr` from git history — `filter-branch` + GC + orphan pack deletion. No secrets reachable from any ref.
- **SEC-002**: Replaced CORS `allow_origins=["*"]` with env-driven `settings.allowed_origins` (no wildcard + credentials violation).

#### High
- **SEC-003**: Added `AuthMiddleware` (JWT Bearer + X-API-Key) to all routes except `/api/health/*` and `/api/webhooks/*`.
- **SEC-004**: Moved Docker passwords to environment variable substitution (`${POSTGRES_PASSWORD:-default}`).
- **SEC-005**: Evolution engine file writes now require approved execution context (`_execution_approved` flag).
- **SEC-006**: Removed exposed host ports (`5432`, `6379`) from docker-compose; internal docker network only.

#### Medium
- **SEC-007**: Sandbox `Docker unavailable` now fails by default (`docker_sandbox_strict=true`).
- **SEC-008**: Implemented HMAC-SHA256 verification for Stripe, Shopify, and Plaid webhooks.

#### Additional Fixes
- 18 `hashlib.md5()` calls updated with `usedforsecurity=False`
- 5 SQL injection vectors parameterized (artifacts/store.py, database/migrations.py)
- HuggingFace model downloads pinned to `model_revision`
- Created missing `config/settings.py`
- Created `requirements.txt` (90 packages) and `requirements-dev.txt`
- Created `Dockerfile` (Python 3.11-slim, non-root user, healthcheck) and `.dockerignore`
- Extended `.gitignore` with secrets patterns (`*.pem`, `*.p12`, `*.pfx`, `*.jks`)
- Added `README.md`, `SECURITY.md`

**Bandit result**: 18 HIGH → **0 HIGH**.

See `AUDIT_REMEDIATION_REPORT.md` for full details.

---

## Tags

- `Added` for new features
- `Changed` for changes in existing functionality
- `Deprecated` for soon-to-be removed features
- `Removed` for now-removed features
- `Fixed` for any bug fixes
- `Security` for security-related changes
