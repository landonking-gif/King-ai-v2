# Security Policy

**Last updated**: 2026-06-15
**Audit status**: 8 critical/high findings fully remediated on 2026-06-15. See `AUDIT_REMEDIATION_REPORT.md`.

## Supported Versions

| Version | Supported |
|---------|-----------|
| master (latest commit) | ✅ Active |
| < 30 days old | ✅ Active |
| > 90 days old | ❌ End-of-life |

## Reporting a Vulnerability

**DO NOT open a public GitHub issue for security vulnerabilities.**

### Contact

- **Email**: security@landonking.dev (PGP key in `.pgp/public-key.asc` — to be added)
- **GitHub Security Advisories**: Use the private disclosure workflow at <https://github.com/landonking-gif/King-ai-v2/security/advisories/new>

Please include:
1. Description of the vulnerability
2. Reproduction steps (PoC preferred)
3. Impact assessment (data exposure, privilege escalation, etc.)
4. Affected versions

### Response Timeline

| Severity | Acknowledgment | Triage | Fix |
|----------|---------------|--------|-----|
| Critical (auth bypass, RCE) | 24 hours | 48 hours | 7 days |
| High (data leak) | 48 hours | 7 days | 30 days |
| Medium (info disclosure) | 7 days | 30 days | 90 days |
| Low (best practices) | 30 days | Next sprint | Next sprint |

## Security Architecture

### Trust Boundaries

1. **User → API**: Bearer token (JWT) or X-API-Key required for all routes except `/api/health/*` and `/api/webhooks/*`
2. **LLM → Sandbox**: All LLM-generated code runs in Docker container with `--network none` when `docker_sandbox_strict=true`
3. **External APIs (Stripe, Plaid, Shopify)**: HMAC signature verification mandatory if webhook secret is configured
4. **Evolution Engine → Filesystem**: Self-modification only allowed within `execute_proposal()` execution context; `_execution_approved` flag is enforced everywhere

### Privilege Model

| Component | Privilege | Notes |
|-----------|-----------|-------|
| Master AI brain | Read+write application code | MUST be approved via `ProposalStatus.APPROVED` before execution |
| Autonomous business engine | Spend money, manage integrations | Limited by `RISK_PROFILE` and `MAX_AUTO_APPROVE_AMOUNT` |
| External integrations | Webhook receipt | Verifies signatures; trusts no input by default |
| Audit trail | Read-only audit access | Stores all event types for forensic compliance |

### Cryptographic Standards

- **Hashing for security**: SHA-256, SHA-3, blake2b
- **Hashing for non-security** (caching, dedup): MD5 with `usedforsecurity=False`
- **Symmetric encryption**: ChaCha20-Poly1305 (when E2E is needed)
- **Asymmetric**: RSA-4096 (signing) / Ed25519 (signing)
- **JWT**: HS256 with `JWT_SECRET`; RS256 for distributed verification
- **API keys**: Stored as environment variables only; never logged, never in source

## Hardening Checklists

### Pre-Production

- [ ] Override `JWT_SECRET` from default (`change-me-in-production`)
- [ ] Set `ALLOWED_ORIGINS` to production domains only (no `*`)
- [ ] Set `STRIPE_WEBHOOK_SECRET`, `SHOPIFY_WEBHOOK_SECRET`, `PLAID_WEBHOOK_SECRET`
- [ ] Run `bandit -r src/` — must report 0 HIGH severities
- [ ] Run `pip-audit -r requirements.txt` — fix all advisories
- [ ] Verify `DOCKER_SANDBOX_STRICT=true` in production
- [ ] Confirm `docker-compose.yml` postgres/redis ports are NOT exposed to host
- [ ] Set `RISK_PROFILE=conservative` for high-value operations
- [ ] Set `ENABLE_AUTONOMOUS_MODE=false` initially; enable after review

### Runtime Monitoring

- [ ] All API access logged via structured logging with PII redaction
- [ ] Webhook signature failures trigger alerts (integrate with monitoring system)
- [ ] Failed authentication attempts tracked in audit trail
- [ ] Evolution proposal approval process is human-in-the-loop

### Incident Response

If you suspect an active breach:

1. Set `RISK_PROFILE=conservative` (or stop the service)
2. Review audit trail: `SELECT * FROM audit_trail WHERE timestamp > NOW() - INTERVAL '24 hours'`
3. Verify git history integrity: `git fsck --full`
4. Rotate all credentials starting with `JWT_SECRET`
5. Investigate via structured logs

## Acknowledgments

We thank the following for security research and audits:

- Trail of Bits (audit methodology)
- OpenSSF community (security tooling)
- All responsible disclosure reporters

---

**For full audit findings and remediation details, see `AUDIT_REMEDIATION_REPORT.md`.**
