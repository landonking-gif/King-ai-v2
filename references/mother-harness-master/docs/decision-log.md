# Decision Log

Track key architectural and product decisions for Mother-Harness.

## Template
| Date | Decision | Context | Status | Owners | Notes |
| --- | --- | --- | --- | --- | --- |
| YYYY-MM-DD | brief decision | what drove the decision | Proposed/Accepted/Deprecated | name(s) | optional links |

## Decisions
| Date | Decision | Context | Status | Owners | Notes |
| --- | --- | --- | --- | --- | --- |
| 2024-01-01 | Adopt three-tiered agent architecture (Mother Orchestrator + Specialist Agents + shared services) | Needed a scalable orchestration structure for research, coding, analysis, and design tasks | Accepted | Core team | See README overview |
| 2024-01-01 | Standardize on Redis Stack as the unified data store for tasks, vectors, sessions, and memory | Required a single operational database supporting vector search and session state | Accepted | Core team | Redis Stack called out in README |
| 2024-01-01 | Set default model roster with local gpt-oss:20b and cloud fallbacks per agent role | Needed consistent model selection defaults and cloud escalation options | Accepted | Core team | Agent roster + model selector |
| 2024-01-01 | Require decision log entries for new architectural choices | Ensure architectural decisions are documented alongside changes | Accepted | Core team | Policy: add a decision log entry with architectural changes |
