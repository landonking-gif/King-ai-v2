# Launch Readiness Checklist

## Role Contracts & Agent Outputs
- [x] Keep existing role contracts and update agent outputs to match required outputs.
- [x] Critic outputs include `review_report` (approval + summary + positives + counts) and `issues_found` (issues array).
- [x] Skeptic outputs include `challenges` (map derived from concerns) while keeping `alternatives` as-is.
- [x] Ensure contract-required outputs are emitted by agents (critic: `review_report`, `issues_found`; skeptic: `challenges`, `alternatives`).

## Naming Clarity (Critic vs Skeptic)
- [x] Document the semantics: skeptic = critique/stress-test; critic = QA/review gate.
- [ ] Evaluate whether to rename agent type/file from `skeptic` → `critique` for long-term clarity.

## Capabilities Match Behavior
- [x] Remove `code_execution` from critic capabilities unless tool execution is wired.
- [x] Set `critic.requires_tool_calling` based on actual behavior.
- [x] Audit remaining roles to ensure `capabilities` and `requires_tool_calling` match behavior.

## Output Schema Consistency (Chaining-Friendly)
- [x] Standardize severity vocabulary across critic and skeptic outputs.
- [x] Align on a shared category/area field name across review outputs.
- [x] Align on a shared recommendation/mitigation field name across review outputs.
- [x] Add a shared adapter/type to normalize critic and skeptic outputs for downstream chaining.

## Contract Alignment (Do Not Postpone)
- [x] Ensure agent contracts reflect actual output fields and delivered artifacts.
- [x] Add validation tests that contracts reject missing required outputs.
- [x] Ensure agents emit contract-required outputs for `update`, `toolsmith`, `librarian`.

## Orchestration & Replay Validation
- [x] Core orchestration flows validated via integration tests (plan → execute → review → finalize).
- [x] Replay/timeline validated via per-run activity stream + replay retrieval.
