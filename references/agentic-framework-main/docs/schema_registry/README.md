# Schema Registry

This directory contains JSON Schema definitions (draft-07) for all typed artifacts used in the Agent Framework. All artifacts must be validated against their respective schemas before being persisted or passed between subagents.

## Overview

The schema registry enforces type safety, data validation, and interoperability across the agentic framework. Each schema defines the structure, required fields, data types, and validation rules for a specific artifact type.

## Schemas

### 1. Research Snippet (`research_snippet.json`)

**Purpose**: Represents research findings, text excerpts, or knowledge extracted by research subagents.

**Key Fields**:
- `id`: UUID v4 identifier
- `source`: Origin of research (URL, doc_id, type, author, etc.)
- `text`: Extracted content (up to 50,000 chars)
- `summary`: Concise summary (10-2,000 chars)
- `tags`: Classification keywords (max 20)
- `confidence`: Quality/relevance score (0.0-1.0)
- `provenance_refs`: References to provenance records
- `embedding_ref`: Vector database reference
- `safety_class`: Content classification (`public`, `internal`, `confidential`, `restricted`, `pii_risk`, `requires_review`)
- `created_by`: Subagent/system identifier
- `created_at`: ISO 8601 timestamp

**Use Cases**:
- Web research results
- Document extraction
- Knowledge base queries
- Multi-source research compilation

**Example**:
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "source": {
    "type": "web",
    "url": "https://arxiv.org/abs/1706.03762",
    "title": "Attention Is All You Need"
  },
  "text": "The Transformer is the first transduction model...",
  "summary": "Introduces the Transformer architecture...",
  "confidence": 0.95,
  "safety_class": "public",
  "created_by": "subagent:research-agent-001"
}
```

---

### 2. Claim Verification (`claim_verification.json`)

**Purpose**: Documents the verification of factual claims with evidence, confidence, and recommended actions.

**Key Fields**:
- `id`: UUID v4 identifier
- `claim_text`: The claim being verified (5-5,000 chars)
- `verdict`: Verification result (`true`, `false`, `partially_true`, `misleading`, `unverifiable`, `insufficient_evidence`, `requires_context`)
- `confidence`: Verdict confidence (0.0-1.0)
- `evidence_refs`: Array of supporting/refuting evidence
- `disagreement_notes`: Conflicts or nuances (up to 2,000 chars)
- `method`: Verification method (`multi_source_research`, `expert_consultation`, `data_analysis`, etc.)
- `verifier`: Subagent/system identifier
- `provenance`: Provenance record reference
- `action_suggestion`: Recommended action (`accept_as_is`, `add_context`, `flag_for_review`, etc.)

**Use Cases**:
- Fact-checking user claims
- Cross-referencing information
- Validating generated content
- Quality assurance workflows

**Example**:
```json
{
  "id": "650e8400-e29b-41d4-a716-446655440000",
  "claim_text": "Python 3.11 is faster than Python 3.10 by up to 60%",
  "verdict": "true",
  "confidence": 0.88,
  "evidence_refs": [
    {
      "type": "url",
      "ref": "https://www.python.org/downloads/release/python-3110/",
      "supporting": true
    }
  ],
  "method": "multi_source_research",
  "action_suggestion": "add_context"
}
```

---

### 3. Code Patch (`code_patch.json`)

**Purpose**: Represents code changes with files modified, tests, validation results, and approval status.

**Key Fields**:
- `id`: UUID v4 identifier
- `repo`: Repository identifier or URL
- `base_commit`: Git SHA (40 hex chars)
- `files_changed`: Array of file modifications (path, change_type, diff, lines added/removed)
- `patch_summary`: High-level description (10-2,000 chars)
- `tests`: Test cases with status, type, duration, coverage
- `confidence`: Patch correctness confidence (0.0-1.0)
- `risks`: Array of identified risks with severity, category, mitigation
- `authoring_subagent`: Code agent identifier
- `validation_results`: Linting, type checking, security scans, test coverage
- `provenance`: Provenance record reference
- `approved_by`: Human/system approver identifier
- `merge_ready`: Boolean indicating if ready to merge

**Use Cases**:
- Automated code generation
- Refactoring workflows
- Bug fixes
- Feature implementation with approval gates

**Example**:
```json
{
  "id": "750e8400-e29b-41d4-a716-446655440000",
  "repo": "github.com/company/agent-framework",
  "base_commit": "1234567890abcdef1234567890abcdef12345678",
  "files_changed": [
    {
      "path": "memory-service/service/compaction.py",
      "change_type": "modified",
      "lines_added": 45,
      "lines_removed": 12
    }
  ],
  "patch_summary": "Implement memory compaction strategy...",
  "confidence": 0.92,
  "merge_ready": true
}
```

---

### 4. Workflow Manifest (`manifest.json`)

**Purpose**: Defines agentic workflows with steps, memory management, tool access, and governance policies.

**Key Fields**:
- `manifest_id`: Unique manifest identifier (kebab-case)
- `name`: Human-readable workflow name
- `version`: Semantic version (e.g., "1.0.0")
- `steps`: Array of workflow steps with:
  - `id`, `role`, `agent`, `capabilities`
  - `inputs`, `outputs`, `timeout`, `retries`
  - `conditions` (skip_if, run_if)
- `memory`: Persistence triggers, compaction strategy, retention policies
- `tools`: MCP catalog IDs, scopes, rate limits, access mode
- `policies`: Human approval requirements, PII handling, cost limits, security level

**Use Cases**:
- Defining multi-step agentic workflows
- Configuring memory and tool access
- Setting governance policies
- Version-controlled workflow definitions

**Example**:
```yaml
# In YAML format (serialized from JSON Schema)
manifest_id: customer-support-workflow
name: Customer Support Ticket Resolution
version: 1.0.0
steps:
  - id: research-context
    role: research
    capabilities: [web_search, kb_query]
    timeout: 60
  - id: verify-claims
    role: verify
    timeout: 45
memory:
  compaction:
    strategy: summarize
    max_tokens: 8000
tools:
  catalog_ids: [filesystem, github]
policies:
  requires_human_approval: false
  pii_handling: redact
```

---

### 5. Provenance Record (`provenance.json`)

**Purpose**: Tracks the complete lineage, creation process, and audit trail of any artifact.

**Key Fields**:
- `id`: Provenance record ID (format: `prov_<uuid>`)
- `actor_id`: Creator identifier (format: `subagent:|system:|human:<id>`)
- `actor_type`: Actor type enum (`subagent`, `system`, `human`)
- `timestamp`: ISO 8601 creation timestamp
- `artifact_id`: Associated artifact identifier
- `artifact_type`: Type of artifact being tracked
- `inputs_hash`: SHA-256 hash of input data
- `outputs_hash`: SHA-256 hash of output artifact
- `tool_ids`: Array of tools/MCP servers used
- `manifest_id`: Workflow manifest identifier
- `version`: Manifest version
- `step_id`: Workflow step identifier
- `parent_provenance_ids`: Lineage chain
- `execution_context`: LLM model, tokens, cost, duration
- `validation`: Schema validation, safety checks, PII detection
- `mutation_history`: Audit trail of changes

**Use Cases**:
- Audit trails for compliance
- Debugging workflow failures
- Cost tracking and optimization
- Lineage tracking for data governance
- Reproducibility and transparency

**Example**:
```json
{
  "id": "prov_850e8400-e29b-41d4-a716-446655440000",
  "actor_id": "subagent:research-agent-001",
  "actor_type": "subagent",
  "timestamp": "2025-12-28T15:20:35Z",
  "artifact_id": "550e8400-e29b-41d4-a716-446655440000",
  "artifact_type": "research_snippet",
  "inputs_hash": "e3b0c442...",
  "outputs_hash": "d7a8fbb3...",
  "tool_ids": ["web_search", "document_read"],
  "manifest_id": "customer-support-workflow",
  "execution_context": {
    "llm_model": "claude-sonnet-4-20250514",
    "total_cost_usd": 0.042,
    "duration_ms": 3500
  }
}
```

---

## Schema Validation

### Python (Pydantic)

```python
from typing import Any
import json
from pydantic import BaseModel, ValidationError

def load_schema(schema_name: str) -> dict[str, Any]:
    """Load JSON Schema from registry."""
    with open(f"docs/schema_registry/{schema_name}.json") as f:
        return json.load(f)

def validate_artifact(artifact: dict[str, Any], schema_name: str) -> bool:
    """Validate artifact against schema."""
    from jsonschema import validate, ValidationError as JsonSchemaError

    schema = load_schema(schema_name)
    try:
        validate(instance=artifact, schema=schema)
        return True
    except JsonSchemaError as e:
        print(f"Validation error: {e.message}")
        return False

# Example usage
research_data = {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "source": {"type": "web", "url": "https://example.com"},
    "text": "Sample research text...",
    "summary": "Brief summary",
    "confidence": 0.95,
    "safety_class": "public",
    "created_by": "subagent:research-001",
    "created_at": "2025-12-28T10:00:00Z"
}

if validate_artifact(research_data, "research_snippet"):
    print("Artifact is valid!")
```

### TypeScript

```typescript
import Ajv from 'ajv';
import addFormats from 'ajv-formats';
import researchSnippetSchema from './research_snippet.json';

const ajv = new Ajv({ allErrors: true });
addFormats(ajv);

const validate = ajv.compile(researchSnippetSchema);

const artifact = {
  id: "550e8400-e29b-41d4-a716-446655440000",
  source: { type: "web", url: "https://example.com" },
  // ... other fields
};

if (validate(artifact)) {
  console.log("Artifact is valid!");
} else {
  console.error("Validation errors:", validate.errors);
}
```

---

## Schema Registry Patterns

### 1. Artifact ID Formats

- **Standard UUID**: `^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$`
- **Provenance ID**: `^prov_[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$`
- **Workflow Run ID**: `^run_[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$`
- **Session ID**: `^sess_[0-9a-zA-Z_-]+$`
- **Embedding Ref**: `^emb_[0-9a-zA-Z_-]+$`

### 2. Actor ID Format

Pattern: `^(subagent|system|human):[a-zA-Z0-9_@.-]+$`

Examples:
- `subagent:research-agent-001`
- `system:orchestrator`
- `human:developer@company.com`

### 3. Safety Classifications

- `public`: Publicly shareable information
- `internal`: Internal company use only
- `confidential`: Sensitive business information
- `restricted`: Highly restricted access
- `pii_risk`: Contains potential PII
- `requires_review`: Needs human review before use

### 4. Timestamp Format

All timestamps use ISO 8601 format: `YYYY-MM-DDTHH:mm:ssZ`

Example: `2025-12-28T15:20:35Z`

---

## Integration with Framework Components

### Memory Service

The Memory Service validates all artifacts against their schemas before persistence:

```python
from memory_service import MemoryClient

client = MemoryClient(redis_url="redis://localhost:6379")

# Validation happens automatically
artifact_id = client.store_artifact(
    artifact=research_snippet,
    artifact_type="research_snippet"
)
```

### Orchestrator

The Orchestrator validates workflow manifests on load:

```python
from orchestrator import WorkflowOrchestrator

orchestrator = WorkflowOrchestrator()

# Validates against manifest.json schema
workflow = orchestrator.load_manifest("manifests/customer-support.yaml")
```

### Code Executor

Skills produce typed outputs validated against schemas:

```python
from code_exec import SkillExecutor

executor = SkillExecutor()

result = executor.execute(
    skill="extract_entities",
    args={"text": "Sample text..."}
)

# Result is validated as research_snippet if configured
```

---

## Schema Versioning

Schemas follow semantic versioning in the `$id` field:

```json
{
  "$id": "https://agent-framework.io/schemas/research_snippet.json",
  "version": "1.0.0"
}
```

### Version Compatibility

- **Major version** (1.x.x → 2.x.x): Breaking changes, not backward compatible
- **Minor version** (1.0.x → 1.1.x): New optional fields, backward compatible
- **Patch version** (1.0.0 → 1.0.1): Bug fixes, clarifications

### Migration Strategy

When updating schemas:

1. Create new schema version file (e.g., `research_snippet_v2.json`)
2. Update manifest `version` field
3. Implement migration script for existing artifacts
4. Deprecate old schema after migration period

---

## Best Practices

### 1. Always Validate

Never persist or transmit artifacts without schema validation:

```python
# BAD
memory.store(artifact)

# GOOD
if validate_artifact(artifact, "research_snippet"):
    memory.store(artifact)
else:
    raise ValidationError("Invalid artifact")
```

### 2. Use Type Hints

Leverage Pydantic models generated from schemas:

```python
from pydantic import BaseModel
from typing import Literal

class ResearchSnippet(BaseModel):
    id: str
    source: dict
    text: str
    summary: str
    confidence: float
    safety_class: Literal["public", "internal", "confidential", "restricted"]
    created_by: str
    created_at: str
```

### 3. Log Validation Failures

Always log validation errors for debugging:

```python
import logging

logger = logging.getLogger(__name__)

try:
    validate(artifact, schema)
except ValidationError as e:
    logger.error(f"Validation failed: {e.message}", extra={
        "artifact_type": artifact_type,
        "artifact_id": artifact.get("id"),
        "errors": e.errors()
    })
    raise
```

### 4. Provenance Everything

Every artifact should have an associated provenance record:

```python
# Create artifact
artifact = create_research_snippet(...)

# Create provenance
provenance = create_provenance_record(
    artifact_id=artifact["id"],
    artifact_type="research_snippet",
    actor_id="subagent:research-001",
    inputs_hash=hash_inputs(inputs),
    outputs_hash=hash_artifact(artifact)
)

# Store both
memory.store_artifact(artifact)
memory.store_provenance(provenance)
```

---

## Schema Catalog

| Schema | File | Purpose | Required Fields |
|--------|------|---------|-----------------|
| Research Snippet | `research_snippet.json` | Research findings | id, source, text, summary, confidence, safety_class, created_by, created_at |
| Claim Verification | `claim_verification.json` | Fact-checking results | id, claim_text, verdict, confidence, method, verifier, created_at, provenance |
| Code Patch | `code_patch.json` | Code changes | id, repo, base_commit, files_changed, patch_summary, confidence, authoring_subagent, provenance, created_at, merge_ready |
| Manifest | `manifest.json` | Workflow definition | manifest_id, name, version, steps |
| Provenance | `provenance.json` | Audit trail | id, actor_id, actor_type, timestamp, artifact_id, artifact_type |

---

## Contributing

When adding new schemas:

1. Follow JSON Schema draft-07 specification
2. Include comprehensive `description` fields
3. Define `required` fields explicitly
4. Provide realistic `examples`
5. Use appropriate `pattern` for string validation
6. Set reasonable `minLength`, `maxLength`, `minimum`, `maximum` constraints
7. Include at least one complete example in the `examples` array
8. Update this README with schema documentation

### Schema Template

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "$id": "https://agent-framework.io/schemas/your_schema.json",
  "title": "Your Schema Title",
  "description": "Detailed description of purpose",
  "type": "object",
  "required": ["id", "other_required_fields"],
  "properties": {
    "id": {
      "type": "string",
      "description": "Unique identifier",
      "pattern": "^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
    }
  },
  "additionalProperties": false,
  "examples": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000"
    }
  ]
}
```

---

## License

These schemas are part of the Agent Framework project and follow the same license terms.

## Support

For questions or issues with schemas:
- Open an issue in the main repository
- Contact the architecture team
- Check the framework documentation at `/docs`
