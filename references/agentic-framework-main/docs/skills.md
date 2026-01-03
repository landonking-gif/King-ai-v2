# Skills Development Guide

Complete guide to creating, testing, and distributing custom skills for the Agentic Framework.

## Table of Contents

- [Overview](#overview)
- [Skill Formats](#skill-formats)
- [Creating Native Skills](#creating-native-skills)
- [Creating Anthropic Skills](#creating-anthropic-skills)
- [Hybrid Skills](#hybrid-skills)
- [Skill Registry](#skill-registry)
- [Testing Skills](#testing-skills)
- [Safety and Approval](#safety-and-approval)
- [Distribution](#distribution)
- [Best Practices](#best-practices)
- [Advanced Topics](#advanced-topics)

## Overview

Skills are deterministic, reusable functions that agents can invoke to perform specific tasks. Unlike LLM prompts, skills execute as Python code in sandboxed environments with validated inputs and outputs.

### Key Characteristics

- **Deterministic**: Same inputs always produce same outputs
- **Type-Safe**: JSON Schema validation for all I/O
- **Sandboxed**: Execution isolated from main process
- **Versioned**: Semantic versioning for compatibility
- **Auditable**: Full execution logs and provenance tracking

### When to Use Skills

**Use Skills For:**
- Data transformation (summarization, parsing, formatting)
- Computation (calculations, statistical analysis)
- External API calls (with safety flags)
- File operations (with approval policies)
- Deterministic NLP tasks (entity extraction, sentiment analysis)

**Don't Use Skills For:**
- Tasks better suited for LLM reasoning
- One-off operations (write inline code instead)
- Tasks requiring real-time user interaction

## Skill Formats

The framework supports two skill formats, plus a hybrid approach:

### Format Comparison

| Feature | Native Format | Anthropic Format | Hybrid Format |
|---------|--------------|------------------|---------------|
| **Files** | skill.yaml + schema.json + handler.py | SKILL.md (Markdown + YAML frontmatter) | Both |
| **Execution** | Python function | Instruction-based (LLM interprets) | Python function |
| **Validation** | Strict JSON Schema | Loose (Anthropic limits) | Strict JSON Schema |
| **Marketplace** | ❌ Not compatible | ✅ Compatible | ✅ Compatible |
| **Safety Flags** | ✅ Explicit flags | ⚠️ Manual review | ✅ Explicit flags |
| **Versioning** | Semver (required) | Optional | Semver |
| **Progressive Loading** | ❌ Eager loading | ✅ 3-level disclosure | ✅ 3-level disclosure |
| **Best For** | Internal framework use | Sharing on Anthropic marketplace | Both internal + marketplace |

**Recommendation:** Use **Hybrid format** for maximum compatibility and control.

## Creating Native Skills

Native skills consist of three required files:

```
my_skill/
├── skill.yaml          # Metadata and configuration
├── schema.json         # JSON Schema for I/O validation
└── handler.py          # Python implementation
```

### 1. skill.yaml (Metadata)

Define skill metadata and configuration:

```yaml
name: text_summarize
version: "1.0.0"
description: Summarize text in various styles (concise, bullet-points, paragraph)

# Safety classification
safety_flags:
  - none  # Options: none, file_system, network_access, pii_risk, side_effect

# Approval requirement
requires_approval: false

# Handler function reference
handler: handler.summarize

# Input/output schema references
inputs_schema: schema.json#/definitions/input
outputs_schema: schema.json#/definitions/output

# Optional: Dependencies
dependencies:
  - nltk>=3.8
  - spacy>=3.5

# Optional: Resource limits
limits:
  timeout_seconds: 30
  max_memory_mb: 512
  max_cpu_percent: 50

# Optional: Tags for discovery
tags:
  - nlp
  - text-processing
  - summarization

# Optional: Anthropic compatibility
anthropic_compatible: true  # Enables dual-format support
```

### 2. schema.json (Validation)

Define strict JSON Schema for inputs and outputs:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "definitions": {
    "input": {
      "type": "object",
      "properties": {
        "text": {
          "type": "string",
          "description": "Text to summarize",
          "minLength": 10,
          "maxLength": 50000
        },
        "style": {
          "type": "string",
          "enum": ["concise", "bullet-points", "paragraph"],
          "default": "concise",
          "description": "Summarization style"
        },
        "max_sentences": {
          "type": "integer",
          "minimum": 1,
          "maximum": 10,
          "default": 3,
          "description": "Maximum number of sentences in summary"
        },
        "preserve_entities": {
          "type": "boolean",
          "default": true,
          "description": "Preserve named entities in summary"
        }
      },
      "required": ["text"],
      "additionalProperties": false
    },
    "output": {
      "type": "object",
      "properties": {
        "summary": {
          "type": "string",
          "description": "Generated summary"
        },
        "word_count": {
          "type": "integer",
          "description": "Number of words in summary"
        },
        "compression_ratio": {
          "type": "number",
          "minimum": 0,
          "maximum": 1,
          "description": "Ratio of summary length to original length"
        },
        "extracted_entities": {
          "type": "array",
          "items": {
            "type": "string"
          },
          "description": "Named entities preserved in summary"
        }
      },
      "required": ["summary", "word_count", "compression_ratio"],
      "additionalProperties": false
    }
  }
}
```

### 3. handler.py (Implementation)

Implement the skill as a Python function:

```python
"""
Text Summarization Skill

Provides deterministic text summarization in multiple styles.
"""

from typing import Dict, Any, List
import re


def summarize(
    text: str,
    style: str = "concise",
    max_sentences: int = 3,
    preserve_entities: bool = True
) -> Dict[str, Any]:
    """
    Summarize text in specified style.

    Args:
        text: Input text to summarize
        style: Summarization style (concise, bullet-points, paragraph)
        max_sentences: Maximum sentences in summary
        preserve_entities: Whether to preserve named entities

    Returns:
        Dictionary with summary, word_count, compression_ratio, extracted_entities
    """
    # Input validation (redundant with JSON Schema, but defensive)
    if not text or len(text) < 10:
        raise ValueError("Text must be at least 10 characters")

    if style not in ["concise", "bullet-points", "paragraph"]:
        raise ValueError(f"Invalid style: {style}")

    # Extract sentences
    sentences = _extract_sentences(text)

    # Extract named entities if requested
    entities = []
    if preserve_entities:
        entities = _extract_entities(text)

    # Select key sentences (simple extraction algorithm)
    key_sentences = _select_key_sentences(sentences, max_sentences, entities)

    # Format based on style
    if style == "bullet-points":
        summary = "\n".join(f"• {s}" for s in key_sentences)
    elif style == "paragraph":
        summary = " ".join(key_sentences)
    else:  # concise
        summary = ". ".join(key_sentences) + "."

    # Calculate metrics
    original_words = len(text.split())
    summary_words = len(summary.split())
    compression_ratio = summary_words / original_words if original_words > 0 else 0

    return {
        "summary": summary,
        "word_count": summary_words,
        "compression_ratio": round(compression_ratio, 3),
        "extracted_entities": entities
    }


def _extract_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    # Simple sentence splitting (production would use nltk.sent_tokenize)
    sentences = re.split(r'[.!?]+\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 0]


def _extract_entities(text: str) -> List[str]:
    """Extract named entities (simplified)."""
    # Production would use spaCy or similar
    # This is a simple regex-based approach for demonstration
    entity_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
    entities = re.findall(entity_pattern, text)
    return list(set(entities))[:10]  # Limit to 10 unique entities


def _select_key_sentences(
    sentences: List[str],
    max_count: int,
    entities: List[str]
) -> List[str]:
    """Select most important sentences."""
    # Score sentences by length and entity presence
    scored = []
    for sent in sentences:
        score = len(sent.split())  # Basic: longer = more info
        # Bonus for containing entities
        for entity in entities:
            if entity in sent:
                score += 10
        scored.append((score, sent))

    # Sort by score and take top N
    scored.sort(reverse=True, key=lambda x: x[0])
    return [sent for score, sent in scored[:max_count]]
```

### 4. test_handler.py (Unit Tests)

Write comprehensive tests:

```python
"""
Unit tests for text_summarize skill
"""

import pytest
from handler import summarize


def test_concise_style():
    """Test concise summarization."""
    text = "This is sentence one. This is sentence two. This is sentence three."
    result = summarize(text, style="concise", max_sentences=2)

    assert "summary" in result
    assert result["word_count"] > 0
    assert 0 < result["compression_ratio"] <= 1
    assert isinstance(result["extracted_entities"], list)


def test_bullet_points_style():
    """Test bullet-point summarization."""
    text = "First point here. Second point here. Third point here."
    result = summarize(text, style="bullet-points", max_sentences=2)

    assert "•" in result["summary"]
    assert result["summary"].count("•") <= 2


def test_paragraph_style():
    """Test paragraph summarization."""
    text = "Sentence one. Sentence two. Sentence three."
    result = summarize(text, style="paragraph", max_sentences=3)

    assert "\n" not in result["summary"]  # No newlines in paragraph
    assert "•" not in result["summary"]  # No bullets


def test_entity_preservation():
    """Test named entity extraction."""
    text = "John Smith works at Microsoft in Seattle. Mary Johnson is the CEO."
    result = summarize(text, preserve_entities=True)

    # Should extract some entities
    assert len(result["extracted_entities"]) > 0
    # Common entities should be present
    entities_str = " ".join(result["extracted_entities"])
    assert any(name in entities_str for name in ["John", "Smith", "Microsoft", "Seattle"])


def test_compression_ratio():
    """Test compression ratio calculation."""
    short_text = "Short text here."
    long_text = "This is a much longer text with many sentences. " * 10

    short_result = summarize(short_text, max_sentences=1)
    long_result = summarize(long_text, max_sentences=3)

    # Longer text should have better compression
    assert long_result["compression_ratio"] < short_result["compression_ratio"]


def test_max_sentences_limit():
    """Test max_sentences parameter."""
    text = ". ".join([f"Sentence {i}" for i in range(20)])

    result = summarize(text, max_sentences=5)

    # Count sentences in output
    sentence_count = result["summary"].count(".")
    assert sentence_count <= 5


def test_input_validation():
    """Test input validation."""
    # Too short
    with pytest.raises(ValueError):
        summarize("Short")

    # Invalid style
    with pytest.raises(ValueError):
        summarize("Valid text here", style="invalid_style")


def test_output_schema():
    """Test output matches schema."""
    text = "Valid input text for testing schema compliance."
    result = summarize(text)

    # Required fields
    assert "summary" in result
    assert "word_count" in result
    assert "compression_ratio" in result
    assert "extracted_entities" in result

    # Type checking
    assert isinstance(result["summary"], str)
    assert isinstance(result["word_count"], int)
    assert isinstance(result["compression_ratio"], float)
    assert isinstance(result["extracted_entities"], list)

    # Value constraints
    assert 0 <= result["compression_ratio"] <= 1
    assert result["word_count"] >= 0
```

## Creating Anthropic Skills

Anthropic skills use a single Markdown file with YAML frontmatter:

```
my_skill/
└── SKILL.md            # All-in-one: metadata + documentation + examples
```

### SKILL.md Format

```markdown
---
name: pdf-form-filler
description: Fill PDF forms with structured data
version: 1.2.0
dependencies: python>=3.8, pypdf>=3.0
safety: This skill reads and writes PDF files to the local filesystem.
---

# PDF Form Filler

This skill fills PDF forms with structured data.

## How it works

The skill:
1. Reads the PDF form template
2. Maps input data to form fields
3. Fills the form fields
4. Saves the completed PDF

## Input Schema

- **template_path** (string, required): Path to PDF template file
- **data** (object, required): Field name → value mapping
- **output_path** (string, required): Where to save filled PDF

Example:
```json
{
  "template_path": "/path/to/template.pdf",
  "data": {
    "name": "John Doe",
    "date": "2024-01-15",
    "amount": "1000.00"
  },
  "output_path": "/path/to/output.pdf"
}
```

## Output Schema

- **status** (string): "success" or "error"
- **output_path** (string): Path to filled PDF
- **fields_filled** (integer): Number of fields populated

Example:
```json
{
  "status": "success",
  "output_path": "/tmp/filled_form.pdf",
  "fields_filled": 15
}
```

## Safety Considerations

- Requires file system access (read + write)
- Only accesses paths explicitly provided in inputs
- Does not make network requests
- Does not execute embedded JavaScript in PDFs

## Examples

### Example 1: Simple Form

Fill a basic contact form:

```json
{
  "template_path": "templates/contact_form.pdf",
  "data": {
    "full_name": "Jane Smith",
    "email": "jane@example.com",
    "phone": "+1-555-0123"
  },
  "output_path": "output/contact_filled.pdf"
}
```

### Example 2: Tax Form

Fill a tax form with calculations:

```json
{
  "template_path": "templates/tax_form.pdf",
  "data": {
    "taxpayer_name": "John Doe",
    "ssn": "XXX-XX-1234",
    "gross_income": "75000",
    "deductions": "12000",
    "tax_owed": "9450"
  },
  "output_path": "output/tax_2024.pdf"
}
```

## Limitations

- Maximum file size: 50MB
- Supports PDF 1.4 - 1.7 formats
- Does not support password-protected PDFs
- Form field names must match exactly (case-sensitive)

## Version History

- **1.2.0** (2024-01-15): Added support for calculated fields
- **1.1.0** (2024-01-01): Multi-page form support
- **1.0.0** (2023-12-01): Initial release
```

### Progressive Disclosure

Anthropic skills use 3-level progressive disclosure:

**Level 1 - YAML Frontmatter:**
- Name, description, version
- Shown in skill catalog listing
- ~50 characters

**Level 2 - Description Section:**
- How it works overview
- Input/output schemas (simplified)
- Shown on skill detail page
- ~500 characters

**Level 3 - Full Documentation:**
- Complete examples
- Safety considerations
- Limitations
- Version history
- Shown when skill is invoked

## Hybrid Skills

Combine both formats for maximum compatibility:

```
my_skill/
├── SKILL.md           # Anthropic marketplace format
├── skill.yaml         # Native metadata
├── schema.json        # Strict validation schemas
└── handler.py         # Deterministic implementation
```

### Benefits

1. **Anthropic Marketplace**: Shareable on marketplace
2. **Deterministic Execution**: Uses Python handler (not LLM interpretation)
3. **Strict Validation**: JSON Schema enforcement
4. **Safety Flags**: Explicit safety classification
5. **Progressive Loading**: JIT loading for better performance

### Creating Hybrid Skills

Use `kautilya` to create hybrid skills:

```bash
kautilya skill new my-skill --format hybrid
```

Or convert existing skills:

```bash
# Add Anthropic format to native skill
kautilya skill convert my-skill --to anthropic --include-handler

# Add native format to Anthropic skill
kautilya skill convert my-skill --to native
```

## Skill Registry

Skills are registered in the framework's skill registry for discovery and loading.

### Registration

**Automatic Registration:**
Skills in `code-exec/skills/` are auto-registered on startup.

**Manual Registration:**
```python
from code_exec.service.registry import SkillRegistry

registry = SkillRegistry()
await registry.register_skill(
    skill_path="/path/to/my_skill",
    format="hybrid"  # or "native" or "anthropic"
)
```

### Discovery

List available skills:

```python
skills = await registry.list_skills(
    tags=["nlp"],
    safety_flags=["none"],
    format="hybrid"
)

for skill in skills:
    print(f"{skill.name} v{skill.version} ({skill.format})")
```

### JIT Loading

Skills load lazily to reduce memory footprint:

```python
# Skill not loaded yet
skill_info = await registry.get_skill_info("text_summarize")

# Load on first use
skill = await registry.load_skill("text_summarize")

# Execute
result = await skill.execute({"text": "..."})

# Unload to free memory
await registry.unload_skill("text_summarize")
```

## Testing Skills

### Unit Testing

Write tests using pytest:

```bash
# Run all tests
pytest code-exec/skills/my_skill/test_handler.py -v

# Run specific test
pytest code-exec/skills/my_skill/test_handler.py -k test_input_validation

# With coverage
pytest --cov=handler code-exec/skills/my_skill/test_handler.py
```

### Integration Testing

Test skill in sandboxed executor:

```python
from code_exec.service.executor import SkillExecutor
import asyncio

async def test_skill_execution():
    executor = SkillExecutor()

    result = await executor.execute_skill(
        skill_name="text_summarize",
        inputs={
            "text": "Test article text here...",
            "style": "bullet-points",
            "max_sentences": 3
        },
        timeout=30
    )

    assert result.status == "success"
    assert "summary" in result.output
    assert result.execution_stats.duration_ms < 5000

asyncio.run(test_skill_execution())
```

### Validation Testing

Test schema validation:

```python
from code_exec.service.validator import SkillValidator

validator = SkillValidator()

# Validate skill structure
issues = validator.validate_skill_structure("my_skill/")
assert len(issues) == 0, f"Validation errors: {issues}"

# Validate inputs against schema
validator.validate_inputs(
    skill_name="text_summarize",
    inputs={"text": "Valid input", "style": "concise"}
)

# This should fail
with pytest.raises(ValidationError):
    validator.validate_inputs(
        skill_name="text_summarize",
        inputs={"style": "invalid_style"}  # Missing required 'text'
    )
```

## Safety and Approval

### Safety Flags

Classify skills by what they access:

| Flag | Description | Examples |
|------|-------------|----------|
| `none` | Read-only, no external access | Text processing, calculations |
| `file_system` | Reads/writes local files | PDF generation, file parsing |
| `network_access` | Makes HTTP requests | API calls, web scraping |
| `pii_risk` | May handle sensitive data | Data anonymization, entity extraction |
| `side_effect` | Modifies external state | Database writes, email sending |

**Multiple flags:**
```yaml
safety_flags:
  - file_system
  - pii_risk
```

### Approval Policies

Require human approval for sensitive operations:

```yaml
# In skill.yaml
requires_approval: true

approval_policy:
  required_for:
    - operation: file_write
      paths: ["/production/*"]
    - operation: api_call
      domains: ["*.internal.company.com"]

  approvers:
    - role: engineering_lead
    - email: security@company.com

  timeout_seconds: 3600  # 1 hour
  on_timeout: reject
```

**Approval Workflow:**

1. Skill requests approval before execution
2. Notification sent to approvers
3. Approver reviews skill inputs and potential impact
4. Approver approves/rejects via API or UI
5. Skill executes (if approved) or fails (if rejected/timeout)

## Distribution

### Packaging Skills

Create distributable ZIP:

```bash
# Package native skill
kautilya skill package my-skill --output my-skill-v1.0.0.zip

# Package for Anthropic marketplace
kautilya skill package my-skill --format anthropic --output my-skill-anthropic.zip
```

### Sharing on Anthropic Marketplace

1. **Export skill:**
```bash
kautilya skill export my-skill --format anthropic
```

2. **Validate marketplace requirements:**
- SKILL.md with complete frontmatter
- Clear input/output schemas
- Safety considerations documented
- Examples provided
- Version history

3. **Submit to marketplace:**
- Upload to https://marketplace.anthropic.com
- Provide description and tags
- Set visibility (public/private)

### Installing Skills

**From ZIP:**
```bash
kautilya skill import my-skill-v1.0.0.zip
```

**From URL:**
```bash
kautilya skill import https://example.com/skills/my-skill.zip
```

**From Anthropic Marketplace:**
```bash
kautilya skill import anthropic://pdf-form-filler@1.2.0
```

## Best Practices

### 1. Naming Conventions

```yaml
# Good names (verb_noun or descriptive)
text_summarize
extract_entities
analyze_sentiment
generate_report

# Bad names
skill1
processor
handler
util
```

### 2. Versioning

Use semantic versioning:

```yaml
# Breaking change (incompatible API)
1.0.0 → 2.0.0

# New feature (backward compatible)
1.0.0 → 1.1.0

# Bug fix
1.0.0 → 1.0.1
```

### 3. Schema Design

**Be specific:**
```json
{
  "text": {
    "type": "string",
    "minLength": 10,        // Prevent empty inputs
    "maxLength": 50000,     // Prevent OOM
    "pattern": "^[\\w\\s.,!?-]+$"  // Constrain format
  }
}
```

**Provide defaults:**
```json
{
  "max_sentences": {
    "type": "integer",
    "default": 3,           // Sensible default
    "minimum": 1,
    "maximum": 10
  }
}
```

### 4. Error Handling

```python
def my_skill(input_data: str) -> Dict[str, Any]:
    try:
        # Main logic
        result = process(input_data)
        return {"status": "success", "result": result}

    except ValueError as e:
        # User error (bad input)
        raise ValueError(f"Invalid input: {e}")

    except ExternalAPIError as e:
        # Transient error (retry)
        raise RetryableError(f"API unavailable: {e}")

    except Exception as e:
        # Unexpected error (don't retry)
        raise SkillExecutionError(f"Unexpected error: {e}")
```

### 5. Logging

```python
import logging

logger = logging.getLogger(__name__)

def my_skill(text: str) -> Dict[str, Any]:
    logger.info(f"Processing text of length {len(text)}")

    # Don't log sensitive data
    # BAD: logger.debug(f"Processing: {text}")

    result = process(text)

    logger.info(f"Generated {result['word_count']} word summary")

    return result
```

### 6. Resource Management

```python
def file_processing_skill(file_path: str) -> Dict[str, Any]:
    # Use context managers
    with open(file_path, 'r') as f:
        data = f.read()

    # Process in chunks for large files
    if len(data) > 1_000_000:
        return process_in_chunks(data)

    return process(data)
```

## Advanced Topics

### Custom Validators

Add custom validation logic beyond JSON Schema:

```python
from code_exec.service.validator import SkillValidator

class CustomValidator(SkillValidator):
    def validate_custom(self, skill_name: str, inputs: Dict[str, Any]) -> None:
        """Custom validation logic."""
        if skill_name == "text_summarize":
            text = inputs.get("text", "")

            # Check for profanity (example)
            if self._contains_profanity(text):
                raise ValidationError("Text contains inappropriate content")

            # Check language
            if not self._is_english(text):
                raise ValidationError("Text must be in English")

    def _contains_profanity(self, text: str) -> bool:
        # Implementation here
        return False

    def _is_english(self, text: str) -> bool:
        # Implementation here
        return True
```

### Streaming Outputs

For long-running skills, stream outputs:

```python
from typing import AsyncIterator

async def long_running_skill(large_file: str) -> AsyncIterator[Dict[str, Any]]:
    """Process large file and stream results."""
    total_lines = count_lines(large_file)

    with open(large_file) as f:
        for i, line in enumerate(f):
            result = process_line(line)

            # Yield intermediate result
            yield {
                "progress": (i + 1) / total_lines,
                "partial_result": result
            }

    # Final result
    yield {
        "progress": 1.0,
        "status": "completed",
        "total_processed": total_lines
    }
```

### Caching

Cache expensive operations:

```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=100)
def expensive_operation(input_hash: str) -> Dict[str, Any]:
    """Cached expensive computation."""
    # Actual input passed separately to avoid cache key issues
    return {"result": "..."}

def my_skill(large_input: str) -> Dict[str, Any]:
    # Hash input for cache key
    input_hash = hashlib.sha256(large_input.encode()).hexdigest()

    return expensive_operation(input_hash)
```

### Metrics and Monitoring

Export skill execution metrics:

```python
from prometheus_client import Counter, Histogram

skill_executions = Counter(
    'skill_executions_total',
    'Total skill executions',
    ['skill_name', 'status']
)

skill_duration = Histogram(
    'skill_duration_seconds',
    'Skill execution duration',
    ['skill_name']
)

def my_skill(input_data: str) -> Dict[str, Any]:
    with skill_duration.labels(skill_name="my_skill").time():
        try:
            result = process(input_data)
            skill_executions.labels(skill_name="my_skill", status="success").inc()
            return result
        except Exception as e:
            skill_executions.labels(skill_name="my_skill", status="error").inc()
            raise
```

## See Also

- [Workflow Manifests](manifests.md) - Using skills in workflows
- [API Reference](api-reference.md) - Skill execution API
- [MCP Integration](mcp.md) - External tool integration
- [Examples](../examples/03-custom-skill/) - Working skill examples
