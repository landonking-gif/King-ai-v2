# Custom Skill Example

This example demonstrates how to create a custom skill from scratch with proper structure, validation, and tests.

## Overview

Creates a sentiment analysis skill that:
- Analyzes text sentiment (positive/negative/neutral)
- Returns polarity score (-1 to 1)
- Provides confidence level
- Follows the dual-format skill structure

## Prerequisites

```bash
# No additional dependencies required
# The skill uses only Python standard library for demonstration
```

## Running the Example

```bash
# From the repository root
cd examples/03-custom-skill
python run.py
```

## What This Example Shows

1. **Skill Structure**: Complete skill directory layout
2. **Metadata Definition**: skill.yaml configuration
3. **Schema Validation**: JSON Schema for inputs/outputs
4. **Handler Implementation**: Python function implementation
5. **Unit Testing**: Test coverage for the skill

## Skill Structure

```
my_custom_skill/
├── skill.yaml          # Skill metadata and configuration
├── schema.json         # JSON Schema for validation
├── handler.py          # Skill implementation
└── test_handler.py     # Unit tests
```

### 1. skill.yaml (Metadata)

```yaml
name: sentiment_analyzer
version: "1.0.0"
description: Analyze sentiment of text and return polarity score

safety_flags:
  - none  # This is a safe, read-only skill

requires_approval: false

handler: handler.analyze_sentiment

inputs_schema: schema.json#/definitions/input
outputs_schema: schema.json#/definitions/output

tags:
  - nlp
  - sentiment
  - analysis
```

### 2. schema.json (Validation)

```json
{
  "definitions": {
    "input": {
      "type": "object",
      "properties": {
        "text": {
          "type": "string",
          "description": "Text to analyze for sentiment"
        }
      },
      "required": ["text"]
    },
    "output": {
      "type": "object",
      "properties": {
        "sentiment": {
          "type": "string",
          "enum": ["positive", "negative", "neutral"]
        },
        "polarity": {
          "type": "number",
          "minimum": -1.0,
          "maximum": 1.0
        },
        "confidence": {
          "type": "number"
        },
        "word_count": {
          "type": "integer"
        }
      },
      "required": ["sentiment", "polarity", "confidence", "word_count"]
    }
  }
}
```

### 3. handler.py (Implementation)

```python
def analyze_sentiment(text: str) -> Dict[str, Any]:
    """Analyze the sentiment of the given text."""
    # Implementation here
    return {
        "sentiment": "positive",  # or "negative", "neutral"
        "polarity": 0.75,         # -1.0 to 1.0
        "confidence": 0.85,       # 0.0 to 1.0
        "word_count": 10
    }
```

### 4. test_handler.py (Tests)

```python
def test_positive_sentiment():
    text = "This is an excellent product!"
    result = analyze_sentiment(text)
    assert result["sentiment"] == "positive"
    assert result["polarity"] > 0
```

## Expected Output

```
============================================================
Custom Skill Example
============================================================

📦 Skill: sentiment_analyzer
   Location: my_custom_skill/
   Type: NLP - Sentiment Analysis
   Safety: No flags (read-only operation)

✓ Skill loaded successfully

🔄 Running Skill Tests:

[Test 1/4] Input: "This is an excellent product! I absolutely love i..."
  ✓ Sentiment: positive (expected: positive)
  Polarity: +0.357 | Confidence: 71.4%
  Words analyzed: 7

[Test 2/4] Input: "Terrible experience. Very disappointed and frustra..."
  ✓ Sentiment: negative (expected: negative)
  Polarity: -0.571 | Confidence: 85.7%
  Words analyzed: 7

[Test 3/4] Input: "The product arrived on time. It has standard feat..."
  ✓ Sentiment: neutral (expected: neutral)
  Polarity: +0.000 | Confidence: 30.0%
  Words analyzed: 10

[Test 4/4] Input: "The service was great but the quality was poor...."
  ✓ Sentiment: neutral (expected: neutral)
  Polarity: +0.000 | Confidence: 40.0%
  Words analyzed: 10

============================================================
Running Unit Tests
============================================================

✓ Positive sentiment
✓ Negative sentiment
✓ Neutral sentiment
✓ Empty text handling
✓ Output schema validation

============================================================
Skill Structure
============================================================

my_custom_skill/
├── skill.yaml          # Skill metadata
├── schema.json         # Input/output schemas
├── handler.py          # Implementation
└── test_handler.py     # Unit tests
```

## Key Concepts

### Safety Flags

Control what the skill can access:
- `none`: No special permissions needed
- `file_system`: Requires file system access
- `network_access`: Makes network requests
- `pii_risk`: May handle sensitive data
- `side_effect`: Modifies external state

### Approval Requirement

```yaml
requires_approval: false  # Set to true for sensitive operations
```

### Schema Validation

All inputs and outputs are validated against JSON Schema to ensure type safety and data integrity.

## Installing the Skill

To use this skill in the framework:

```bash
# Copy to framework skills directory
cp -r my_custom_skill/ ../../code-exec/skills/

# Or create a symbolic link
ln -s $(pwd)/my_custom_skill ../../code-exec/skills/sentiment_analyzer
```

## Testing the Skill

```bash
# Run the test file directly
cd my_custom_skill
python test_handler.py

# Or use pytest if installed
pytest test_handler.py -v
```

## Next Steps

- See `04-mcp-integration/` for external tool integration
- Read docs/skills.md for advanced skill development
- Explore code-exec/skills/ for more skill examples
- Learn about the Anthropic Skills format (SKILL.md)
