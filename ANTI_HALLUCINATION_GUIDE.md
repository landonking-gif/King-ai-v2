# Anti-Hallucination Configuration Guide

This document describes the anti-hallucination measures implemented in King AI v2 to ensure factual accuracy and reduce false information.

## Overview

King AI was experiencing hallucination issues where the LLM would generate plausible-sounding but incorrect information. The following measures have been implemented:

## 1. Temperature Control

**Problem**: High temperature (0.7) encourages creative/varied outputs, leading to hallucinations.

**Solution**: Dynamic temperature based on task type:

```python
# Task-specific temperatures
"research": 0.2,        # Factual research
"finance": 0.1,         # Financial calculations
"legal": 0.1,           # Legal analysis  
"analytics": 0.2,       # Data analysis
"query": 0.2,           # Information retrieval
"conversation": 0.5,    # Natural conversation (can be more creative)
"content": 0.6,         # Creative content generation
```

**Files Modified**:
- `src/utils/llm_router.py` - Added `_get_temperature_for_task()` method
- `src/utils/claude_client.py` - Temperature parameter support
- `src/utils/ollama_client.py` - Temperature parameter support  
- `src/utils/gemini_client.py` - Temperature parameter support

## 2. System Prompt Improvements

**Problem**: System prompt didn't emphasize accuracy or discourage hallucination.

**Solution**: Added explicit anti-hallucination guidelines:

```
ACCURACY REQUIREMENTS:
- NEVER make up or fabricate data, metrics, or facts
- Only state information you can verify from the provided context
- If you don't have specific information, explicitly say "I don't have that data"
- Cite specific sources when making factual claims
- Distinguish clearly between facts, estimates, and predictions
- When uncertain, express confidence levels
```

**Files Modified**:
- `src/master_ai/prompts.py` - Updated `SYSTEM_PROMPT` with accuracy requirements

## 3. Fact-Checking Validation Layer

**Problem**: No validation of LLM outputs against provided context.

**Solution**: Implemented `FactChecker` class that:

1. **Detects hallucination patterns**: Identifies phrases like "I recall that", "based on my knowledge"
2. **Validates grounding**: Checks if response references provided context
3. **Verifies numbers**: Ensures numerical claims match context data
4. **Detects fabricated URLs**: Identifies suspicious/fake URL patterns
5. **Validates against expected data**: Checks response against known facts

**Files Created**:
- `src/utils/fact_checker.py` - Complete fact-checking module

**Integration**:
- Integrated into `MasterAI._call_llm()` method
- Runs automatically for accuracy-critical tasks
- Rejects responses with confidence < 0.5 for high-risk tasks

## 4. Grounding Requirements in Prompts

**Problem**: Prompts didn't require LLM to cite sources or ground responses in context.

**Solution**: Updated all key prompts to require:

- Explicit references to context data
- Citation of specific metrics/facts from provided information
- Identification of missing information rather than assumptions
- Evidence-based reasoning

**Prompts Updated**:
- `PLANNING_PROMPT` - Added context_reference field
- `TASK_DECOMPOSITION_PROMPT` - Added justification field
- `EVOLUTION_PROMPT` - Added evidence field

## 5. Configuration Settings

### Environment Variables

No new environment variables are required. The system uses existing LLM configurations.

### Temperature Override

You can override temperature for specific calls:

```python
response = await llm_router.complete(
    prompt="Your prompt",
    system="System prompt",
    context=TaskContext(...),
    temperature=0.1  # Force very low temperature
)
```

### Fact-Checking Configuration

Adjust sensitivity in `src/utils/fact_checker.py`:

```python
class FactChecker:
    # Adjust these thresholds as needed
    HALLUCINATION_PATTERNS = [...]  # Add more patterns
    UNCERTAINTY_PHRASES = [...]     # Phrases indicating AI knows limits
    GROUNDING_PHRASES = [...]       # Phrases showing grounded responses
```

## Usage Examples

### High-Accuracy Financial Analysis

```python
task_context = TaskContext(
    task_type="finance",
    risk_level="high",
    requires_accuracy=True,
    token_estimate=2000,
    priority="critical"
)

# This will use temperature=0.1 and fact-check the response
response = await master_ai._call_llm(
    prompt="Analyze Q4 revenue",
    system=SYSTEM_PROMPT,
    task_context=task_context
)
```

### Creative Content (Allows More Variation)

```python
task_context = TaskContext(
    task_type="content",
    risk_level="low",
    requires_accuracy=False,
    token_estimate=1000,
    priority="normal"
)

# This will use temperature=0.6, no strict fact-checking
response = await master_ai._call_llm(
    prompt="Write marketing copy",
    task_context=task_context
)
```

## Monitoring

The system logs hallucination detections:

```python
logger.warning(
    "Potential hallucination detected",
    issues=fact_check.issues,
    warnings=fact_check.warnings,
    confidence=fact_check.confidence
)
```

Monitor these metrics:
- `llm.hallucination_detected` - Count of detected hallucinations
- `llm.calls_success` - Successful LLM calls
- `llm.calls_error` - Failed LLM calls

## Testing

To verify anti-hallucination measures:

1. **Test with missing data**:
   ```python
   # Provide minimal context
   # Verify AI says "I don't have that data" rather than making up numbers
   ```

2. **Test temperature settings**:
   ```python
   # Run same query with different temperatures
   # Verify lower temp = more consistent, factual responses
   ```

3. **Test fact-checker**:
   ```python
   from src.utils.fact_checker import check_for_hallucination
   
   result = check_for_hallucination(
       response="Revenue is $1M based on Q3 data",
       context="Q3 revenue: $500K",
       expected_data={"revenue": "$500K"}
   )
   
   assert not result.is_valid  # Should detect mismatch
   ```

## Troubleshooting

### Issue: Responses are too conservative

**Solution**: Lower the temperature slightly for that task type in `_get_temperature_for_task()`

### Issue: Too many false positive hallucination warnings

**Solution**: Adjust fact-checker thresholds:
- Reduce pattern matching sensitivity
- Increase confidence threshold
- Add more valid grounding phrases

### Issue: Still seeing hallucinations

**Solution**:
1. Check task context is properly set with `requires_accuracy=True`
2. Verify system prompt is being used
3. Review and strengthen prompt grounding requirements
4. Consider switching to Claude for critical tasks (highest accuracy)

## Best Practices

1. **Always provide context**: The more relevant context you provide, the less room for hallucination
2. **Set appropriate TaskContext**: Use `requires_accuracy=True` for factual tasks
3. **Monitor fact-check logs**: Review hallucination warnings to improve prompts
4. **Test with edge cases**: Verify behavior when data is missing or ambiguous
5. **Use appropriate temperature**: Don't force low temperature for creative tasks

## Future Improvements

Potential enhancements:
- RAG integration for fact verification against knowledge base
- Multi-pass validation for critical tasks
- Confidence scoring for all outputs
- Automated prompt optimization based on hallucination rates
- Integration with external fact-checking APIs
