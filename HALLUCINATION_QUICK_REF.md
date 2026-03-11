# Quick Reference: Preventing Hallucinations

## ✅ What Was Fixed

1. **Temperature Control** - Lowered from 0.7 to task-specific values (0.1-0.6)
2. **System Prompts** - Added explicit accuracy requirements and anti-hallucination rules
3. **Fact-Checking** - Automatic validation of LLM outputs for accuracy-critical tasks
4. **Grounding Requirements** - All prompts now require citations and context references

## 🎯 How To Use

### For Factual/Critical Tasks
```python
task_context = TaskContext(
    task_type="finance",      # research, finance, legal, analytics
    risk_level="high",        # high or critical
    requires_accuracy=True,   # ⚠️ IMPORTANT - enables fact-checking
    token_estimate=2000,
    priority="critical"
)
```
→ **Result**: Temperature = 0.1-0.2, automatic fact-checking enabled

### For Creative Tasks
```python
task_context = TaskContext(
    task_type="content",      # content, conversation
    risk_level="low",
    requires_accuracy=False,
    token_estimate=1000,
    priority="normal"
)
```
→ **Result**: Temperature = 0.5-0.6, fact-checking disabled

## 📊 Temperature Settings

| Task Type | Temperature | Use Case |
|-----------|-------------|----------|
| finance | 0.1 | Financial calculations, analysis |
| legal | 0.1 | Legal analysis, compliance |
| research | 0.2 | Factual research, data gathering |
| analytics | 0.2 | Data analysis, reporting |
| query | 0.2 | Information retrieval |
| code | 0.2 | Code generation |
| summary | 0.3 | Text summarization |
| planning | 0.4 | Strategic planning |
| conversation | 0.5 | Natural conversation |
| content | 0.6 | Creative content generation |

## 🛡️ Fact-Checking

Automatic for tasks with `requires_accuracy=True`:

**Detects**:
- ✗ Hallucination patterns ("I recall that...", "based on my knowledge...")
- ✗ Ungrounded numbers not in context
- ✗ Fabricated URLs
- ✗ Missing citations for factual claims

**Validates**:
- ✓ Response references provided context
- ✓ Numbers match provided data
- ✓ Uncertainty acknowledged when appropriate

## ⚠️ When Hallucinations Are Detected

**For high/critical risk tasks**: Response is REJECTED if confidence < 0.5

**For other tasks**: Warning logged, but response still returned

## 🔍 Manual Fact-Check

```python
from src.utils.fact_checker import check_for_hallucination

result = check_for_hallucination(
    response=llm_response,
    context=provided_context,
    expected_data={"metric": "expected_value"}
)

if not result.is_valid:
    print(f"Issues: {result.issues}")
    print(f"Confidence: {result.confidence}")
```

## 📝 Updated Prompts

All prompts now include:
- **ACCURACY REQUIREMENTS** section
- **GROUNDING REQUIREMENTS** section  
- Citation/reference fields in responses
- Explicit "don't hallucinate" instructions

## 🚨 Red Flags to Watch For

If you see responses with:
- "I recall that..."
- "Based on my knowledge..."
- "I've learned that..."
- Numbers not from your context
- Definitive claims without citations

→ Check `task_context.requires_accuracy` is set to `True`

## 💡 Best Practices

1. **Always set requires_accuracy=True for factual tasks**
2. **Provide comprehensive context** - more context = less hallucination
3. **Monitor logs** for hallucination warnings
4. **Use appropriate risk_level** - affects provider routing
5. **Test with missing data** - verify AI admits "I don't have that data"

## 🎛️ Override Temperature

```python
response = await llm_router.complete(
    prompt="...",
    system="...",
    context=task_context,
    temperature=0.05  # Force ultra-deterministic
)
```

## 📈 Monitoring Metrics

- `llm.hallucination_detected` - Count of detected hallucinations
- `llm.calls_success` - Successful calls
- `llm.calls_error` - Failed calls

## 🔧 Configuration Files

- **Temperature logic**: `src/utils/llm_router.py` → `_get_temperature_for_task()`
- **Fact-checker**: `src/utils/fact_checker.py`
- **System prompts**: `src/master_ai/prompts.py`
- **Integration**: `src/master_ai/brain.py` → `_call_llm()`

## ❓ Troubleshooting

| Problem | Solution |
|---------|----------|
| Still hallucinating | Set `requires_accuracy=True` in TaskContext |
| Too conservative | Increase temperature for that task type |
| False positives | Adjust fact-checker thresholds |
| Need highest accuracy | Use `risk_level="high"` to route to Claude |

---

**For detailed documentation, see**: [ANTI_HALLUCINATION_GUIDE.md](ANTI_HALLUCINATION_GUIDE.md)
