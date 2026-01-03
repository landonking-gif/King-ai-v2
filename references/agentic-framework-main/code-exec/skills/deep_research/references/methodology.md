# Deep Research Methodology Reference

This document provides detailed methodology for conducting comprehensive web research.

## Search Query Generation

### Query Expansion Strategies

1. **Direct Query**: Use the original research question as-is
2. **Analysis Variant**: Append "analysis" or "research" to get analytical content
3. **Comparison Variant**: Append "comparison" or "vs" for comparative perspectives
4. **Statistics Variant**: Append "statistics" or "data" for quantitative sources
5. **Expert Opinion**: Append "expert opinion" or "review" for authoritative sources

### Depth Levels

| Depth | Queries | Best For |
|-------|---------|----------|
| quick | 2-3 | Simple factual lookups |
| standard | 4-5 | General research questions |
| thorough | 6-8 | Complex topics requiring multiple perspectives |

## Source Ranking Algorithm

Sources are ranked by relevance using these factors:

1. **Query Term Matches** (40% weight)
   - Title matches: 2 points per term
   - URL matches: 1 point per term
   - Snippet matches: 1 point per term

2. **Domain Authority** (30% weight)
   - .edu domains: +3 points
   - .gov domains: +3 points
   - .org domains: +2 points
   - Known authoritative domains: +2 points (wikipedia, reuters, bbc, etc.)

3. **Content Freshness** (20% weight)
   - Recent dates in URL/title: +2 points

4. **URL Quality** (10% weight)
   - Shorter, cleaner URLs preferred
   - Avoid tracking parameters

## Content Extraction Guidelines

### Firecrawl Integration

When using Firecrawl MCP for content extraction:

```python
result = tool_executor._exec_mcp_call(
    tool_id="firecrawl_mcp",
    tool_name="firecrawl_scrape",
    arguments={
        "url": url,
        "formats": ["markdown"],
        "onlyMainContent": True,
    },
)
```

### Content Processing

1. **Truncate** to 8000 characters to manage context window
2. **Preserve** headings and structure
3. **Remove** navigation, ads, footers
4. **Extract** key facts and data points

## Fact Extraction Patterns

### Key Information Types

- **Statistics**: Numbers, percentages, measurements
- **Dates**: Temporal information, timelines
- **Entities**: Organizations, people, products
- **Claims**: Assertions that can be verified
- **Definitions**: Technical terms explained

### Cross-Source Validation

Facts appearing in multiple sources receive higher confidence scores:
- 1 source: 0.6 confidence
- 2 sources: 0.75 confidence
- 3+ sources: 0.9 confidence

## Output Formats

### Markdown Format (Default)
Best for human-readable reports with citations.

### JSON Format
Best for programmatic consumption and further processing.

### Summary Format
Best for quick overviews, returns condensed synthesis context.

## Error Handling

### Common Issues

1. **Rate Limiting**: Implement exponential backoff
2. **Blocked URLs**: Skip and continue with other sources
3. **Empty Content**: Mark as "failed" but don't halt research
4. **Timeout**: Use 10-second timeout per URL

### Minimum Source Requirements

If unable to reach minimum sources:
- Return partial results with `success: false`
- Include `failed_sources` count
- Provide available synthesis context
