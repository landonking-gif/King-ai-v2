---
name: deep-research
description: Conduct comprehensive web research with source fetching, content extraction, and synthesis into structured reports
version: 1.0.0
dependencies: python>=3.8, requests>=2.28.0
tags:
  - research
  - web
  - analysis
  - reports
---

# Deep Research Skill

This skill conducts comprehensive research by searching the web, fetching full page content from multiple sources, extracting key data points, and preparing structured data for synthesis into detailed reports with citations.

## When to Use

- User asks for research, analysis, or investigation on a topic
- Questions require verification from multiple sources
- Market data, price trends, or financial analysis needed
- Reports with proper source citations are required
- Qualitative reasoning across multiple data points needed

## Quick Start

```python
result = deep_research(
    query="silver prices India last 30 days",
    min_sources=10,
    output_format="markdown"
)
```

## Resources

- **Methodology Guide**: See [methodology.md](references/methodology.md) for detailed research methodology
- **URL Validator**: Use [validate_sources.py](scripts/validate_sources.py) to pre-validate URLs

## Process

1. **Search Phase**: Execute web search to find relevant sources (minimum 10 URLs)
2. **Rank Phase**: Score URLs by relevance - see [methodology.md](references/methodology.md#source-ranking-algorithm)
3. **Fetch Phase**: Retrieve full page content from top sources via Firecrawl
4. **Extract Phase**: Parse and extract key data points from each source
5. **Prepare Phase**: Structure extracted data for LLM synthesis

## Configuration

The following environment variables control behavior:

- `DEEP_RESEARCH_MIN_SOURCES`: Minimum sources to fetch (default: 10)
- `DEEP_RESEARCH_MAX_SOURCES`: Maximum sources to fetch (default: 15)
- `DEEP_RESEARCH_SEARCH_RESULTS`: Initial search results to retrieve (default: 20)

## Input Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | Yes | - | The research question or topic |
| `min_sources` | integer | No | 10 | Minimum sources to fetch |
| `max_sources` | integer | No | 15 | Maximum sources to fetch |
| `output_format` | string | No | "markdown" | "markdown", "json", or "summary" |
| `search_depth` | string | No | "standard" | "quick", "standard", or "thorough" |
| `include_raw_content` | boolean | No | false | Include raw fetched content |

## Output Schema

```json
{
  "success": true,
  "query": "original query",
  "sources": [
    {
      "url": "https://...",
      "title": "Page Title",
      "domain": "example.com",
      "content_summary": "Extracted content...",
      "relevance_score": 0.85,
      "fetch_status": "success"
    }
  ],
  "source_count": 10,
  "failed_sources": 0,
  "extracted_facts": [
    {"fact": "...", "source_index": 0, "confidence": 0.9}
  ],
  "synthesis_context": {
    "combined_content": "...",
    "source_citations": ["[1]", "[2]"],
    "synthesis_instructions": "..."
  }
}
```

## Scripts

### URL Validation Script

Before fetching, validate URLs to avoid blocked domains:

```bash
python scripts/validate_sources.py https://example.com https://another.com
```

Returns JSON with validation results including blocked domains and paywall warnings.

## Example Usage

Query: "Find silver prices in India for the last 30 days with qualitative analysis"

Expected behavior:
1. Search for "silver prices India December 2024 30 days trends"
2. Fetch content from top 10 financial/commodity websites
3. Extract price data, dates, trends, expert opinions
4. Return structured data for comprehensive report generation

## Advanced Topics

For detailed information on:
- Query expansion strategies - see [methodology.md#search-query-generation](references/methodology.md#search-query-generation)
- Source ranking algorithm - see [methodology.md#source-ranking-algorithm](references/methodology.md#source-ranking-algorithm)
- Fact extraction patterns - see [methodology.md#fact-extraction-patterns](references/methodology.md#fact-extraction-patterns)
- Error handling - see [methodology.md#error-handling](references/methodology.md#error-handling)

## Safety Considerations

This skill makes external network requests to:
- Web search APIs (DuckDuckGo/Tavily)
- Firecrawl API for content scraping
- Target websites for content retrieval

Rate limiting and respectful crawling practices are enforced.
