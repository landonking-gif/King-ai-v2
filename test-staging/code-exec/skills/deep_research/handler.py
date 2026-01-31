"""
Deep Research Skill Handler.

Module: code-exec/skills/deep_research/handler.py

Provides deterministic implementation for comprehensive web research:
1. Search for relevant sources
2. Rank URLs by relevance
3. Fetch full page content via Firecrawl
4. Extract key data points
5. Prepare synthesis context for LLM

Configuration via environment variables:
- DEEP_RESEARCH_MIN_SOURCES: Minimum sources to fetch (default: 10)
- DEEP_RESEARCH_MAX_SOURCES: Maximum sources to fetch (default: 15)
- DEEP_RESEARCH_SEARCH_RESULTS: Initial search results (default: 20)
"""

import os
import re
import time
import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def _get_config() -> Dict[str, int]:
    """Load configuration from environment variables."""
    return {
        "min_sources": int(os.getenv("DEEP_RESEARCH_MIN_SOURCES", "10")),
        "max_sources": int(os.getenv("DEEP_RESEARCH_MAX_SOURCES", "15")),
        "search_results": int(os.getenv("DEEP_RESEARCH_SEARCH_RESULTS", "20")),
    }


def deep_research(
    query: str,
    documents: Optional[List[str]] = None,
    min_sources: Optional[int] = None,
    max_sources: Optional[int] = None,
    output_format: str = "markdown",
    search_depth: str = "standard",
    include_raw_content: bool = False,
    tool_executor: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Conduct comprehensive web research on a topic.

    This handler orchestrates the research pipeline:
    1. (Optional) Process local documents with document_qa skill
    2. Execute web search to find relevant sources
    3. Rank and filter URLs by relevance
    4. Fetch full content from top sources via Firecrawl MCP
    5. Extract and structure key data points
    6. Merge document and web sources with unified citations
    7. Prepare context for LLM synthesis

    Args:
        query: Research question or topic
        documents: Optional list of local documents (PDF, DOCX, XLSX, PPTX)
        min_sources: Minimum sources to fetch (default from env: 10)
        max_sources: Maximum sources to fetch (default from env: 15)
        output_format: Output format (markdown, json, summary)
        search_depth: Search depth (quick, standard, thorough)
        include_raw_content: Include raw fetched content
        tool_executor: ToolExecutor instance for web search and MCP calls

    Returns:
        Dictionary with sources, extracted data, and synthesis context
    """
    start_time = time.time()
    config = _get_config()

    # Apply defaults from config
    min_sources = min_sources or config["min_sources"]
    max_sources = max_sources or config["max_sources"]
    search_results_count = config["search_results"]

    # Adjust search results based on depth
    if search_depth == "quick":
        search_results_count = min(search_results_count, 15)
    elif search_depth == "thorough":
        search_results_count = max(search_results_count, 25)

    logger.info(f"Starting deep research: '{query}' (min={min_sources}, max={max_sources})")

    # Initialize result structure
    result = {
        "success": False,
        "query": query,
        "sources": [],
        "source_count": 0,
        "failed_sources": 0,
        "extracted_facts": [],
        "synthesis_context": {},
        "metadata": {
            "search_queries_used": [],
            "total_content_tokens": 0,
            "execution_time_seconds": 0,
            "documents_processed": 0,
        },
    }

    # Process local documents if provided
    document_context = None
    document_sources = []
    if documents:
        try:
            from skills.document_qa.handler import document_qa

            logger.info(f"Processing {len(documents)} local documents...")
            doc_result = document_qa(
                documents=documents,
                query=query,
                include_sources=True,
                max_context_tokens=4000,  # Reserve space for web sources
                rerank_top_k=5,
            )

            if doc_result.get("success"):
                document_context = doc_result.get("answer", "")
                document_sources = doc_result.get("sources", [])
                result["metadata"]["documents_processed"] = len(documents)
                logger.info(f"Extracted {len(document_sources)} sources from documents")
            else:
                logger.warning(f"Document processing failed: {doc_result.get('error')}")

        except ImportError:
            logger.warning("document_qa skill not available, skipping document processing")
        except Exception as e:
            logger.warning(f"Error processing documents: {e}")

    try:
        # Step 1: Generate search queries
        search_queries = _generate_search_queries(query, search_depth)
        result["metadata"]["search_queries_used"] = search_queries

        # Step 2: Execute web searches
        all_search_results = []
        for search_query in search_queries:
            if tool_executor:
                search_result = tool_executor._exec_web_search(
                    query=search_query,
                    max_results=search_results_count,
                )
                if search_result.get("success") and search_result.get("results"):
                    all_search_results.extend(search_result["results"])
            else:
                # Fallback: return instructions for manual execution
                logger.warning("No tool_executor provided, returning search instructions")
                result["synthesis_context"]["manual_steps"] = [
                    f"1. Search web for: {search_query}",
                    "2. Fetch content from top 10 URLs using firecrawl_mcp",
                    "3. Extract key data points",
                    "4. Synthesize into report",
                ]

        if not all_search_results:
            result["error"] = "No search results found"
            return result

        # Step 3: Deduplicate and rank URLs
        ranked_urls = _rank_and_deduplicate_urls(all_search_results, query, max_sources)
        logger.info(f"Ranked {len(ranked_urls)} unique URLs")

        # Step 4: Fetch content from top URLs
        fetched_sources = []
        failed_count = 0

        for url_data in ranked_urls[:max_sources]:
            url = url_data["url"]
            try:
                content = _fetch_page_content(url, tool_executor)
                if content:
                    source_data = {
                        "url": url,
                        "title": url_data.get("title", ""),
                        "domain": urlparse(url).netloc,
                        "content_summary": _summarize_content(content, max_tokens=800),
                        "extracted_data": _extract_structured_data(content, query),
                        "relevance_score": url_data.get("score", 0.5),
                        "fetch_status": "success",
                    }
                    if include_raw_content:
                        source_data["raw_content"] = content[:5000]
                    fetched_sources.append(source_data)
                else:
                    failed_count += 1
            except Exception as e:
                logger.warning(f"Failed to fetch {url}: {e}")
                failed_count += 1

            # Check if we have minimum sources
            if len(fetched_sources) >= min_sources:
                logger.info(f"Reached minimum sources: {len(fetched_sources)}")
                # Continue fetching up to max if more URLs available
                if len(fetched_sources) >= max_sources:
                    break

        result["sources"] = fetched_sources
        result["source_count"] = len(fetched_sources)
        result["failed_sources"] = failed_count

        # Step 5: Extract facts across all sources
        result["extracted_facts"] = _extract_cross_source_facts(fetched_sources, query)

        # Step 6: Add document sources to results if available
        if document_sources:
            for doc_src in document_sources:
                # Convert document sources to web source format for consistency
                result["sources"].append({
                    "url": f"local://{doc_src.get('file', 'unknown')}",
                    "title": doc_src.get("file", "Local Document"),
                    "domain": "local",
                    "content_summary": doc_src.get("preview", ""),
                    "extracted_data": {
                        "page": doc_src.get("page"),
                        "section": doc_src.get("section"),
                        "type": "document",
                    },
                    "relevance_score": 0.9,  # High relevance for user-provided docs
                    "fetch_status": "success",
                    "source_type": "document",
                })

        # Step 7: Prepare synthesis context
        result["synthesis_context"] = _prepare_synthesis_context(
            query=query,
            sources=fetched_sources,
            extracted_facts=result["extracted_facts"],
            output_format=output_format,
            document_context=document_context,
        )

        # Calculate token estimate
        total_content = " ".join([s.get("content_summary", "") for s in fetched_sources])
        if document_context:
            total_content += " " + document_context
        result["metadata"]["total_content_tokens"] = len(total_content.split()) * 1.3

        result["success"] = len(fetched_sources) >= min_sources or len(document_sources) > 0

    except Exception as e:
        logger.error(f"Deep research failed: {e}")
        result["error"] = str(e)

    result["metadata"]["execution_time_seconds"] = round(time.time() - start_time, 2)
    return result


def _generate_search_queries(query: str, depth: str) -> List[str]:
    """
    Generate multiple search queries for comprehensive coverage.

    Args:
        query: Original research query
        depth: Search depth (quick, standard, thorough)

    Returns:
        List of search queries
    """
    queries = [query]

    if depth in ["standard", "thorough"]:
        # Add variations
        queries.append(f"{query} analysis")
        queries.append(f"{query} latest news")

    if depth == "thorough":
        # Add more specific queries
        queries.append(f"{query} expert opinion")
        queries.append(f"{query} data statistics")
        queries.append(f"{query} trends 2024 2025")

    return queries


def _rank_and_deduplicate_urls(
    search_results: List[Dict], query: str, max_urls: int
) -> List[Dict]:
    """
    Rank and deduplicate URLs from search results.

    Args:
        search_results: Raw search results with url, title, snippet
        query: Original query for relevance scoring
        max_urls: Maximum URLs to return

    Returns:
        Ranked and deduplicated URL list
    """
    seen_urls = set()
    seen_domains = {}
    ranked = []

    # Keywords for relevance scoring
    query_words = set(query.lower().split())

    for result in search_results:
        url = result.get("url", "")
        if not url or url in seen_urls:
            continue

        # Extract domain
        try:
            domain = urlparse(url).netloc
        except Exception:
            continue

        # Limit per domain (max 2 from same domain)
        if seen_domains.get(domain, 0) >= 2:
            continue

        # Calculate relevance score
        title = result.get("title", "").lower()
        snippet = result.get("snippet", "").lower()
        combined = f"{title} {snippet}"

        score = 0.0
        for word in query_words:
            if word in combined:
                score += 0.1

        # Boost for authoritative domains
        authoritative_domains = [
            "reuters", "bloomberg", "economist", "ft.com", "wsj.com",
            "moneycontrol", "economictimes", "livemint", "ndtv",
            "bbc", "cnbc", "forbes", "wikipedia",
        ]
        for auth_domain in authoritative_domains:
            if auth_domain in domain.lower():
                score += 0.3
                break

        # Penalize social media and forums
        low_quality = ["reddit", "quora", "facebook", "twitter", "pinterest"]
        for lq in low_quality:
            if lq in domain.lower():
                score -= 0.2
                break

        ranked.append({
            "url": url,
            "title": result.get("title", ""),
            "snippet": result.get("snippet", ""),
            "score": min(1.0, max(0.0, score)),
            "domain": domain,
        })

        seen_urls.add(url)
        seen_domains[domain] = seen_domains.get(domain, 0) + 1

    # Sort by score descending
    ranked.sort(key=lambda x: x["score"], reverse=True)

    return ranked[:max_urls]


def _fetch_page_content(url: str, tool_executor: Optional[Any]) -> Optional[str]:
    """
    Fetch full page content using Firecrawl MCP.

    Args:
        url: URL to fetch
        tool_executor: ToolExecutor with MCP access

    Returns:
        Extracted markdown content or None
    """
    if not tool_executor:
        return None

    try:
        # Try Firecrawl MCP first
        result = tool_executor._exec_mcp_call(
            tool_id="firecrawl_mcp",
            tool_name="firecrawl_scrape",
            arguments={
                "url": url,
                "formats": ["markdown"],
                "onlyMainContent": True,
            },
        )

        if result.get("success") and result.get("result"):
            content = result["result"]
            if isinstance(content, dict):
                return content.get("markdown", content.get("content", ""))
            return str(content)

    except Exception as e:
        logger.warning(f"Firecrawl failed for {url}: {e}")

    # Fallback: try basic web fetch if available
    try:
        if hasattr(tool_executor, "_exec_web_fetch"):
            result = tool_executor._exec_web_fetch(url=url)
            if result.get("success"):
                return result.get("content", "")
    except Exception:
        pass

    return None


def _summarize_content(content: str, max_tokens: int = 800) -> str:
    """
    Summarize content to fit within token limit.

    Uses extractive summarization based on sentence scoring.

    Args:
        content: Full page content
        max_tokens: Maximum tokens in summary

    Returns:
        Summarized content
    """
    if not content:
        return ""

    # Estimate current tokens
    words = content.split()
    if len(words) <= max_tokens:
        return content

    # Split into sentences
    sentences = re.split(r"[.!?]+", content)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 20]

    if not sentences:
        return content[:max_tokens * 4]  # Rough char estimate

    # Score sentences by position and length
    scored = []
    for i, sentence in enumerate(sentences):
        score = 1.0
        # Boost first sentences
        if i < 3:
            score += 0.5
        # Penalize very long sentences
        if len(sentence) > 300:
            score -= 0.3
        scored.append((score, sentence))

    # Select top sentences
    scored.sort(key=lambda x: x[0], reverse=True)

    summary_sentences = []
    current_tokens = 0
    for score, sentence in scored:
        sentence_tokens = len(sentence.split())
        if current_tokens + sentence_tokens > max_tokens:
            break
        summary_sentences.append(sentence)
        current_tokens += sentence_tokens

    return ". ".join(summary_sentences) + "."


def _extract_structured_data(content: str, query: str) -> Dict[str, Any]:
    """
    Extract structured data points from content.

    Uses regex patterns to find:
    - Numbers and statistics
    - Dates
    - Monetary values
    - Percentages

    Args:
        content: Page content
        query: Research query for context

    Returns:
        Dictionary of extracted data
    """
    extracted = {
        "numbers": [],
        "dates": [],
        "monetary": [],
        "percentages": [],
        "key_phrases": [],
    }

    if not content:
        return extracted

    # Extract monetary values (including Indian Rupees)
    money_pattern = r"(?:Rs\.?|INR|USD|\$|EUR|£)\s*[\d,]+(?:\.\d{2})?(?:\s*(?:crore|lakh|million|billion|thousand))?"
    extracted["monetary"] = re.findall(money_pattern, content, re.IGNORECASE)[:10]

    # Extract percentages
    pct_pattern = r"\d+(?:\.\d+)?%"
    extracted["percentages"] = re.findall(pct_pattern, content)[:10]

    # Extract dates
    date_patterns = [
        r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b",
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b",
        r"\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b",
    ]
    for pattern in date_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        extracted["dates"].extend(matches[:5])

    # Extract significant numbers with context
    number_pattern = r"(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(kg|gram|ounce|oz|per|price|rate|level|points?)"
    matches = re.findall(number_pattern, content, re.IGNORECASE)
    extracted["numbers"] = [f"{m[0]} {m[1]}" for m in matches[:10]]

    return extracted


def _extract_cross_source_facts(sources: List[Dict], query: str) -> List[Dict]:
    """
    Extract and consolidate facts across all sources.

    Args:
        sources: List of fetched source data
        query: Research query

    Returns:
        List of extracted facts with source references
    """
    facts = []
    query_words = set(query.lower().split())

    for idx, source in enumerate(sources):
        content = source.get("content_summary", "")
        extracted = source.get("extracted_data", {})

        # Add monetary facts
        for money in extracted.get("monetary", [])[:3]:
            facts.append({
                "fact": f"Price/Value: {money}",
                "source_index": idx,
                "confidence": 0.8,
            })

        # Add percentage facts
        for pct in extracted.get("percentages", [])[:3]:
            facts.append({
                "fact": f"Change/Rate: {pct}",
                "source_index": idx,
                "confidence": 0.7,
            })

        # Extract sentences containing query keywords
        sentences = re.split(r"[.!?]+", content)
        for sentence in sentences[:10]:
            sentence_lower = sentence.lower()
            if any(word in sentence_lower for word in query_words):
                if len(sentence) > 30 and len(sentence) < 300:
                    facts.append({
                        "fact": sentence.strip(),
                        "source_index": idx,
                        "confidence": 0.6,
                    })

    # Deduplicate similar facts
    unique_facts = []
    seen_hashes = set()
    for fact in facts:
        fact_hash = hashlib.md5(fact["fact"][:50].lower().encode()).hexdigest()[:8]
        if fact_hash not in seen_hashes:
            unique_facts.append(fact)
            seen_hashes.add(fact_hash)

    return unique_facts[:30]  # Limit to 30 facts


def _prepare_synthesis_context(
    query: str,
    sources: List[Dict],
    extracted_facts: List[Dict],
    output_format: str,
    document_context: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Prepare context for LLM synthesis.

    Args:
        query: Original research query
        sources: Fetched source data
        extracted_facts: Extracted facts
        output_format: Desired output format
        document_context: Optional context from local documents

    Returns:
        Synthesis context with combined content and instructions
    """
    # Build source citations
    citations = []
    for idx, source in enumerate(sources):
        citations.append({
            "index": idx + 1,
            "title": source.get("title", "Untitled"),
            "url": source.get("url", ""),
            "domain": source.get("domain", ""),
        })

    # Combine content from all sources
    combined_parts = []

    # Add document context first if available
    if document_context:
        combined_parts.append(f"[Local Documents]\n{document_context}")

    # Add web sources
    for idx, source in enumerate(sources):
        summary = source.get("content_summary", "")
        if summary:
            combined_parts.append(f"[Source {idx + 1}] {summary}")

    combined_content = "\n\n".join(combined_parts)

    # Build synthesis instructions based on output format
    if output_format == "markdown":
        instructions = f"""Based on the following research data, generate a comprehensive markdown report:

## Report Requirements:
1. **Executive Summary** (2-3 sentences)
2. **Key Findings** (bullet points with citations [1], [2], etc.)
3. **Detailed Analysis** (multiple paragraphs with data)
4. **Data Table** (if numerical data available)
5. **Qualitative Assessment** (trends, implications)
6. **Conclusions**
7. **Sources** (numbered list)

## Research Query:
{query}

## Extracted Facts:
{chr(10).join([f"- {f['fact']} [Source {f['source_index'] + 1}]" for f in extracted_facts[:15]])}

## Source Content:
{combined_content[:8000]}

Generate a well-researched, comprehensive report with proper citations."""

    elif output_format == "json":
        instructions = f"""Extract and structure the research findings as JSON:
{{
  "executive_summary": "...",
  "key_findings": [...],
  "data_points": [...],
  "trends": [...],
  "conclusions": "...",
  "confidence_score": 0.0-1.0
}}

Query: {query}
Use the provided source content and facts."""

    else:  # summary
        instructions = f"""Provide a concise summary (3-5 paragraphs) answering: {query}
Include key data points and cite sources as [1], [2], etc."""

    return {
        "combined_content": combined_content,
        "source_citations": citations,
        "synthesis_instructions": instructions,
        "output_format": output_format,
    }
