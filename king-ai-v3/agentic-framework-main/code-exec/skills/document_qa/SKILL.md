---
name: document-qa
description: Extract content from documents (PDF, Word, Excel, PowerPoint) and answer questions with source citations
version: 1.0.0
author: Agentic Framework
tags:
  - document
  - qa
  - extraction
  - rag
  - haystack
  - pdf
  - word
  - excel
  - powerpoint
dependencies:
  - python>=3.11
  - haystack-ai>=2.0.0
  - markitdown>=0.1.0
  - sentence-transformers>=2.2.0
  - openai>=1.0.0
---

# Document Q&A Skill

Extract and analyze content from various document formats, then answer questions with precise source citations.

## Capabilities

- **Document Extraction**: PDF, Word (.docx), Excel (.xlsx), PowerPoint (.pptx), HTML, CSV
- **Image Analysis**: Complex diagrams and charts via Vision LLM (GPT-4o)
- **Semantic Chunking**: Structure-aware splitting that preserves tables and code blocks
- **Hybrid Search**: Combined vector (semantic) + BM25 (keyword) retrieval
- **LLM Reranking**: LLM-as-judge scoring for relevance (no cross-encoder model)
- **Source Tracking**: Every answer includes [src_XXX] citations
- **Context Engineering**: Strategic positioning to address "lost in the middle" problem

## When to Use

- User provides documents and asks questions about their content
- Need to extract and analyze information from business documents
- Comparing information across multiple documents
- Research requiring source citations
- Processing presentations, spreadsheets, or reports

## Quick Start

```python
from skills.document_qa import document_qa

result = document_qa(
    documents=["report.pdf", "slides.pptx"],
    query="What was the Q3 revenue?"
)

print(result["answer"])
# "The Q3 revenue was $2.3M [src_001], representing a 15% increase from Q2 [src_002]."

print(result["sources"])
# [{"id": "src_001", "file": "report.pdf", "page": 3}, ...]
```

## Input Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `documents` | array[string] | Yes | - | File paths to process (PDF, DOCX, XLSX, PPTX, HTML, CSV) |
| `query` | string | Yes | - | Question to answer from documents |
| `include_sources` | boolean | No | true | Include source citations in response |
| `max_context_tokens` | integer | No | 6000 | Maximum tokens for context window |
| `rerank_top_k` | integer | No | 5 | Number of chunks after LLM reranking |
| `process_images` | boolean | No | true | Process images with Vision LLM |
| `chunk_size` | integer | No | 512 | Target chunk size in tokens |

## Output Schema

```json
{
  "success": true,
  "answer": "The Q3 revenue was $2.3M [src_001]...",
  "sources": [
    {
      "id": "src_001",
      "file": "report.pdf",
      "page": 3,
      "section": "Revenue Analysis"
    }
  ],
  "chunks_retrieved": 5,
  "confidence": 0.92,
  "metadata": {
    "documents_processed": 2,
    "chunks_created": 45,
    "images_processed": 3,
    "execution_time_seconds": 12.5
  }
}
```

## Configuration

Environment variables for customization:

```bash
# LLM Configuration (required)
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini          # Model for reranking and generation

# Document Q&A Settings (optional)
DOCUMENT_QA_MAX_IMAGES=10          # Max images to send to Vision LLM
DOCUMENT_QA_IMAGE_MIN_SIZE_KB=50   # Skip images smaller than this
DOCUMENT_QA_CHUNK_SIZE=512         # Target chunk size in tokens
DOCUMENT_QA_CHUNK_OVERLAP=50       # Overlap between chunks
```

## Process

1. **Extraction Phase**: Convert documents to Markdown using MarkItDown
   - Preserves tables, lists, and structure
   - Extracts images for optional Vision LLM processing

2. **Image Processing**: Filter and describe complex images
   - Smart filtering skips ~95% of images (icons, logos, decorative)
   - Only diagrams, charts, and complex visuals sent to Vision LLM

3. **Chunking Phase**: Semantic chunking with structure awareness
   - Never splits tables or code blocks mid-content
   - Respects heading boundaries
   - Adds overlap for context continuity

4. **Indexing Phase**: Create embeddings and index
   - Uses sentence-transformers for embeddings
   - Supports both vector and BM25 retrieval

5. **Retrieval Phase**: Hybrid search with RRF fusion
   - Vector retrieval for semantic similarity
   - BM25 retrieval for keyword matching
   - Reciprocal Rank Fusion combines results

6. **Reranking Phase**: LLM-as-judge scoring
   - Each chunk scored 0-10 for relevance
   - Top-k highest scored chunks selected
   - Reasoning provided for transparency

7. **Context Assembly**: Strategic positioning
   - Most relevant content at START and END
   - Addresses "lost in the middle" problem
   - Source citations injected inline

8. **Generation Phase**: Answer with citations
   - LLM generates answer using assembled context
   - [src_XXX] citations included for verification

## Example Usage

### Basic Q&A

```python
result = document_qa(
    documents=["quarterly_report.pdf"],
    query="What were the main revenue drivers in Q3?"
)
```

### Multi-Document Research

```python
result = document_qa(
    documents=[
        "financial_report.pdf",
        "board_presentation.pptx",
        "sales_data.xlsx"
    ],
    query="How do the revenue projections compare to actual results?",
    rerank_top_k=10,
    max_context_tokens=8000
)
```

### Without Image Processing (Faster)

```python
result = document_qa(
    documents=["document.pdf"],
    query="What is the executive summary?",
    process_images=False
)
```

## Integration with Deep Research

This skill can be used alongside the `deep_research` skill for hybrid research:

```python
from skills.deep_research import deep_research

# Combine document and web research
result = deep_research(
    topic="Q3 earnings analysis",
    documents=["earnings_report.pdf"],  # Local documents
    min_sources=10                       # Also search web
)
```

## Safety Considerations

- **File System Access**: Reads local files specified in `documents` parameter
- **External API Calls**: Calls OpenAI API for embeddings, reranking, Vision LLM, and generation
- **Data Privacy**: Document content is sent to OpenAI API - ensure compliance with data policies
- **Rate Limiting**: Large documents with many images may incur significant API costs

## Supported Formats

| Format | Extension | Features |
|--------|-----------|----------|
| PDF | .pdf | Full text, tables, page tracking |
| Word | .docx | Full text, tables, sections |
| Excel | .xlsx | Sheet names, tables, row tracking |
| PowerPoint | .pptx | Slides, text, images |
| HTML | .html, .htm | Full content |
| CSV | .csv | Tabular data |

## Troubleshooting

### "MarkItDown not installed"
```bash
pip install markitdown
```

### "sentence-transformers not available"
```bash
pip install sentence-transformers
```

### "OPENAI_API_KEY not set"
```bash
export OPENAI_API_KEY=sk-your-key-here
```

### Large documents timing out
- Reduce `chunk_size` for faster processing
- Set `process_images=False` to skip Vision LLM
- Split large documents into smaller files

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Document Q&A Skill                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐ │
│  │  MarkItDown  │──▶│   Vision     │──▶│   Semantic   │──▶│   Vector +   │ │
│  │  Converter   │   │   Describer  │   │   Chunker    │   │   BM25 Index │ │
│  └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘ │
│                                                                              │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐ │
│  │   Hybrid     │──▶│  LLM-as-     │──▶│   Context    │──▶│   Response   │ │
│  │  Retriever   │   │   Judge      │   │  Assembler   │   │  Generator   │ │
│  └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘ │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                     Source Tracker (Unified)                          │  │
│  │        [src_001] doc.pdf:p3  │  [src_002] slides.pptx:s5             │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```
