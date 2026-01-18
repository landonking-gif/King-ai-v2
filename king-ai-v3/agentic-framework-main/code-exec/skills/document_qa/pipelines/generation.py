"""
Generation Pipeline for Document Q&A.

Module: code-exec/skills/document_qa/pipelines/generation.py

Haystack 2.x pipeline for Q&A generation:
1. Hybrid retrieval (Vector + BM25)
2. Reciprocal Rank Fusion
3. LLM-as-judge reranking
4. Context assembly
5. LLM generation

Uses LLM adapters for provider-agnostic LLM calls.
Configuration from .env (OPENAI_MODEL, ANTHROPIC_MODEL, etc.) with runtime overrides.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from haystack import Document, Pipeline, component
from haystack.components.joiners import DocumentJoiner
from haystack.document_stores.in_memory import InMemoryDocumentStore

# Add adapters to path for import
_repo_root = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from adapters.llm import create_sync_adapter, SyncLLMWrapper

from ..components.context_assembler import ContextAssembler
from ..components.llm_reranker import LLMReranker
from ..utils.source_tracker import SourceTracker

logger = logging.getLogger(__name__)


def create_generation_pipeline(
    document_store: InMemoryDocumentStore,
    llm_client: Optional[Any] = None,
    retrieval_top_k: int = 15,
    rerank_top_k: int = 5,
    max_context_tokens: int = 6000,
) -> Pipeline:
    """
    Create the Q&A generation pipeline.

    Pipeline flow:
    query -> [VectorRetriever, BM25Retriever] -> DocumentJoiner (RRF)
                                                       |
                                                       v
                                                  LLMReranker
                                                       |
                                                       v
                                               ContextAssembler
                                                       |
                                                       v
                                                  LLMGenerator

    Args:
        document_store: Document store with indexed documents
        llm_client: Optional LLM client (uses OpenAI if not provided)
        retrieval_top_k: Number of docs from each retriever
        rerank_top_k: Number of docs after reranking
        max_context_tokens: Maximum context tokens

    Returns:
        Configured Haystack Pipeline
    """
    pipeline = Pipeline()

    # Add retrievers
    try:
        from haystack.components.retrievers import (
            InMemoryBM25Retriever,
            InMemoryEmbeddingRetriever,
        )

        pipeline.add_component(
            "vector_retriever",
            InMemoryEmbeddingRetriever(
                document_store=document_store,
                top_k=retrieval_top_k,
            ),
        )

        pipeline.add_component(
            "bm25_retriever",
            InMemoryBM25Retriever(
                document_store=document_store,
                top_k=retrieval_top_k,
            ),
        )
    except ImportError:
        logger.warning("Haystack retrievers not available, using fallback")
        # Add simple fallback retriever
        pipeline.add_component(
            "vector_retriever",
            SimpleFallbackRetriever(document_store, retrieval_top_k),
        )
        pipeline.add_component(
            "bm25_retriever",
            SimpleFallbackRetriever(document_store, retrieval_top_k),
        )

    # Add document joiner for RRF fusion
    pipeline.add_component(
        "joiner",
        DocumentJoiner(join_mode="reciprocal_rank_fusion"),
    )

    # Add LLM reranker
    pipeline.add_component(
        "reranker",
        LLMReranker(llm_client=llm_client, top_k=rerank_top_k),
    )

    # Add context assembler
    pipeline.add_component(
        "assembler",
        ContextAssembler(max_tokens=max_context_tokens),
    )

    # Add LLM generator
    pipeline.add_component(
        "generator",
        LLMGenerator(llm_client=llm_client),
    )

    # Connect components
    pipeline.connect("vector_retriever", "joiner")
    pipeline.connect("bm25_retriever", "joiner")
    pipeline.connect("joiner", "reranker.documents")
    pipeline.connect("reranker", "assembler.documents")
    pipeline.connect("assembler.context", "generator.context")
    pipeline.connect("assembler.sources", "generator.sources")

    return pipeline


def run_generation(
    query: str,
    document_store: InMemoryDocumentStore,
    llm_client: Optional[Any] = None,
    retrieval_top_k: int = 15,
    rerank_top_k: int = 5,
    max_context_tokens: int = 6000,
    source_tracker: Optional[SourceTracker] = None,
) -> Dict[str, Any]:
    """
    Run the Q&A generation pipeline.

    Args:
        query: User question
        document_store: Document store with indexed documents
        llm_client: Optional LLM client
        retrieval_top_k: Number of docs from each retriever
        rerank_top_k: Number of docs after reranking
        max_context_tokens: Maximum context tokens
        source_tracker: Optional source tracker

    Returns:
        Dictionary with 'answer', 'sources', 'confidence', and 'statistics'
    """
    # Create query embedding
    from .indexing import create_query_embedding

    query_embedding = create_query_embedding(query)

    pipeline = create_generation_pipeline(
        document_store=document_store,
        llm_client=llm_client,
        retrieval_top_k=retrieval_top_k,
        rerank_top_k=rerank_top_k,
        max_context_tokens=max_context_tokens,
    )

    tracker = source_tracker or SourceTracker()

    # Build input
    input_data = {
        "vector_retriever": {"query_embedding": query_embedding},
        "bm25_retriever": {"query": query},
        "reranker": {"query": query},
        "assembler": {"source_tracker": tracker},
        "generator": {"query": query},
    }

    # Run pipeline
    result = pipeline.run(input_data)

    # Extract results
    generator_result = result.get("generator", {})
    assembler_result = result.get("assembler", {})

    answer = generator_result.get("answer", "")
    confidence = generator_result.get("confidence", 0.0)
    sources = assembler_result.get("sources", [])

    statistics = {
        "query": query,
        "chunks_retrieved": retrieval_top_k * 2,
        "chunks_after_rerank": rerank_top_k,
        "sources_cited": len(sources),
    }

    logger.info(
        f"Generation complete: {len(sources)} sources cited, "
        f"confidence={confidence:.2f}"
    )

    return {
        "answer": answer,
        "sources": sources,
        "confidence": confidence,
        "statistics": statistics,
    }


@component
class LLMGenerator:
    """
    LLM-based answer generator component.

    Takes assembled context and generates an answer with source citations.
    """

    GENERATION_PROMPT = """Answer the following question based ONLY on the provided context.
Include source citations in your answer using the [src_XXX] format provided in the context.

Question: {query}

Context:
{context}

Instructions:
1. Answer the question directly and comprehensively
2. Use ONLY information from the context
3. Include [src_XXX] citations for facts and claims
4. If the context doesn't contain enough information, say so
5. Be concise but thorough

Answer:"""

    def __init__(self, llm_client: Optional[Any] = None):
        """Initialize the generator."""
        self.llm_client = llm_client

    @component.output_types(answer=str, confidence=float)
    def run(
        self,
        query: str,
        context: str,
        sources: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Generate an answer from the context.

        Args:
            query: User question
            context: Assembled context with citations
            sources: Source metadata

        Returns:
            Dictionary with 'answer' and 'confidence'
        """
        prompt = self.GENERATION_PROMPT.format(query=query, context=context)

        try:
            answer = self._call_llm(prompt)

            # Estimate confidence based on citation usage
            citation_count = answer.count("[src_")
            confidence = min(1.0, 0.5 + (citation_count * 0.1))

            return {
                "answer": answer,
                "confidence": confidence,
            }

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return {
                "answer": f"Error generating answer: {e}",
                "confidence": 0.0,
            }

    def _call_llm(self, prompt: str) -> str:
        """
        Call the LLM using adapter factory.

        Uses the configured LLM adapter (from .env or runtime override).
        Supports all providers: OpenAI, Anthropic, Azure, Gemini, Local, vLLM.
        """
        # If a custom client was provided, use it
        if self.llm_client:
            if isinstance(self.llm_client, SyncLLMWrapper):
                return self.llm_client.complete_text(prompt, temperature=0.3, max_tokens=2000)
            elif hasattr(self.llm_client, "complete_text"):
                return self.llm_client.complete_text(prompt)
            elif hasattr(self.llm_client, "complete"):
                response = self.llm_client.complete(prompt)
                return response if isinstance(response, str) else str(response)
            elif hasattr(self.llm_client, "chat"):
                messages = [{"role": "user", "content": prompt}]
                response = self.llm_client.chat(messages)
                if isinstance(response, dict):
                    return response.get("content", "")
                return str(response)

        # Use adapter factory - reads from .env by default
        try:
            adapter = create_sync_adapter()
            logger.info(f"LLMGenerator using adapter with model: {adapter.model}")
            return adapter.complete_text(prompt, temperature=0.3, max_tokens=2000)
        except Exception as e:
            logger.error(f"Adapter creation failed: {e}")
            raise


@component
class SimpleFallbackRetriever:
    """Simple fallback retriever when Haystack retrievers are not available."""

    def __init__(self, document_store: InMemoryDocumentStore, top_k: int):
        self.document_store = document_store
        self.top_k = top_k

    @component.output_types(documents=List[Document])
    def run(self, **kwargs) -> Dict[str, List[Document]]:
        """Return all documents (no actual retrieval)."""
        # This is a fallback - in production, Haystack retrievers should be used
        try:
            # Try to get documents from store
            docs = list(self.document_store.storage.values())[: self.top_k]
            return {"documents": docs}
        except Exception:
            return {"documents": []}
