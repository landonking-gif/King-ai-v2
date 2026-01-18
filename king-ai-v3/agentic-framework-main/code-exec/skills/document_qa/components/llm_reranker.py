"""
LLM Reranker Component for Haystack 2.x.

Module: code-exec/skills/document_qa/components/llm_reranker.py

Reranks retrieved chunks using LLM-as-judge scoring.
No cross-encoder model - pure LLM-based relevance assessment.

Uses LLM adapters for provider-agnostic LLM calls.
Configuration from .env (OPENAI_MODEL, ANTHROPIC_MODEL, etc.) with runtime overrides.
"""

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from haystack import Document, component

# Add adapters to path for import
_repo_root = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from adapters.llm import create_sync_adapter, SyncLLMWrapper

logger = logging.getLogger(__name__)


@component
class LLMReranker:
    """
    Rerank retrieved chunks using LLM-as-judge scoring.

    Strategy:
    - Initial retrieval: top-20 from hybrid search
    - LLM scores each on 0-10 scale
    - Return top_k highest scored
    - Include reasoning for transparency

    No cross-encoder model - uses the same LLM configured for the framework.
    """

    RERANK_PROMPT = """You are a relevance judge. Score how relevant this passage is to the query.

Query: {query}

Passage:
{passage}

Instructions:
1. Score relevance from 0-10 (10 = perfectly relevant, 0 = not relevant at all)
2. Consider: Does the passage answer the query? Does it contain key information?
3. Be strict - only score high if the passage directly addresses the query

Respond with ONLY a JSON object:
{{"score": <0-10>, "reason": "<brief 1-sentence explanation>"}}"""

    BATCH_RERANK_PROMPT = """You are a relevance judge. Score how relevant each passage is to the query.

Query: {query}

Passages:
{passages}

Instructions:
1. Score each passage from 0-10 (10 = perfectly relevant, 0 = not relevant)
2. Consider: Does it answer the query? Does it contain key information?
3. Be strict in scoring

Respond with ONLY a JSON array:
[{{"id": 1, "score": <0-10>, "reason": "<brief explanation>"}}, ...]"""

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        top_k: int = 5,
        batch_size: int = 5,
        min_score: float = 3.0,
    ):
        """
        Initialize the LLM Reranker.

        Args:
            llm_client: LLM client for reranking (uses OpenAI if not provided)
            top_k: Number of top documents to return
            batch_size: Number of documents to score in one LLM call
            min_score: Minimum score to include in results
        """
        self.llm_client = llm_client
        self.top_k = top_k
        self.batch_size = batch_size
        self.min_score = min_score

        # Statistics
        self._total_scored = 0
        self._llm_calls = 0

    @component.output_types(documents=List[Document])
    def run(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None,
    ) -> Dict[str, List[Document]]:
        """
        Rerank documents using LLM-as-judge.

        Args:
            query: Query to rank against
            documents: Documents to rerank
            top_k: Override default top_k

        Returns:
            Dictionary with 'documents' key containing reranked documents
        """
        if not documents:
            return {"documents": []}

        top_k = top_k or self.top_k

        # Score all documents
        scored_docs = self._score_documents(query, documents)

        # Sort by score descending
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Filter by minimum score and take top_k
        filtered = [
            (doc, score, reason)
            for doc, score, reason in scored_docs
            if score >= self.min_score
        ][:top_k]

        # Update document metadata with scores
        result_docs = []
        for doc, score, reason in filtered:
            doc.meta["rerank_score"] = score
            doc.meta["rerank_reason"] = reason
            result_docs.append(doc)

        logger.info(
            f"Reranked {len(documents)} docs -> {len(result_docs)} "
            f"(top_k={top_k}, min_score={self.min_score})"
        )

        return {"documents": result_docs}

    def _score_documents(
        self, query: str, documents: List[Document]
    ) -> List[Tuple[Document, float, str]]:
        """Score all documents, optionally in batches."""
        results: List[Tuple[Document, float, str]] = []

        # Batch scoring for efficiency
        if self.batch_size > 1 and len(documents) > self.batch_size:
            for i in range(0, len(documents), self.batch_size):
                batch = documents[i : i + self.batch_size]
                batch_results = self._score_batch(query, batch)
                results.extend(batch_results)
        else:
            # Score individually
            for doc in documents:
                score, reason = self._score_single(query, doc)
                results.append((doc, score, reason))
                self._total_scored += 1

        return results

    def _score_single(self, query: str, document: Document) -> Tuple[float, str]:
        """Score a single document."""
        passage = document.content[:1500]  # Limit passage length

        prompt = self.RERANK_PROMPT.format(
            query=query,
            passage=passage,
        )

        try:
            response = self._call_llm(prompt)
            self._llm_calls += 1

            # Parse JSON response
            score, reason = self._parse_score_response(response)
            return score, reason

        except Exception as e:
            logger.warning(f"Failed to score document: {e}")
            return 0.0, "scoring_failed"

    def _score_batch(
        self, query: str, documents: List[Document]
    ) -> List[Tuple[Document, float, str]]:
        """Score a batch of documents in one LLM call."""
        passages = []
        for i, doc in enumerate(documents, 1):
            passage = doc.content[:800]  # Shorter for batch
            passages.append(f"[{i}] {passage}")

        prompt = self.BATCH_RERANK_PROMPT.format(
            query=query,
            passages="\n\n".join(passages),
        )

        try:
            response = self._call_llm(prompt)
            self._llm_calls += 1

            # Parse batch response
            scores = self._parse_batch_response(response, len(documents))

            results = []
            for i, doc in enumerate(documents):
                score_data = scores.get(i + 1, {"score": 0, "reason": "not_scored"})
                results.append(
                    (doc, score_data["score"], score_data.get("reason", ""))
                )
                self._total_scored += 1

            return results

        except Exception as e:
            logger.warning(f"Batch scoring failed: {e}")
            # Fall back to individual scoring
            return [(doc, 0.0, "batch_failed") for doc in documents]

    def _call_llm(self, prompt: str) -> str:
        """
        Call the LLM using adapter factory.

        Uses the configured LLM adapter (from .env or runtime override).
        Supports all providers: OpenAI, Anthropic, Azure, Gemini, Local, vLLM.
        """
        if self.llm_client:
            return self._call_with_client(prompt)
        else:
            return self._call_with_adapter(prompt)

    def _call_with_client(self, prompt: str) -> str:
        """Use provided LLM client."""
        if isinstance(self.llm_client, SyncLLMWrapper):
            return self.llm_client.complete_text(prompt, temperature=0, max_tokens=200)
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
        else:
            raise ValueError("LLM client doesn't have complete or chat method")

    def _call_with_adapter(self, prompt: str) -> str:
        """
        Use LLM adapter factory.

        Creates adapter from .env configuration (provider-agnostic).
        """
        try:
            adapter = create_sync_adapter()
            logger.debug(f"LLMReranker using adapter with model: {adapter.model}")
            return adapter.complete_text(prompt, temperature=0, max_tokens=200)
        except Exception as e:
            logger.error(f"Adapter creation failed: {e}")
            raise

    def _parse_score_response(self, response: str) -> Tuple[float, str]:
        """Parse single score response."""
        try:
            # Try to extract JSON from response
            json_match = re.search(r"\{[^}]+\}", response)
            if json_match:
                data = json.loads(json_match.group())
                score = float(data.get("score", 0))
                reason = data.get("reason", "")
                return min(10, max(0, score)), reason

        except (json.JSONDecodeError, ValueError) as e:
            logger.debug(f"Failed to parse score response: {e}")

        # Fallback: try to extract just a number
        numbers = re.findall(r"\b(\d+(?:\.\d+)?)\b", response)
        if numbers:
            score = float(numbers[0])
            return min(10, max(0, score)), "parsed_from_text"

        return 0.0, "parse_failed"

    def _parse_batch_response(
        self, response: str, expected_count: int
    ) -> Dict[int, Dict[str, Any]]:
        """Parse batch score response."""
        try:
            # Try to extract JSON array
            json_match = re.search(r"\[[\s\S]*\]", response)
            if json_match:
                data = json.loads(json_match.group())
                results = {}
                for item in data:
                    doc_id = item.get("id", 0)
                    score = float(item.get("score", 0))
                    reason = item.get("reason", "")
                    results[doc_id] = {
                        "score": min(10, max(0, score)),
                        "reason": reason,
                    }
                return results

        except (json.JSONDecodeError, ValueError) as e:
            logger.debug(f"Failed to parse batch response: {e}")

        return {}

    def get_statistics(self) -> Dict[str, Any]:
        """Get reranking statistics."""
        return {
            "total_scored": self._total_scored,
            "llm_calls": self._llm_calls,
            "top_k": self.top_k,
            "min_score": self.min_score,
        }


def rerank_documents(
    query: str,
    documents: List[Document],
    top_k: int = 5,
) -> List[Document]:
    """
    Convenience function to rerank documents.

    Args:
        query: Query to rank against
        documents: Documents to rerank
        top_k: Number of top documents to return

    Returns:
        Reranked documents
    """
    reranker = LLMReranker(top_k=top_k)
    result = reranker.run(query=query, documents=documents)
    return result["documents"]
