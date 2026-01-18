"""
Embedding Generation Module.

Module: memory-service/service/embedding.py
Uses sentence-transformers for generating embeddings.
"""

from typing import Any, Dict, List, Optional
import tiktoken

import anyio
from sentence_transformers import SentenceTransformer

from .config import Settings


class EmbeddingGenerator:
    """Generate embeddings for artifacts using sentence-transformers."""

    def __init__(self, settings: Settings) -> None:
        """
        Initialize embedding generator.

        Args:
            settings: Service configuration settings
        """
        self.settings = settings
        self.model: Optional[SentenceTransformer] = None
        self.tokenizer: Optional[tiktoken.Encoding] = None

    async def initialize(self) -> None:
        """Load embedding model."""

        def _load_model() -> SentenceTransformer:
            return SentenceTransformer(self.settings.embedding_model)

        self.model = await anyio.to_thread.run_sync(_load_model)

        # Initialize tokenizer for token counting
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            # Fallback if tiktoken not available
            self.tokenizer = None

    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector for text.

        Args:
            text: Input text

        Returns:
            Embedding vector as list of floats
        """
        if not self.model:
            raise RuntimeError("Embedding model not initialized")

        def _encode() -> Any:
            return self.model.encode(text, convert_to_numpy=True)

        embedding = await anyio.to_thread.run_sync(_encode)
        return embedding.tolist()

    async def generate_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of input texts

        Returns:
            List of embedding vectors
        """
        if not self.model:
            raise RuntimeError("Embedding model not initialized")

        def _encode_batch() -> Any:
            return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

        embeddings = await anyio.to_thread.run_sync(_encode_batch)
        return embeddings.tolist()

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Input text

        Returns:
            Token count
        """
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Rough approximation: 1 token ≈ 4 characters
            return len(text) // 4

    def extract_searchable_text(self, artifact_content: Dict[str, Any]) -> str:
        """
        Extract searchable text from artifact content.

        Args:
            artifact_content: Artifact content dictionary

        Returns:
            Concatenated searchable text
        """
        text_parts = []

        # Extract common fields
        for key in ["text", "summary", "description", "content", "claim_text"]:
            if key in artifact_content and artifact_content[key]:
                text_parts.append(str(artifact_content[key]))

        # Extract tags
        if "tags" in artifact_content and artifact_content["tags"]:
            text_parts.append(" ".join(artifact_content["tags"]))

        # Extract source information
        if "source" in artifact_content and isinstance(artifact_content["source"], dict):
            source = artifact_content["source"]
            for key in ["url", "title", "doc_id"]:
                if key in source and source[key]:
                    text_parts.append(str(source[key]))

        return " ".join(text_parts)

    async def generate_artifact_embedding(
        self, artifact_content: Dict[str, Any]
    ) -> tuple[List[float], int]:
        """
        Generate embedding for artifact content.

        Args:
            artifact_content: Artifact content dictionary

        Returns:
            Tuple of (embedding vector, token count)
        """
        searchable_text = self.extract_searchable_text(artifact_content)
        embedding = await self.generate_embedding(searchable_text)
        token_count = self.count_tokens(searchable_text)
        return embedding, token_count

    def get_embedding_dimension(self) -> int:
        """
        Get embedding vector dimension.

        Returns:
            Embedding dimension
        """
        return self.settings.embedding_dimension


class TokenBudgetManager:
    """Manage token budgets for memory compaction."""

    def __init__(self, settings: Settings) -> None:
        """
        Initialize token budget manager.

        Args:
            settings: Service configuration settings
        """
        self.settings = settings
        self.threshold = settings.memory_compaction_threshold_tokens

    def needs_compaction(self, current_tokens: int) -> bool:
        """
        Check if memory needs compaction.

        Args:
            current_tokens: Current token count

        Returns:
            True if compaction needed
        """
        return current_tokens > self.threshold

    def calculate_target_tokens(self, strategy: str = "summarize") -> int:
        """
        Calculate target token count after compaction.

        Args:
            strategy: Compaction strategy

        Returns:
            Target token count
        """
        if strategy == "summarize":
            # Target 75% of threshold
            return int(self.threshold * 0.75)
        elif strategy == "truncate":
            # Target 50% of threshold
            return int(self.threshold * 0.5)
        else:
            return self.threshold

    def prioritize_artifacts(
        self, artifacts: List[Dict[str, Any]], preserve_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Prioritize artifacts for compaction.

        Args:
            artifacts: List of artifacts with metadata
            preserve_ids: IDs to always preserve

        Returns:
            Sorted list of artifacts (most important first)
        """
        # Separate preserved and compactable artifacts
        preserved = [a for a in artifacts if a["id"] in preserve_ids]
        compactable = [a for a in artifacts if a["id"] not in preserve_ids]

        # Sort compactable by recency and importance
        # Priority: 1) Recent artifacts, 2) High confidence, 3) Referenced by others
        compactable.sort(
            key=lambda a: (
                a.get("created_at", ""),
                a.get("confidence", 0),
                len(a.get("references", [])),
            ),
            reverse=True,
        )

        return preserved + compactable
