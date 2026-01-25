"""
Embedding Generation Module.

Module: memory-service/service/embedding.py
Uses sentence-transformers for generating embeddings, with mock fallback.
"""

from typing import Any, Dict, List, Optional
import hashlib
import numpy as np

# Try to import sentence_transformers, fall back to mock if not available
try:
    import tiktoken
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    tiktoken = None

from .config import Settings


class EmbeddingGenerator:
    """Generate embeddings for artifacts using sentence-transformers or mock."""

    def __init__(self, settings: Settings) -> None:
        """
        Initialize embedding generator.

        Args:
            settings: Service configuration settings
        """
        self.settings = settings
        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
        self.use_mock = not SENTENCE_TRANSFORMERS_AVAILABLE

    async def initialize(self) -> None:
        """Initialize the embedding model."""
        if self.use_mock:
            print("Using mock embedding generator (sentence-transformers not available)")
            return

        try:
            # Initialize sentence transformer model
            self.model = SentenceTransformer('all-MiniLM-L6-v2')

            # Initialize tokenizer for token counting
            if tiktoken:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
            print("Embedding generator initialized with sentence-transformers")
        except Exception as e:
            print(f"Failed to initialize sentence-transformers: {e}")
            print("Falling back to mock embedding generator")
            self.use_mock = True

    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text.

        Args:
            text: Input text to embed

        Returns:
            Embedding vector
        """
        if self.use_mock:
            # Generate deterministic mock embedding based on text hash
            hash_obj = hashlib.md5(text.encode())
            hash_int = int(hash_obj.hexdigest(), 16)
            np.random.seed(hash_int % 2**32)  # Use hash as seed for reproducibility
            embedding = np.random.normal(0, 1, self.settings.embedding_dimension).tolist()
            return embedding

        if not self.model:
            raise RuntimeError("Embedding model not initialized")

        # Generate embedding using sentence-transformers
        embedding = self.model.encode(text, convert_to_tensor=False).tolist()
        return embedding

    async def generate_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of input texts

        Returns:
            List of embedding vectors
        """
        if self.use_mock:
            # Generate mock embeddings for batch
            embeddings = []
            for text in texts:
                hash_obj = hashlib.md5(text.encode())
                hash_int = int(hash_obj.hexdigest(), 16)
                np.random.seed(hash_int % 2**32)
                embedding = np.random.normal(0, 1, self.settings.embedding_dimension).tolist()
                embeddings.append(embedding)
            return embeddings

        if not self.model:
            raise RuntimeError("Embedding model not initialized")

        # Generate embeddings using sentence-transformers
        embeddings = self.model.encode(texts, convert_to_tensor=False, show_progress_bar=False)
        return embeddings.tolist()

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Input text

        Returns:
            Number of tokens
        """
        if self.use_mock or not tiktoken:
            # Rough approximation: 1 token per 4 characters
            return len(text) // 4 + 1

        if not self.tokenizer:
            raise RuntimeError("Tokenizer not initialized")

        return len(self.tokenizer.encode(text))

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
