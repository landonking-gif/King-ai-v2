"""
Vector Database Adapter for Semantic Search.

Module: memory-service/service/storage/vector.py
Supports both Milvus (production) and ChromaDB (development).
"""

from typing import Any, Dict, List, Optional, Tuple
from abc import ABC, abstractmethod

import anyio

from ..config import Settings


class VectorDBAdapter(ABC):
    """Abstract base class for vector database adapters."""

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to vector database."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to vector database."""
        pass

    @abstractmethod
    async def create_collection(self, collection_name: str, dimension: int) -> None:
        """Create a collection/index for storing vectors."""
        pass

    @abstractmethod
    async def insert_vectors(
        self,
        collection_name: str,
        ids: List[str],
        vectors: List[List[float]],
        metadata: List[Dict[str, Any]],
    ) -> bool:
        """Insert vectors with metadata."""
        pass

    @abstractmethod
    async def search_vectors(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int,
        filter_expr: Optional[str] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar vectors. Returns list of (id, similarity, metadata)."""
        pass

    @abstractmethod
    async def delete_vectors(self, collection_name: str, ids: List[str]) -> bool:
        """Delete vectors by IDs."""
        pass


class ChromaAdapter(VectorDBAdapter):
    """ChromaDB adapter for development."""

    def __init__(self, settings: Settings) -> None:
        """
        Initialize ChromaDB adapter.

        Args:
            settings: Service configuration settings
        """
        self.settings = settings
        self.client: Optional[Any] = None
        self.collections: Dict[str, Any] = {}

    async def connect(self) -> None:
        """Establish connection to ChromaDB."""
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        # Run sync ChromaDB calls in thread pool
        def _connect() -> Any:
            return chromadb.PersistentClient(
                path=self.settings.chroma_path,
                settings=ChromaSettings(anonymized_telemetry=False),
            )

        self.client = await anyio.to_thread.run_sync(_connect)

    async def disconnect(self) -> None:
        """Close ChromaDB connection."""
        # ChromaDB doesn't require explicit disconnect
        self.client = None
        self.collections = {}

    async def create_collection(self, collection_name: str, dimension: int) -> None:
        """
        Create a collection in ChromaDB.

        Args:
            collection_name: Name of collection
            dimension: Vector dimension (not used by Chroma but kept for interface consistency)
        """
        if not self.client:
            raise RuntimeError("ChromaDB client not connected")

        def _create_collection() -> Any:
            return self.client.get_or_create_collection(
                name=collection_name, metadata={"hnsw:space": "cosine"}
            )

        collection = await anyio.to_thread.run_sync(_create_collection)
        self.collections[collection_name] = collection

    async def insert_vectors(
        self,
        collection_name: str,
        ids: List[str],
        vectors: List[List[float]],
        metadata: List[Dict[str, Any]],
    ) -> bool:
        """
        Insert vectors into ChromaDB collection.

        Args:
            collection_name: Collection name
            ids: List of IDs
            vectors: List of embedding vectors
            metadata: List of metadata dicts

        Returns:
            True if successful
        """
        if collection_name not in self.collections:
            await self.create_collection(collection_name, len(vectors[0]))

        collection = self.collections[collection_name]

        def _insert() -> None:
            collection.add(ids=ids, embeddings=vectors, metadatas=metadata)

        await anyio.to_thread.run_sync(_insert)
        return True

    async def search_vectors(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int,
        filter_expr: Optional[str] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search for similar vectors in ChromaDB.

        Args:
            collection_name: Collection name
            query_vector: Query embedding vector
            top_k: Number of results to return
            filter_expr: Optional filter expression (Chroma where clause)

        Returns:
            List of (id, similarity, metadata) tuples
        """
        if collection_name not in self.collections:
            return []

        collection = self.collections[collection_name]

        def _search() -> Dict[str, Any]:
            # ChromaDB filter format: {"metadata_key": "value"}
            where = None
            if filter_expr:
                # Simple parsing of filter_expr (production would need proper parser)
                try:
                    where = eval(filter_expr)  # Use with caution
                except Exception:
                    where = None

            return collection.query(
                query_embeddings=[query_vector], n_results=top_k, where=where
            )

        results = await anyio.to_thread.run_sync(_search)

        # Parse ChromaDB results
        output = []
        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                similarity = 1.0 - results["distances"][0][i]  # Convert distance to similarity
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                output.append((doc_id, similarity, metadata))

        return output

    async def delete_vectors(self, collection_name: str, ids: List[str]) -> bool:
        """
        Delete vectors from ChromaDB.

        Args:
            collection_name: Collection name
            ids: List of IDs to delete

        Returns:
            True if successful
        """
        if collection_name not in self.collections:
            return False

        collection = self.collections[collection_name]

        def _delete() -> None:
            collection.delete(ids=ids)

        await anyio.to_thread.run_sync(_delete)
        return True


class MilvusAdapter(VectorDBAdapter):
    """Milvus adapter for production."""

    def __init__(self, settings: Settings) -> None:
        """
        Initialize Milvus adapter.

        Args:
            settings: Service configuration settings
        """
        self.settings = settings
        self.connections: Optional[Any] = None

    async def connect(self) -> None:
        """Establish connection to Milvus."""
        from pymilvus import connections

        def _connect() -> None:
            uri = self.settings.milvus_url
            # Parse host and port from URL
            host = uri.replace("http://", "").replace("https://", "").split(":")[0]
            port = uri.split(":")[-1] if ":" in uri else "19530"

            connections.connect(alias="default", host=host, port=port)

        await anyio.to_thread.run_sync(_connect)
        self.connections = connections

    async def disconnect(self) -> None:
        """Close Milvus connection."""
        if self.connections:

            def _disconnect() -> None:
                self.connections.disconnect(alias="default")

            await anyio.to_thread.run_sync(_disconnect)

    async def create_collection(self, collection_name: str, dimension: int) -> None:
        """
        Create a collection in Milvus.

        Args:
            collection_name: Name of collection
            dimension: Vector dimension
        """
        from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, utility

        def _create() -> None:
            if utility.has_collection(collection_name):
                return

            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimension),
                FieldSchema(name="metadata", dtype=DataType.JSON),
            ]
            schema = CollectionSchema(fields, description="Artifact embeddings")
            collection = Collection(name=collection_name, schema=schema)

            # Create index for vector search
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024},
            }
            collection.create_index(field_name="embedding", index_params=index_params)

        await anyio.to_thread.run_sync(_create)

    async def insert_vectors(
        self,
        collection_name: str,
        ids: List[str],
        vectors: List[List[float]],
        metadata: List[Dict[str, Any]],
    ) -> bool:
        """
        Insert vectors into Milvus collection.

        Args:
            collection_name: Collection name
            ids: List of IDs
            vectors: List of embedding vectors
            metadata: List of metadata dicts

        Returns:
            True if successful
        """
        from pymilvus import Collection

        def _insert() -> None:
            collection = Collection(collection_name)
            data = [ids, vectors, metadata]
            collection.insert(data)
            collection.flush()

        await anyio.to_thread.run_sync(_insert)
        return True

    async def search_vectors(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int,
        filter_expr: Optional[str] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search for similar vectors in Milvus.

        Args:
            collection_name: Collection name
            query_vector: Query embedding vector
            top_k: Number of results to return
            filter_expr: Optional filter expression (Milvus expr format)

        Returns:
            List of (id, similarity, metadata) tuples
        """
        from pymilvus import Collection

        def _search() -> Any:
            collection = Collection(collection_name)
            collection.load()

            search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}

            results = collection.search(
                data=[query_vector],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                expr=filter_expr,
                output_fields=["metadata"],
            )
            return results

        results = await anyio.to_thread.run_sync(_search)

        output = []
        for hits in results:
            for hit in hits:
                output.append((hit.id, hit.distance, hit.entity.get("metadata", {})))

        return output

    async def delete_vectors(self, collection_name: str, ids: List[str]) -> bool:
        """
        Delete vectors from Milvus.

        Args:
            collection_name: Collection name
            ids: List of IDs to delete

        Returns:
            True if successful
        """
        from pymilvus import Collection

        def _delete() -> None:
            collection = Collection(collection_name)
            expr = f"id in {ids}"
            collection.delete(expr)

        await anyio.to_thread.run_sync(_delete)
        return True


def get_vector_adapter(settings: Settings) -> VectorDBAdapter:
    """
    Factory function to get appropriate vector DB adapter.

    Args:
        settings: Service configuration settings

    Returns:
        Vector database adapter instance
    """
    if settings.vector_db_type == "milvus":
        return MilvusAdapter(settings)
    else:
        return ChromaAdapter(settings)
