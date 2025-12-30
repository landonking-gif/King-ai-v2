"""
Pinecone integration for RAG (Retrieval Augmented Generation).
Stores and retrieves business data embeddings for semantic search.
"""

from pinecone import Pinecone
from config.settings import settings
from typing import List, Dict, Any

class VectorStore:
    """
    Manages semantic storage and retrieval of business-related documents.
    """
    
    def __init__(self):
        if settings.pinecone_api_key:
            self.pc = Pinecone(api_key=settings.pinecone_api_key)
            self.index_name = settings.pinecone_index
            
            # Ensure index exists (assumes it's pre-created or handled via admin)
            self.index = self.pc.Index(self.index_name)
        else:
            self.pc = None
            self.index = None

    async def upsert_business_summary(self, business_id: str, text: str, embedding: List[float]):
        """
        Stores a business summary and its embedding in Pinecone.
        """
        if not self.index:
            return None
            
        return self.index.upsert(
            vectors=[{
                "id": business_id,
                "values": embedding,
                "metadata": {
                    "text": text,
                    "type": "business_summary"
                }
            }]
        )

    async def search_similar_businesses(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Searches for businesses with similar semantic profiles.
        """
        if not self.index:
            return []
            
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        return [
            {
                "id": match.id,
                "score": match.score,
                "text": match.metadata.get("text", "")
            }
            for match in results.matches
        ]
    
    async def upsert(self, id: str, text: str, metadata: Dict[str, Any], embedding: List[float] = None):
        """
        Generic upsert method for storing any type of document.
        
        Args:
            id: Unique identifier for the document
            text: Text content to store
            metadata: Additional metadata
            embedding: Optional pre-computed embedding (if None, will need to be computed)
        """
        if not self.index:
            return None
        
        # For now, we'll store without embedding if not provided
        # In production, you'd want to compute embeddings here
        if embedding is None:
            # Mock embedding - in production, generate from text
            embedding = [0.0] * 1536  # Standard OpenAI embedding size
        
        metadata["text"] = text
        
        return self.index.upsert(
            vectors=[{
                "id": id,
                "values": embedding,
                "metadata": metadata
            }]
        )
