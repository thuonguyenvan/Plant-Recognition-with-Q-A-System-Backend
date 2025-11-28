"""
Vietnamese Embedding Service using AITeamVN/Vietnamese_Embedding
"""
from sentence_transformers import SentenceTransformer
from typing import List, Union
import numpy as np
from functools import lru_cache


class VietnameseEmbeddingService:
    """Service for generating Vietnamese text embeddings"""
    
    def __init__(self, model_name: str = "AITeamVN/Vietnamese_Embedding"):
        """
        Initialize embedding service
        
        Args:
            model_name: HuggingFace model name
        """
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.dimension = 1024  # Vietnamese_Embedding actual dimension is 1024
        print(f"Model loaded successfully. Dimension: {self.dimension}")
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for single text
        
        Args:
            text: Input text in Vietnamese
            
        Returns:
            Embedding vector as list of floats
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generate embeddings for multiple texts (more efficient)
        
        Args:
            texts: List of input texts
            batch_size: Batch size for encoding
            
        Returns:
            List of embedding vectors
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 100
        )
        return embeddings.tolist()
    
    def similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Similarity score (0 to 1)
        """
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        return float(dot_product / (norm1 * norm2))


@lru_cache()
def get_embedding_service() -> VietnameseEmbeddingService:
    """Get cached embedding service instance"""
    return VietnameseEmbeddingService()
