"""
OG-RAG Query Engine Service
Wrapper around Supabase vector search for semantic retrieval
"""
from typing import List, Dict, Any, Optional
from services.embedding_service import VietnameseEmbeddingService
from services.vector_db_service import SupabaseVectorDB


class OGRAGQueryEngine:
    """OG-RAG query engine using HyperGraph in Supabase"""
    
    def __init__(
        self,
        embed_service: VietnameseEmbeddingService,
        vector_db: SupabaseVectorDB,
        top_k: int = 20,
        similarity_threshold: float = 0.3  # Lowered from 0.5 to allow less strict matching
    ):
        """
        Initialize OG-RAG query engine
        
        Args:
            embed_service: Embedding service
            vector_db: Vector database service
            top_k: Number of results to retrieve
            similarity_threshold: Minimum similarity score
        """
        self.embed_service = embed_service
        self.vector_db = vector_db
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
    
    def query(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        plant_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        OG-RAG Two-Stage Retrieval
        
        Stage 1: Search by KEY embeddings to find relevant attributes
        Stage 2: Search by VALUE embeddings for content matching
        Combine and re-rank results
        
        Args:
            query_text: Query string
            top_k: Override default top_k
            plant_filter: Filter by specific plant name
            
        Returns:
            List of relevant HyperNodes with combined similarity scores
        """
        k = top_k or self.top_k
        query_embedding = self.embed_service.embed_text(query_text)
        
        # STAGE 1: Search by KEY (attribute names)
        # This finds relevant attributes like "Công dụng y học", "Phân bố", etc.
        # REDUCED multiplier to 2x (was 3x) to avoid timeout with 21k+ nodes
        print(f"[OG-RAG] Stage 1: Searching by KEY embeddings...")
        key_results = self.vector_db.search_by_key(
            query_embedding=query_embedding,
            top_k=k * 2,  # Reduced from k*3 to avoid timeout
            threshold=self.similarity_threshold,
            plant_filter=plant_filter
        )
        print(f"[OG-RAG] Stage 1: Found {len(key_results)} nodes by key")
        
        # STAGE 2: Search by VALUE (attribute content)
        # This finds relevant content regardless of attribute type
        # REDUCED multiplier to 2x (was 3x) to avoid timeout
        print(f"[OG-RAG] Stage 2: Searching by VALUE embeddings...")
        value_results = self.vector_db.search_by_value(
            query_embedding=query_embedding,
            top_k=k * 2,  # Reduced from k*3 to avoid timeout
            threshold=self.similarity_threshold,
            plant_filter=plant_filter
        )
        print(f"[OG-RAG] Stage 2: Found {len(value_results)} nodes by value")
        
        # MERGE & RE-RANK: Combine results with weighted scoring
        combined = self._merge_and_rerank(
            key_results=key_results,
            value_results=value_results,
            top_k=k,
            key_weight=0.3,  # Keys are important for filtering
            value_weight=0.7  # Values are more important for content
        )
        
        print(f"[OG-RAG] Final: Returning {len(combined)} nodes after merge\n")
        return combined
    
    def _merge_and_rerank(
        self,
        key_results: List[Dict],
        value_results: List[Dict],
        top_k: int,
        key_weight: float = 0.3,
        value_weight: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Merge and re-rank results from key and value searches
        
        Uses a simple weighted combination:
        final_score = key_weight * key_similarity + value_weight * value_similarity
        
        Args:
            key_results: Results from key search
            value_results: Results from value search
            top_k: Number of results to return
            key_weight: Weight for key similarity (0-1)
            value_weight: Weight for value similarity (0-1)
            
        Returns:
            Merged and re-ranked list of nodes
        """
        # Build index by node ID
        nodes = {}
        
        # Add key results
        for node in key_results:
            node_id = node['id']
            nodes[node_id] = {
                **node,
                'key_similarity': node.get('similarity', 0),
                'value_similarity': 0,
                'combined_score': 0
            }
        
        # Add/update value results
        for node in value_results:
            node_id = node['id']
            if node_id in nodes:
                # Node found in both searches - update value similarity
                nodes[node_id]['value_similarity'] = node.get('similarity', 0)
            else:
                # New node from value search only
                nodes[node_id] = {
                    **node,
                    'key_similarity': 0,
                    'value_similarity': node.get('similarity', 0),
                    'combined_score': 0
                }
        
        # Calculate combined scores
        for node_id, node in nodes.items():
            key_sim = node['key_similarity']
            val_sim = node['value_similarity']
            node['combined_score'] = key_weight * key_sim + value_weight * val_sim
            # Keep original similarity for backward compatibility
            node['similarity'] = node['combined_score']
        
        # Sort by combined score and return top-k
        sorted_nodes = sorted(
            nodes.values(),
            key=lambda x: x['combined_score'],
            reverse=True
        )
        
        return sorted_nodes[:top_k]
    
    def get_plant_context(self, plant_name: str) -> Dict[str, Any]:
        """
        Get full context for a specific plant
        
        Args:
            plant_name: Plant name (Vietnamese)
            
        Returns:
            Dictionary with all plant information organized by sections
        """
        nodes = self.vector_db.get_plant_nodes(plant_name)
        
        # Organize by section
        context = {
            "plant_name": plant_name,
            "sections": {}
        }
        
        for node in nodes:
            section = node.get('section', 'General')
            if section not in context['sections']:
                context['sections'][section] = []
            
            context['sections'][section].append({
                "key": node['key'],
                "value": node['value'],
                "is_chunked": node.get('is_chunked', False),
                "chunk_id": node.get('chunk_id', 0)
            })
        
        return context
    
    def build_rag_context(
        self,
        query_results: List[Dict[str, Any]],
        max_context_length: int = 2000
    ) -> str:
        """
        Build context string from query results
        
        Args:
            query_results: Results from query()
            max_context_length: Maximum context length in chars
            
        Returns:
            Formatted context string for LLM
        """
        context_parts = []
        current_length = 0
        
        # Group by plant
        plants = {}
        for result in query_results:
            plant_name = result['plant_name']
            if plant_name not in plants:
                plants[plant_name] = []
            plants[plant_name].append(result)
        
        # Build context
        for plant_name, nodes in plants.items():
            plant_context = f"\n## {plant_name}\n"
            
            for node in nodes:
                section = node.get('section', '')
                key = node['key']
                value = node['value']
                
                node_text = f"- **{key}** ({section}): {value}\n"
                
                if current_length + len(node_text) > max_context_length:
                    break
                
                plant_context += node_text
                current_length += len(node_text)
            
            context_parts.append(plant_context)
            
            if current_length >= max_context_length:
                break
        
        return "\n".join(context_parts)


def get_og_rag_engine(
    embed_service: VietnameseEmbeddingService,
    vector_db: SupabaseVectorDB
) -> OGRAGQueryEngine:
    """Factory function for OG-RAG engine"""
    return OGRAGQueryEngine(embed_service, vector_db)
