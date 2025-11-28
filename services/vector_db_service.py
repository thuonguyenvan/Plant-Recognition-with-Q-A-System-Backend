"""
Supabase Vector Database Service for HyperNodes storage and retrieval
"""
from supabase import create_client, Client
from typing import List, Dict, Optional, Any
from functools import lru_cache


class SupabaseVectorDB:
    """Service for interacting with Supabase pgvector database"""
    
    def __init__(self, url: str, key: str, timeout: int = 120):
        """
        Initialize Supabase client with extended timeout
        
        Args:
            url: Supabase project URL
            key: Supabase anon key
            timeout: Request timeout in seconds (default: 120 for vector search)
        """
        # Create client with default settings
        # Note: Supabase Python client doesn't support custom timeout in options
        # Timeout is handled at HTTP client level
        self.client: Client = create_client(url, key)
        self.timeout = timeout
        print(f"Connected to Supabase: {url} (timeout: {timeout}s)")
    
    def insert_hypernode(self, node_data: Dict[str, Any]) -> Dict:
        """
        Insert a single hypernode
        
        Args:
            node_data: Dictionary containing node data
                Required fields: key, value, key_embedding, value_embedding, plant_name
                
        Returns:
            Inserted record
        """
        result = self.client.table('hypernodes').insert(node_data).execute()
        return result.data[0] if result.data else None
    
    def insert_hypernodes_batch(self, nodes: List[Dict[str, Any]]) -> List[Dict]:
        """
        Insert multiple hypernodes in batch
        
        Args:
            nodes: List of node data dictionaries
            
        Returns:
            List of inserted records
        """
        result = self.client.table('hypernodes').insert(nodes).execute()
        return result.data
    
    def search_by_key(
        self,
        query_embedding: List[float],
        top_k: int = 10,  # Reduced from 20 for better performance
        threshold: float = 0.4,  # Lowered threshold
        plant_filter: Optional[str] = None,
        retry_count: int = 2
    ) -> List[Dict]:
        """
        Search hypernodes by key embedding similarity with retry
        
        Args:
            query_embedding: Query vector (1024 dim)
            top_k: Number of results (default: 10 for performance)
            threshold: Minimum similarity threshold
            plant_filter: Optional plant name filter
            retry_count: Number of retries on timeout
            
        Returns:
            List of matching hypernodes with similarity scores
        """
        for attempt in range(retry_count + 1):
            try:
                # Build RPC params
                rpc_params = {
                    'query_embedding': query_embedding,
                    'match_threshold': threshold,
                    'match_count': top_k
                }
                
                # Add plant filter if specified
                if plant_filter:
                    rpc_params['filter_plant_name'] = plant_filter
                
                result = self.client.rpc(
                    'match_hypernodes_by_key',
                    rpc_params
                ).execute()
                
                nodes = result.data
                
                return nodes
            except Exception as e:
                if 'timeout' in str(e).lower() and attempt < retry_count:
                    print(f"Timeout on attempt {attempt + 1}, retrying with reduced top_k...")
                    top_k = max(5, top_k // 2)  # Reduce top_k on retry
                    continue
                elif 'timeout' in str(e).lower():
                    # All retries failed - return empty instead of raising
                    print(f"⚠️ All retries timed out. Returning empty results.")
                    return []
                else:
                    raise
    
    def search_by_value(
        self,
        query_embedding: List[float],
        top_k: int = 10,  # Reduced from 20
        threshold: float = 0.4,
        plant_filter: Optional[str] = None,
        retry_count: int = 2
    ) -> List[Dict]:
        """
        Search hypernodes by value embedding similarity with retry
        
        Args:
            query_embedding: Query vector (1024 dim)
            top_k: Number of results (default: 10 for performance)
            threshold: Minimum similarity threshold  
            plant_filter: Optional plant name filter
            retry_count: Number of retries on timeout
            
        Returns:
            List of matching hypernodes with similarity scores
        """
        for attempt in range(retry_count + 1):
            try:
                # Build RPC params
                rpc_params = {
                    'query_embedding': query_embedding,
                    'match_threshold': threshold,
                    'match_count': top_k
                }
                
                # Add plant filter if specified
                if plant_filter:
                    rpc_params['filter_plant_name'] = plant_filter
                    print(f"[VectorDB] Calling RPC with plant_filter='{plant_filter}'")
                else:
                    print(f"[VectorDB] Calling RPC without plant_filter")
                
                result = self.client.rpc(
                    'match_hypernodes_by_value',
                    rpc_params
                ).execute()
                
                nodes = result.data
                print(f"[VectorDB] RPC returned {len(nodes)} results")
                
                return nodes
            except Exception as e:
                print(f"[VectorDB ERROR] {type(e).__name__}: {str(e)}")
                if 'timeout' in str(e).lower() and attempt < retry_count:
                    print(f"Timeout on attempt {attempt + 1}, retrying with reduced top_k...")
                    top_k = max(5, top_k // 2)  # Reduce top_k on retry
                    continue
                elif 'timeout' in str(e).lower():
                    # All retries failed - return empty instead of raising
                    print(f"⚠️ All retries timed out. Returning empty results.")
                    return []
                else:
                    raise
    
    def search_combined(
        self,
        query_embedding: List[float],
        top_k: int = 20,
        threshold: float = 0.5,
        key_weight: float = 0.5
    ) -> List[Dict]:
        """
        Search hypernodes using combined key+value similarity
        
        Args:
            query_embedding: Query vector (768 dim)
            top_k: Number of results to return
            threshold: Minimum similarity threshold (0-1)
            key_weight: Weight for key similarity (0-1), value gets (1-key_weight)
            
        Returns:
            List of matching hypernodes with combined similarity scores
        """
        result = self.client.rpc(
            'match_hypernodes_combined',
            {
                'query_embedding': query_embedding,
                'match_threshold': threshold,
                'match_count': top_k,
                'key_weight': key_weight
            }
        ).execute()
        
        return result.data
    
    def get_plant_nodes(self, plant_name: str) -> List[Dict]:
        """
        Get all hypernodes for a specific plant
        
        Args:
            plant_name: Name of the plant
            
        Returns:
            List of all hypernodes for this plant
        """
        result = self.client.table('hypernodes')\
            .select('*')\
            .eq('plant_name', plant_name)\
            .execute()
        
        return result.data
    
    def count_nodes(self) -> int:
        """Get total number of hypernodes in database"""
        result = self.client.table('hypernodes')\
            .select('id', count='exact')\
            .execute()
        
        return result.count
    
    def clear_all_nodes(self):
        """Delete all hypernodes (use with caution!)"""
        result = self.client.table('hypernodes').delete().neq('id', 0).execute()
        return result.data


@lru_cache()
def get_vector_db(url: str, key: str) -> SupabaseVectorDB:
    """Get cached vector DB instance"""
    return SupabaseVectorDB(url, key)
