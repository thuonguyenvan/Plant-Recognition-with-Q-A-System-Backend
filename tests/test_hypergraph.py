"""
Test OG-RAG Retrieval
Verify HyperGraph indexing and test search functionality
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_settings
from services.embedding_service import VietnameseEmbeddingService
from services.vector_db_service import SupabaseVectorDB


def test_hypergraph():
    """Test HyperGraph retrieval"""
    print("\n" + "="*60)
    print("Testing OG-RAG HyperGraph Retrieval")
    print("="*60 + "\n")
    
    # Initialize services
    print("Initializing services...")
    settings = get_settings()
    embed_service = VietnameseEmbeddingService()
    vector_db = SupabaseVectorDB(
        url=settings.supabase_url,
        key=settings.supabase_anon_key
    )
    
    # Check database
    node_count = vector_db.count_nodes()
    print(f"Total HyperNodes in database: {node_count}\n")
    
    # Test queries
    test_queries = [
        "chữa ho",
        "bổ thận tráng dương",
        "đau khớp viêm khớp",
        "trị sốt rét",
        "lợi tiểu tiêu sưng"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: '{query}'")
        print(f"{'='*60}")
        
        # Embed query
        query_emb = embed_service.embed_text(query)
        
        # Search by value (most relevant for use cases)
        results = vector_db.search_by_value(
            query_embedding=query_emb,
            top_k=5,
            threshold=0.5
        )
        
        print(f"\nTop {len(results)} results:\n")
        for i, result in enumerate(results, 1):
            print(f"{i}. [Sim: {result['similarity']:.3f}] {result['plant_name']}")
            print(f"   {result['key']}: {result['value'][:100]}...")
            if result.get('section'):
                print(f"   Section: {result['section']}")
            print()
    
    # Test plant-specific query
    print(f"\n{'='*60}")
    print(f"Test: Get all info about 'Sâm cau'")
    print(f"{'='*60}\n")
    
    sam_cau_nodes = vector_db.get_plant_nodes("Sâm cau")
    print(f"Found {len(sam_cau_nodes)} nodes for Sâm cau")
    
    if sam_cau_nodes:
        print("\nSample nodes:")
        for i, node in enumerate(sam_cau_nodes[:5], 1):
            print(f"{i}. {node['key']}: {node['value'][:60]}...")
    
    print(f"\n{'='*60}")
    print("✅ HyperGraph test complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    test_hypergraph()
