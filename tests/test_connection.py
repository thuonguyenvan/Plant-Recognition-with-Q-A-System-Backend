"""
Test script for Supabase and Embedding services
"""
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Set .env path explicitly
os.environ.setdefault('ENV_FILE', str(Path(__file__).parent.parent / '.env'))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / '.env')

from config import get_settings
from services.embedding_service import VietnameseEmbeddingService
from services.vector_db_service import SupabaseVectorDB


def test_embedding_service():
    """Test embedding service"""
    print("\n=== Testing Vietnamese Embedding Service ===")
    
    embed_service = VietnameseEmbeddingService()
    
    # Test single embedding
    test_text = "Sâm cau chữa tê thấp, đau khớp"
    print(f"\nTest text: '{test_text}'")
    
    embedding = embed_service.embed_text(test_text)
    print(f"Embedding dimension: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")
    
    # Test batch embedding
    texts = [
        "Công dụng y học",
        "Cách dùng",
        "Lưu ý khi sử dụng"
    ]
    print(f"\nBatch embedding {len(texts)} texts...")
    batch_embeddings = embed_service.embed_batch(texts)
    print(f"Generated {len(batch_embeddings)} embeddings")
    
    # Test similarity
    emb1 = embed_service.embed_text("chữa ho")
    emb2 = embed_service.embed_text("trị ho")
    emb3 = embed_service.embed_text("bổ thận")
    
    sim_12 = embed_service.similarity(emb1, emb2)
    sim_13 = embed_service.similarity(emb1, emb3)
    
    print(f"\nSimilarity 'chữa ho' vs 'trị ho': {sim_12:.4f}")
    print(f"Similarity 'chữa ho' vs 'bổ thận': {sim_13:.4f}")
    
    print("\n✅ Embedding service test passed!\n")
    return embed_service


def test_vector_db():
    """Test Supabase vector DB connection"""
    print("\n=== Testing Supabase Vector DB ===")
    
    settings = get_settings()
    vector_db = SupabaseVectorDB(
        url=settings.supabase_url,
        key=settings.supabase_anon_key
    )
    
    # Test count
    count = vector_db.count_nodes()
    print(f"Current node count: {count}")
    
    # Test insert
    print("\nInserting test hypernode...")
    embed_service = VietnameseEmbeddingService()
    
    test_node = {
        "key": "Tên",
        "value": "Sâm cau TEST",
        "key_embedding": embed_service.embed_text("Tên"),
        "value_embedding": embed_service.embed_text("Sâm cau TEST"),
        "plant_name": "TEST_PLANT",
        "section": "Basic Info",
        "chunk_id": 0,
        "is_chunked": False
    }
    
    inserted = vector_db.insert_hypernode(test_node)
    print(f"Inserted node ID: {inserted['id']}")
    
    # Test search
    print("\nTesting vector search...")
    query_embedding = embed_service.embed_text("Sâm cau")
    
    results = vector_db.search_by_value(
        query_embedding=query_embedding,
        top_k=5,
        threshold=0.3
    )
    
    print(f"Found {len(results)} results:")
    for i, result in enumerate(results[:3], 1):
        print(f"{i}. {result['plant_name']} - {result['key']}: {result['value'][:50]}... (sim: {result['similarity']:.4f})")
    
    # Cleanup
    print("\nCleaning up test node...")
    vector_db.client.table('hypernodes').delete().eq('plant_name', 'TEST_PLANT').execute()
    
    print("\n✅ Vector DB test passed!\n")
    return vector_db


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Plant Medicine RAG Backend Services")
    print("=" * 60)
    
    try:
        # Test embedding
        embed_service = test_embedding_service()
        
        # Test vector DB
        vector_db = test_vector_db()
        
        print("=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
