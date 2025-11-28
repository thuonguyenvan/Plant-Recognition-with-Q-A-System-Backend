"""
Build HyperGraph and Index to Supabase
Converts flat facts to HyperNodes with embeddings and stores in vector DB
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from typing import List, Dict
from tqdm import tqdm

from config import get_settings
from services.embedding_service import VietnameseEmbeddingService
from services.vector_db_service import SupabaseVectorDB


def build_and_index_hypergraph(
    facts_file: str = "plant_facts.json",
    batch_size: int = 100
):
    """
    Build HyperGraph from facts and index to Supabase
    
    Args:
        facts_file: Path to flattened facts JSON
        batch_size: Batch size for embedding and insertion
    """
    print(f"\n{'='*60}")
    print(f"Building HyperGraph and Indexing to Supabase")
    print(f"{'='*60}\n")
    
    # Load facts
    print(f"Loading facts from {facts_file}...")
    with open(facts_file, 'r', encoding='utf-8') as f:
        facts = json.load(f)
    print(f"Loaded {len(facts)} facts")
    
    # Initialize services
    print("\nInitializing services...")
    settings = get_settings()
    embed_service = VietnameseEmbeddingService()
    vector_db = SupabaseVectorDB(
        url=settings.supabase_url,
        key=settings.supabase_anon_key
    )
    
    # Check existing count
    existing_count = vector_db.count_nodes()
    if existing_count > 0:
        print(f"\n⚠️  Warning: Database already has {existing_count} nodes")
        response = input("Clear existing nodes? (yes/no): ")
        if response.lower() == 'yes':
            print("Clearing database...")
            vector_db.clear_all_nodes()
            print("✅ Database cleared")
    
    # Build HyperNodes
    print(f"\nBuilding HyperNodes from facts...")
    hypernodes = []
    
    for fact in tqdm(facts, desc="Processing facts"):
        # Extract plant name and section
        plant_name = fact.get("Tên", "")
        section = fact.get("Mục", "")
        chunk_id = fact.get("_chunk_id", 0)
        is_chunked = fact.get("_is_chunked", False)
        
        # Process each key-value pair (except metadata)
        for key, value in fact.items():
            if key.startswith("_") or key in ["Tên", "Mục"]:
                continue
            
            # Create HyperNode data
            hypernode = {
                "key": key,
                "value": str(value),
                "plant_name": plant_name,
                "section": section if section else None,
                "chunk_id": chunk_id,
                "is_chunked": is_chunked
            }
            hypernodes.append(hypernode)
    
    print(f"Generated {len(hypernodes)} HyperNodes")
    
    # Embed and index in batches
    print(f"\nEmbedding and indexing (batch size: {batch_size})...")
    
    for i in tqdm(range(0, len(hypernodes), batch_size), desc="Indexing batches"):
        batch = hypernodes[i:i+batch_size]
        
        # Extract texts for batch embedding
        keys = [node["key"] for node in batch]
        values = [node["value"] for node in batch]
        
        # Batch embed
        key_embeddings = embed_service.embed_batch(keys, batch_size=len(keys))
        value_embeddings = embed_service.embed_batch(values, batch_size=len(values))
        
        # Add embeddings to nodes
        nodes_with_embeddings = []
        for j, node in enumerate(batch):
            node["key_embedding"] = key_embeddings[j]
            node["value_embedding"] = value_embeddings[j]
            nodes_with_embeddings.append(node)
        
        # Batch insert to Supabase
        try:
            vector_db.insert_hypernodes_batch(nodes_with_embeddings)
        except Exception as e:
            error_msg = str(e)
            # Check if it's a duplicate or other error
            if "duplicate" in error_msg.lower():
                print(f"\n⚠️ Skipping batch {i//batch_size} (duplicates)")
            else:
                print(f"\n❌ Error in batch {i//batch_size}: {error_msg}")
                print("Retrying with smaller batches...")
                # Retry in smaller chunks
                for j in range(0, len(nodes_with_embeddings), 10):
                    mini_batch = nodes_with_embeddings[j:j+10]
                    try:
                        vector_db.insert_hypernodes_batch(mini_batch)
                    except Exception as e2:
                        print(f"  Failed mini-batch at {j}: {str(e2)[:100]}")
    
    # Final statistics
    final_count = vector_db.count_nodes()
    
    print(f"\n{'='*60}")
    print(f"INDEXING COMPLETE")
    print(f"{'='*60}")
    print(f"Total HyperNodes indexed: {final_count}")
    print(f"Expected: {len(hypernodes)}")
    print(f"Success rate: {final_count/len(hypernodes)*100:.1f}%")
    print(f"{'='*60}\n")
    
    # Test search
    print("Testing vector search...")
    test_query = "chữa ho"
    query_emb = embed_service.embed_text(test_query)
    results = vector_db.search_by_value(query_emb, top_k=5, threshold=0.3)
    
    print(f"\nQuery: '{test_query}'")
    print(f"Top {len(results)} results:")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['plant_name']} - {result['key']}: "
              f"{result['value'][:60]}... (sim: {result['similarity']:.3f})")
    
    print(f"\n✅ HyperGraph successfully built and indexed!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build and index HyperGraph")
    parser.add_argument("--facts", default="plant_facts.json", help="Path to facts JSON")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size")
    
    args = parser.parse_args()
    
    build_and_index_hypergraph(args.facts, args.batch_size)
