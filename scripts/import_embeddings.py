"""
Import Pre-generated Embeddings to Supabase
Use this after running generate_embeddings_kaggle.ipynb on Kaggle
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
from tqdm import tqdm

from config import get_settings
from services.vector_db_service import SupabaseVectorDB


def import_embeddings_from_json(
    embeddings_file: str = "plant_hypernodes_with_embeddings.json",
    batch_size: int = 200
):
    """
    Import pre-generated embeddings from JSON file
    
    Args:
        embeddings_file: Path to JSON file with HyperNodes + embeddings
        batch_size: Batch size for insertion
    """
    print(f"\n{'='*60}")
    print(f"Importing Pre-generated Embeddings to Supabase")
    print(f"{'='*60}\n")
    
    # Load hypernodes with embeddings
    print(f"Loading {embeddings_file}...")
    with open(embeddings_file, 'r', encoding='utf-8') as f:
        hypernodes = json.load(f)
    
    print(f"Loaded {len(hypernodes)} HyperNodes with embeddings")
    
    # Initialize Supabase
    print("\nConnecting to Supabase...")
    settings = get_settings()
    vector_db = SupabaseVectorDB(
        url=settings.supabase_url,
        key=settings.supabase_anon_key
    )
    
    # Check existing
    existing_count = vector_db.count_nodes()
    if existing_count > 0:
        print(f"\n⚠️  Warning: Database already has {existing_count} nodes")
        response = input("Clear existing nodes? (yes/no): ")
        if response.lower() == 'yes':
            print("Clearing database...")
            vector_db.clear_all_nodes()
            print("✅ Database cleared")
    
    # Insert in batches
    print(f"\nInserting {len(hypernodes)} nodes (batch size: {batch_size})...")
    
    for i in tqdm(range(0, len(hypernodes), batch_size), desc="Inserting batches"):
        batch = hypernodes[i:i+batch_size]
        
        try:
            vector_db.insert_hypernodes_batch(batch)
        except Exception as e:
            error_msg = str(e)
            if "duplicate" in error_msg.lower():
                print(f"\n⚠️ Skipping batch {i//batch_size} (duplicates)")
            else:
                print(f"\n❌ Error in batch {i//batch_size}: {error_msg}")
                print("Retrying with smaller batches...")
                # Retry in smaller chunks
                for j in range(0, len(batch), 10):
                    mini_batch = batch[j:j+10]
                    try:
                        vector_db.insert_hypernodes_batch(mini_batch)
                    except Exception as e2:
                        print(f"  Failed mini-batch at {j}: {str(e2)[:100]}")
    
    # Final statistics
    final_count = vector_db.count_nodes()
    
    print(f"\n{'='*60}")
    print(f"IMPORT COMPLETE")
    print(f"{'='*60}")
    print(f"Total HyperNodes in database: {final_count}")
    print(f"Expected: {len(hypernodes)}")
    print(f"Success rate: {final_count/len(hypernodes)*100:.1f}%")
    print(f"{'='*60}\n")
    
    print("✅ Embeddings successfully imported to Supabase!")


def import_embeddings_from_npz(
    embeddings_file: str = "plant_embeddings.npz",
    metadata_file: str = "plant_metadata.json",
    batch_size: int = 200
):
    """
    Import from compressed NumPy format
    
    Args:
        embeddings_file: Path to .npz file with embeddings
        metadata_file: Path to JSON file with node metadata
        batch_size: Batch size for insertion
    """
    print(f"\n{'='*60}")
    print(f"Importing from NPZ format")
    print(f"{'='*60}\n")
    
    # Load embeddings
    print(f"Loading {embeddings_file}...")
    data = np.load(embeddings_file)
    key_embeddings = data['key_embeddings']
    value_embeddings = data['value_embeddings']
    
    print(f"Loaded embeddings:")
    print(f"  Keys: {key_embeddings.shape}")
    print(f"  Values: {value_embeddings.shape}")
    
    # Load metadata
    print(f"\nLoading {metadata_file}...")
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    print(f"Loaded {len(metadata)} metadata entries")
    
    # Combine
    hypernodes = []
    for i, meta in enumerate(metadata):
        node = meta.copy()
        node['key_embedding'] = key_embeddings[i].tolist()
        node['value_embedding'] = value_embeddings[i].tolist()
        hypernodes.append(node)
    
    print(f"Combined into {len(hypernodes)} HyperNodes\n")
    
    # Use JSON import function
    import_embeddings_from_json.__wrapped__(hypernodes, batch_size)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Import pre-generated embeddings")
    parser.add_argument("--format", choices=['json', 'npz'], default='json',
                       help="Input format")
    parser.add_argument("--embeddings", default="plant_hypernodes_with_embeddings.json",
                       help="Path to embeddings file")
    parser.add_argument("--metadata", default="plant_metadata.json",
                       help="Path to metadata file (for NPZ format)")
    parser.add_argument("--batch-size", type=int, default=200,
                       help="Batch size for insertion")
    
    args = parser.parse_args()
    
    if args.format == 'json':
        import_embeddings_from_json(args.embeddings, args.batch_size)
    else:
        import_embeddings_from_npz(args.embeddings, args.metadata, args.batch_size)
