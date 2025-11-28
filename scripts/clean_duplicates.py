"""
Clean Duplicate Nodes from Supabase
Memory-efficient approach using batches
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_settings
from services.vector_db_service import SupabaseVectorDB
from tqdm import tqdm


def clean_duplicates():
    """Clean duplicate nodes efficiently"""
    print("\n" + "="*60)
    print("Cleaning Duplicate Nodes")
    print("="*60 + "\n")
    
    # Initialize
    settings = get_settings()
    vector_db = SupabaseVectorDB(
        url=settings.supabase_url,
        key=settings.supabase_anon_key
    )
    
    # Check current count
    initial_count = vector_db.count_nodes()
    print(f"Current node count: {initial_count}")
    
    # Get all nodes using pagination
    print("\nFetching all nodes (this may take a while)...")
    all_nodes_data = []
    page_size = 1000
    offset = 0
    
    while True:
        batch = vector_db.client.table('hypernodes')\
            .select('id, key, value, plant_name')\
            .range(offset, offset + page_size - 1)\
            .execute()
        
        if not batch.data:
            break
        
        all_nodes_data.extend(batch.data)
        offset += page_size
        print(f"  Fetched {len(all_nodes_data)} nodes so far...")
    
    nodes = all_nodes_data
    print(f"\nTotal fetched: {len(nodes)} nodes")
    
    # Find duplicates
    print("\nIdentifying duplicates...")
    seen = {}
    duplicates_to_delete = []
    
    for node in tqdm(nodes, desc="Processing nodes"):
        key_tuple = (node['key'], node['value'], node['plant_name'])
        
        if key_tuple in seen:
            # This is a duplicate, mark for deletion
            # Keep the one with lower ID
            if node['id'] > seen[key_tuple]['id']:
                duplicates_to_delete.append(node['id'])
            else:
                duplicates_to_delete.append(seen[key_tuple]['id'])
                seen[key_tuple] = node
        else:
            seen[key_tuple] = node
    
    print(f"\nFound {len(duplicates_to_delete)} duplicate nodes to delete")
    
    if len(duplicates_to_delete) == 0:
        print("✅ No duplicates found!")
        return
    
    # Confirm deletion
    response = input(f"\nDelete {len(duplicates_to_delete)} duplicate nodes? (yes/no): ")
    if response.lower() != 'yes':
        print("Cancelled")
        return
    
    # Delete in batches
    batch_size = 100
    deleted_count = 0
    
    print(f"\nDeleting duplicates in batches of {batch_size}...")
    for i in tqdm(range(0, len(duplicates_to_delete), batch_size), desc="Deleting batches"):
        batch = duplicates_to_delete[i:i+batch_size]
        
        try:
            vector_db.client.table('hypernodes')\
                .delete()\
                .in_('id', batch)\
                .execute()
            deleted_count += len(batch)
        except Exception as e:
            print(f"\nError deleting batch at index {i}: {e}")
    
    # Final count
    final_count = vector_db.count_nodes()
    
    print(f"\n{'='*60}")
    print("CLEANUP COMPLETE")
    print(f"{'='*60}")
    print(f"Initial nodes: {initial_count}")
    print(f"Deleted duplicates: {deleted_count}")
    print(f"Final nodes: {final_count}")
    print(f"Expected: {len(seen)}")
    print(f"{'='*60}\n")
    
    if final_count == len(seen):
        print("✅ Cleanup successful!")
    else:
        print("⚠️ Final count doesn't match expected. May need to run again.")


if __name__ == "__main__":
    clean_duplicates()
