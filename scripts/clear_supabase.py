"""
Clear all hypernodes from Supabase before re-importing
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.vector_db_service import SupabaseVectorDB
from config import get_settings

def main():
    settings = get_settings()
    vector_db = SupabaseVectorDB(
        url=settings.supabase_url,
        key=settings.supabase_anon_key
    )
    
    # Count current nodes
    count = vector_db.count_nodes()
    print(f"Current nodes in database: {count}")
    
    if count == 0:
        print("Database is already empty!")
        return
    
    # Confirm deletion
    confirm = input(f"\n⚠️  This will DELETE ALL {count} nodes. Continue? (yes/no): ")
    
    if confirm.lower() != 'yes':
        print("Cancelled.")
        return
    
    print("\nClearing nodes in batches (to avoid timeout)...")
    
    batch_size = 100
    deleted_total = 0
    
    while True:
        # Delete first 100 nodes
        try:
            result = vector_db.client.table('hypernodes')\
                .delete()\
                .limit(batch_size)\
                .execute()
            
            deleted = len(result.data) if result.data else 0
            
            if deleted == 0:
                break
            
            deleted_total += deleted
            print(f"  Deleted {deleted} nodes (total: {deleted_total}/{count})")
            
        except Exception as e:
            print(f"Error: {e}")
            break
    
    print(f"\n✅ Deleted {deleted_total} nodes")
    
    # Verify
    new_count = vector_db.count_nodes()
    print(f"Remaining nodes: {new_count}")

if __name__ == "__main__":
    main()
