"""
Fast import using CORRECT Supabase Connection URI (Tokyo Region)
"""

import sys
import json
import os
import psycopg2
from psycopg2.extras import execute_values
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================
# DATABASE CONNECTION CONFIGURATION
# ============================================================
# Get connection URI from environment variable
CONNECTION_URI = os.getenv("SUPABASE_DB_URI")

if not CONNECTION_URI:
    raise ValueError(
        "SUPABASE_DB_URI not found in environment variables. "
        "Please set it in your .env file."
    )

def import_with_psycopg2(
    embeddings_file="plant_hypernodes_with_embeddings.json",
    batch_size=1000
):
    print(f"\n{'='*60}")
    print(f"Fast Import via Supabase Pooler (Tokyo Region)")
    print(f"{'='*60}\n")

    # 1. Load embeddings
    print(f"Loading {embeddings_file}...")
    try:
        with open(embeddings_file, "r", encoding="utf-8") as f:
            hypernodes = json.load(f)
        print(f"Loaded {len(hypernodes)} HyperNodes\n")
    except FileNotFoundError:
        print(f"❌ Error: File '{embeddings_file}' not found.")
        return

    # 2. Connect
    print(f"Connecting to Database via URI...")
    print(f"Target: aws-1-ap-northeast-1.pooler.supabase.com (Tokyo)")
    
    try:
        # Kết nối bằng chuỗi URI đầy đủ
        conn = psycopg2.connect(dsn=CONNECTION_URI)
        cursor = conn.cursor()
        print("✅ Connected successfully!\n")

        # Check existing count
        cursor.execute("SELECT COUNT(*) FROM hypernodes")
        existing_count = cursor.fetchone()[0]

        if existing_count > 0:
            print(f"⚠️  Warning: Database has {existing_count} existing nodes")
            print("Clearing existing data...")
            cursor.execute("TRUNCATE TABLE hypernodes")
            conn.commit()
            print("✅ Cleared\n")

        # 3. Insert batches
        print(f"Inserting {len(hypernodes)} nodes in batches of {batch_size}...")

        insert_query = """
            INSERT INTO hypernodes (key, value, plant_name, section, key_embedding, value_embedding)
            VALUES %s
        """

        for i in tqdm(range(0, len(hypernodes), batch_size), desc="Inserting"):
            batch = hypernodes[i : i + batch_size]
            values = [
                (
                    node["key"],
                    node["value"],
                    node["plant_name"],
                    node.get("section"),
                    node["key_embedding"],
                    node["value_embedding"],
                )
                for node in batch
            ]

            try:
                execute_values(cursor, insert_query, values)
                conn.commit()
            except Exception as e:
                print(f"\n❌ Error in batch {i//batch_size}: {e}")
                conn.rollback()

        # Final verify
        cursor.execute("SELECT COUNT(*) FROM hypernodes")
        final_count = cursor.fetchone()[0]

        cursor.close()
        conn.close()

        print(f"\n{'='*60}")
        print("IMPORT COMPLETE")
        print(f"Total nodes in DB: {final_count}")
        print(f"{'='*60}\n")

    except psycopg2.OperationalError as e:
        print(f"\n❌ CONNECTION ERROR: {e}")
        print("Nếu vẫn lỗi, hãy kiểm tra lại xem Project có bị 'Paused' trên Dashboard không.")
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", default="plant_hypernodes_with_embeddings.json")
    parser.add_argument("--batch-size", type=int, default=1000)
    args = parser.parse_args()

    import_with_psycopg2(args.embeddings, args.batch_size)