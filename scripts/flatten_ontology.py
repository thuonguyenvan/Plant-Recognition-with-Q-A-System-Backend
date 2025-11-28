"""
Ontology Flattener
Converts hierarchical JSON-LD plant data to flat fact lists for OG-RAG HyperGraph
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import List, Dict, Any
import json
from utils.key_normalizer import normalize_key
from utils.chunker import chunk_long_value, estimate_tokens
from utils.data_loader import PlantDataLoader


def flatten_plant_ontology(
    plant_data: Dict[str, Any],
    chunk_threshold: int = 250
) -> List[Dict[str, Any]]:
    """
    Convert nested JSON-LD to flat fact list with intelligent chunking
    
    Args:
        plant_data: Nested plant ontology data
        chunk_threshold: Maximum tokens before chunking (default: 250)
        
    Returns:
        List of flat facts suitable for HyperGraph
    """
    facts = []
    plant_name = plant_data.get("ten", "")
    
    if not plant_name:
        return facts
    
    # 1. Basic Info (always keep together, no chunking)
    basic_fact = {
        "Tên": plant_name,
        "Tên khoa học": plant_data.get("ten_khoa_hoc", ""),
        "Họ": plant_data.get("ho", "")
    }
    # Remove empty values
    basic_fact = {k: v for k, v in basic_fact.items() if v}
    if basic_fact:
        basic_fact["_is_chunked"] = False
        facts.append(basic_fact)
    
    # 2. Process each section
    sections = [
        "Mô tả", "Phân bố", 
        "Công dụng", "Cách dùng", "Bộ phận dùng",
        "Thông tin khác"
    ]
    
    for section in sections:
        if section not in plant_data:
            continue
        
        section_data = plant_data[section]
        
        if not isinstance(section_data, dict):
            continue
        
        # Process each field in section
        for field_key, field_value in section_data.items():
            if not field_value or field_value == "":
                continue
            
            # Normalize key to Vietnamese
            normalized_key = normalize_key(field_key)
            
            # Convert to string
            value_str = str(field_value)
            
            # Check if chunking needed
            if estimate_tokens(value_str) > chunk_threshold:
                # CHUNK IT!
                chunks = chunk_long_value(
                    normalized_key,
                    value_str,
                    max_tokens=chunk_threshold
                )
                
                for chunk_key, chunk_value, chunk_id in chunks:
                    fact = {
                        "Tên": plant_name,
                        "Mục": normalize_key(section),
                        chunk_key: chunk_value,
                        "_chunk_id": chunk_id,
                        "_is_chunked": True
                    }
                    facts.append(fact)
            else:
                # No chunking needed
                fact = {
                    "Tên": plant_name,
                    "Mục": normalize_key(section),
                    normalized_key: value_str,
                    "_is_chunked": False
                }
                facts.append(fact)
    
    return facts


def build_all_plant_facts(
    data_dir: str = "data",
    output_file: str = "plant_facts.json",
    chunk_threshold: int = 250
) -> List[Dict]:
    """
    Process all plants and generate flat facts
    
    Args:
        data_dir: Directory containing JSON-LD files
        output_file: Output file for facts (optional)
        chunk_threshold: Token threshold for chunking
        
    Returns:
        List of all facts from all plants
    """
    from tqdm import tqdm
    
    loader = PlantDataLoader(data_dir)
    all_facts = []
    
    jsonld_files = sorted(Path(data_dir).glob("ontology_node_*.jsonld"))
    
    print(f"\nProcessing {len(jsonld_files)} plant files...")
    
    for jsonld_file in tqdm(jsonld_files, desc="Flattening plants"):
        # Load plant data
        plant_data = loader._load_jsonld_file(jsonld_file)
        
        if not plant_data:
            continue
        
        # Flatten + chunk
        plant_facts = flatten_plant_ontology(plant_data, chunk_threshold)
        all_facts.extend(plant_facts)
    
    # Save if output file specified
    if output_file:
        print(f"\nSaving {len(all_facts)} facts to {output_file}...")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_facts, f, ensure_ascii=False, indent=2)
    
    # Print statistics
    print(f"\n{'='*60}")
    print(f"STATISTICS")
    print(f"{'='*60}")
    print(f"Total plants processed: {len(jsonld_files)}")
    print(f"Total facts generated: {len(all_facts)}")
    print(f"Avg facts per plant: {len(all_facts) / len(jsonld_files):.1f}")
    
    chunked = [f for f in all_facts if f.get("_is_chunked", False)]
    print(f"Chunked facts: {len(chunked)} ({len(chunked)/len(all_facts)*100:.1f}%)")
    print(f"Unchunked facts: {len(all_facts) - len(chunked)}")
    
    # Section coverage
    sections = [f.get("Mục") for f in all_facts if "Mục" in f]
    section_counts = {}
    for section in sections:
        section_counts[section] = section_counts.get(section, 0) + 1
    
    print(f"\nSection coverage:")
    for section, count in sorted(section_counts.items(), key=lambda x: -x[1]):
        print(f"  {section}: {count}")
    
    print(f"{'='*60}\n")
    
    return all_facts


if __name__ == "__main__":
    import sys
    
    # Allow optional arguments
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "plant_facts.json"
    
    facts = build_all_plant_facts(data_dir, output_file)
    
    print(f"✅ Done! Generated {len(facts)} facts")
    print(f"📄 Saved to {output_file}")
