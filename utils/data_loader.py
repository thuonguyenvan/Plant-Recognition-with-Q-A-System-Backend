"""
JSON-LD Data Loader
Loads and processes plant ontology data from JSON-LD files
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from functools import lru_cache


class PlantDataLoader:
    """Loader for plant ontology JSON-LD data"""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize data loader
        
        Args:
            data_dir: Directory containing JSON-LD files
        """
        self.data_dir = Path(data_dir)
        self._class_to_file_cache = {}
        self._class_to_name_cache = {}
        
        # Load CV model class mapping (from CSV)
        self._cv_class_mapping = self._load_cv_class_mapping()
        
        self._build_index()
    
    def _load_cv_class_mapping(self) -> Dict[str, str]:
        """Load CV model class to Vietnamese name mapping from JSON"""
        try:
            mapping_file = Path("cv_class_to_vietnamese.json")
            if mapping_file.exists():
                with open(mapping_file, 'r', encoding='utf-8') as f:
                    import json
                    mapping = json.load(f)
                    print(f"Loaded CV class mapping: {len(mapping)} classes")
                    return mapping
        except Exception as e:
            print(f"Warning: Could not load CV mapping: {e}")
        
        return {}
    
    def _build_index(self):
        """Build index of all plants for fast lookup by multiple keys"""
        print(f"Building plant data index from {self.data_dir}...")
        
        jsonld_files = list(self.data_dir.glob("ontology_node_*.jsonld"))
        
        for jsonld_file in jsonld_files:
            try:
                plant_data = self._load_jsonld_file(jsonld_file)
                if plant_data:
                    plant_name = plant_data.get("ten", "")
                    scientific_name = plant_data.get("ten_khoa_hoc", "")
                    
                    # Create class name from scientific name
                    # e.g., "Centella asiatica (L.) Urb." -> "Centella_asiatica"
                    class_name = self._scientific_to_class(scientific_name)
                    
                    # Index by scientific underscore format (for CV API compatibility)
                    if class_name:
                        self._class_to_file_cache[class_name] = jsonld_file.name
                        self._class_to_name_cache[class_name] = plant_name
                    
                    # Also index by Vietnamese name for direct lookup
                    if plant_name:
                        self._class_to_file_cache[plant_name] = jsonld_file.name
                        # Add to name cache if not already there
                        if plant_name not in self._class_to_name_cache.values():
                            self._class_to_name_cache[plant_name] = plant_name
                    
                    # Also index by lowercase versions for case-insensitive lookup
                    if class_name:
                        self._class_to_file_cache[class_name.lower()] = jsonld_file.name
                    if plant_name:
                        self._class_to_file_cache[plant_name.lower()] = jsonld_file.name
                        
            except Exception as e:
                print(f"Warning: Failed to index {jsonld_file.name}: {e}")
        
        print(f"Indexed {len(set(self._class_to_file_cache.values()))} plants")
    
    @staticmethod
    def _scientific_to_class(scientific_name: str) -> str:
        """
        Convert scientific name to class name
        
        Handles corrupt data with extra spaces
        e.g., "Centella asiati ca (L.) Urb." -> "Centella_asiatica"
        
        Args:
            scientific_name: Scientific name string
            
        Returns:
            Class name in format "Genus_species"
        """
        if not scientific_name:
            return ""
        
        import re
        
        # Remove everything in parentheses (author citations)
        clean = re.sub(r'\([^)]*\)', '', scientific_name)
        
        # Normalize whitespace (multiple spaces → single space)
        clean = ' '.join(clean.split())
        
        # Extract words
        parts = clean.split()
        
        if len(parts) >= 2:
            genus = parts[0]
            # Only take the NEXT word(s) that look like species name (lowercase)
            # Stop at uppercase (author names like "Urb.")
            species_parts = []
            for part in parts[1:]:
                if part[0].isupper():  # Author name starts - stop here
                    break
                species_parts.append(part)
            
            if species_parts:
                species = ''.join(species_parts)  # Join "asiati" + "ca" -> "asiatica"
                return f"{genus}_{species}"
        
        # Fallback
        return parts[0] if parts else ""
    
    def _load_jsonld_file(self, file_path: Path) -> Optional[Dict]:
        """
        Load and extract plant data from JSON-LD file
        
        Merges ALL nodes from @graph into a single plant dictionary:
        - Plant node (metadata)
        - Mô tả node
        - Phân bố node
        - Công dụng node
        - etc.
        
        Args:
            file_path: Path to JSON-LD file
            
        Returns:
            Complete plant data dictionary with all sections merged
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract and merge all nodes from @graph
            if "@graph" in data:
                plant_data = {}
                
                for node in data["@graph"]:
                    if not isinstance(node, dict):
                        continue
                    
                    node_type = node.get("@type")
                    
                    if node_type == "Plant":
                        # Plant node is the base - copy all fields
                        plant_data.update(node)
                    elif node_type:
                        # Other nodes (Mô tả, Phân bố, Công dụng, etc.)
                        # Add as a section with the @type as key
                        section_data = {k: v for k, v in node.items() if k != "@type"}
                        
                        # Skip if section is empty or all null
                        if section_data and any(v is not None for v in section_data.values()):
                            plant_data[node_type] = section_data
                
                return plant_data if plant_data else None
            
            # If @graph doesn't exist, check if it's the plant node directly
            if data.get("@type") == "Plant":
                return data
                
            return None
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def get_plant_by_class(self, class_name: str) -> Optional[Dict]:
        """
        Get full plant data by class name
        
        Supports:
        - CV model class names (e.g., "Centella_asiatica") via CSV mapping
        - Vietnamese names (e.g., "Rau má") via cache
        - Scientific names via cache
        
        Args:
            class_name: e.g., "Centella_asiatica" or "Curculigo_orchioides" or "Sâm cau"
            
        Returns:
            Full plant data dictionary
        """
        # Try CV model class mapping first (most accurate)
        if class_name in self._cv_class_mapping:
            vietnamese_name = self._cv_class_mapping[class_name]
            return self.get_plant_by_name(vietnamese_name)
        
        # Fallback to cache lookup
        if class_name not in self._class_to_file_cache:
            return None
        
        file_name = self._class_to_file_cache[class_name]
        file_path = self.data_dir / file_name
        
        return self._load_jsonld_file(file_path)
    
    def get_plant_by_name(self, vietnamese_name: str) -> Optional[Dict]:
        """
        Get plant data by Vietnamese name
        
        Args:
            vietnamese_name: e.g., "Sâm cau"
            
        Returns:
            Full plant data dictionary
        """
        return self.get_plant_by_class(vietnamese_name)
    
    def get_all_plant_names(self) -> List[str]:
        """Get list of all Vietnamese plant names"""
        return list(set(self._class_to_name_cache.values()))
    
    def get_all_class_names(self) -> List[str]:
        """Get list of all class names"""
        return list(set(k for k in self._class_to_file_cache.keys() 
                       if k not in self._class_to_name_cache.values()))
    
    def count_plants(self) -> int:
        """Get total number of plants"""
        return len(set(self._class_to_file_cache.values()))


@lru_cache()
def get_plant_data_loader(data_dir: str = "data") -> PlantDataLoader:
    """Get cached plant data loader instance"""
    return PlantDataLoader(data_dir)


# Test if run directly
if __name__ == "__main__":
    loader = PlantDataLoader()
    
    print(f"\nTotal plants: {loader.count_plants()}")
    print(f"Sample plant names: {loader.get_all_plant_names()[:5]}")
    
    # Test loading a plant
    plant = loader.get_plant_by_class("Curculigo_orchioides")
    if plant:
        print(f"\nLoaded: {plant.get('ten')} ({plant.get('ten_khoa_hoc')})")
        print(f"Sections: {list(plant.keys())}")
