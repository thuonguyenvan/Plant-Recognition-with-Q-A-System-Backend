"""
Flow 1 Service: Image-Only Plant Classification
Returns top-5 predictions with plant summaries
"""
from typing import List, Dict, Optional
from services.cv_api_client import CVAPIClient
from utils.data_loader import PlantDataLoader


class Flow1Service:
    """Service for Flow 1: Image-only classification"""
    
    def __init__(
        self,
        cv_client: CVAPIClient,
        data_loader: PlantDataLoader
    ):
        self.cv_client = cv_client
        self.data_loader = data_loader
    
    def classify_and_summarize(
        self,
        image_path: Optional[str] = None,
        image_url: Optional[str] = None
    ) -> Dict:
        """
        Classify image and return top-5 with summaries
        
        Returns:
            {
                "predictions": [
                    {
                        "class_name": str,
                        "vietnamese_name": str,
                        "scientific_name": str,
                        "family": str,
                        "confidence": float,
                        "summary": {...}
                    },
                    ...
                ]
            }
        """
        # Get CV predictions
        predictions = self.cv_client.classify_image(
            image_path=image_path,
            image_url=image_url
        )
        
        # Enrich with plant data
        enriched_predictions = []
        
        for pred in predictions[:5]:  # Top 5
            class_name = pred['class_name']
            confidence = pred['confidence']
            
            # Load plant data
            plant_data = self.data_loader.get_plant_by_class(class_name)
            
            if plant_data:
                enriched_pred = {
                    "class_name": class_name,
                    "vietnamese_name": plant_data.get("ten", ""),
                    "scientific_name": plant_data.get("ten_khoa_hoc", ""),
                    "family": plant_data.get("ho", ""),
                    "confidence": confidence,
                    "summary": self._generate_summary(plant_data)
                }
            else:
                # Fallback if no data found
                enriched_pred = {
                    "class_name": class_name,
                    "vietnamese_name": class_name.replace('_', ' '),
                    "scientific_name": "",
                    "family": "",
                    "confidence": confidence,
                    "summary": {}
                }
            
            enriched_predictions.append(enriched_pred)
        
        return {"predictions": enriched_predictions}
    
    def get_plant_detail(self, class_name: str) -> Dict:
        """
        Get detailed information for selected plant
        
        Returns:
            Complete plant information
        """
        plant_data = self.data_loader.get_plant_by_class(class_name)
        
        if not plant_data:
            return {"error": "Plant not found"}
        
        return {
            "basic_info": {
                "vietnamese_name": plant_data.get("ten", ""),
                "scientific_name": plant_data.get("ten_khoa_hoc", ""),
                "family": plant_data.get("ho", ""),
                "other_names": plant_data.get("ten_khac", "")
            },
            "description": plant_data.get("Mô tả", {}),
            "distribution": plant_data.get("Phân bố", {}),
            "uses": plant_data.get("Công dụng", {}),
            "usage": plant_data.get("Cách dùng", {}),
            "composition": plant_data.get("Thành phần", {}),
            "properties": plant_data.get("Tính vị", {}),
            "parts_used": plant_data.get("Bộ phận dùng", {}),
            "warnings": plant_data.get("luu_y", {})
        }
    
    def _generate_summary(self, plant_data: Dict) -> Dict:
        """Generate concise summary from plant data"""
        summary = {}
        
        # Extract key information
        if "Mô tả" in plant_data and plant_data["Mô tả"]:
            mo_ta = plant_data["Mô tả"]
            if isinstance(mo_ta, dict):
                # Get first description item
                for key, value in mo_ta.items():
                    if value:
                        summary["description"] = str(value)[:200] + "..."
                        break
        
        if "Công dụng" in plant_data and plant_data["Công dụng"]:
            cong_dung = plant_data["Công dụng"]
            if isinstance(cong_dung, dict):
                # Get first use item
                for key, value in cong_dung.items():
                    if value:
                        summary["uses"] = str(value)[:200] + "..."
                        break
        
        if "Cách dùng" in plant_data and plant_data["Cách dùng"]:
            cach_dung = plant_data["Cách dùng"]
            if isinstance(cach_dung, dict):
                for key, value in cach_dung.items():
                    if value:
                        summary["usage"] = str(value)[:150] + "..."
                        break
        
        if "luu_y" in plant_data and plant_data["luu_y"]:
            luu_y = plant_data["luu_y"]
            if isinstance(luu_y, dict):
                for key, value in luu_y.items():
                    if value:
                        summary["warnings"] = str(value)[:150] + "..."
                        break
        
        return summary


def get_flow1_service(
    cv_client: CVAPIClient,
    data_loader: PlantDataLoader
) -> Flow1Service:
    """Factory for Flow 1 service"""
    return Flow1Service(cv_client, data_loader)
