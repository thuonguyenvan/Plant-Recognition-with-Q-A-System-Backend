"""
Flow 2 Service: Image + Text Q&A
LLM-based routing with full plant context
"""
from typing import Dict, Optional
from services.cv_api_client import CVAPIClient
from services.llm_client import MegLLMClient
from services.ograg_engine import OGRAGQueryEngine
from utils.data_loader import PlantDataLoader
import json


class Flow2Service:
    """Service for Flow 2: Image + Text Q&A"""
    
    def __init__(
        self,
        cv_client: CVAPIClient,
        llm_client: MegLLMClient,
        og_rag: OGRAGQueryEngine,
        data_loader: PlantDataLoader
    ):
        self.cv_client = cv_client
        self.llm_client = llm_client
        self.og_rag = og_rag
        self.data_loader = data_loader
    
    def identify_plant(
        self,
        image_path: Optional[str] = None,
        image_url: Optional[str] = None,
        top_k: int = 5
    ) -> Dict:
        """
        Step 1: Identify plant from image only
        
        Args:
            image_path: Path to image file
            image_url: URL to image
            top_k: Number of predictions to return
            
        Returns:
            {
                "predictions": [
                    {
                        "class_name": str,
                        "vietnamese_name": str,
                        "scientific_name": str,
                        "confidence": float
                    },
                    ...
                ]
            }
        """
        # Classify image
        predictions = self.cv_client.classify_image(
            image_path=image_path,
            image_url=image_url
        )
        
        # Enrich predictions with Vietnamese names
        enriched_predictions = []
        for pred in predictions[:top_k]:
            plant_data = self.data_loader.get_plant_by_class(pred['class_name'])
            if plant_data:
                enriched_predictions.append({
                    "class_name": pred['class_name'],
                    "vietnamese_name": plant_data.get('ten', ''),
                    "scientific_name": plant_data.get('ten_khoa_hoc', ''),
                    "confidence": pred['confidence']
                })
            else:
                # Fallback if data not found
                enriched_predictions.append({
                    "class_name": pred['class_name'],
                    "vietnamese_name": pred['class_name'],
                    "scientific_name": "",
                    "confidence": pred['confidence']
                })
        
        return {"predictions": enriched_predictions}
    
    def answer_with_plant(
        self,
        question: str,
        plant_class_name: str,
        use_rag: bool = True
    ) -> Dict:
        """
        Step 2: Answer question about specific plant (after user selection)
        
        Args:
            question: User question
            plant_class_name: Selected plant class name
            use_rag: Whether to search additional plants via RAG
            
        Returns:
            {
                "identified_plant": {...},
                "answer": str,
                "needs_rag": bool,
                "rag_context": [...] (if needs_rag)
            }
        """
        # Load plant data
        plant_data = self.data_loader.get_plant_by_class(plant_class_name)
        
        if not plant_data:
            return {"error": "Plant data not found"}
        
        plant_info = {
            "class_name": plant_class_name,
            "vietnamese_name": plant_data.get("ten", ""),
            "scientific_name": plant_data.get("ten_khoa_hoc", "")
        }
        
        # Build full context
        full_context = self._build_full_context(plant_data)
        
        # LLM routing
        routing_decision = self._llm_routing(question, plant_info['vietnamese_name'])
        needs_rag = routing_decision['needs_rag'] and use_rag
        
        if needs_rag:
            # Query RAG for additional plants
            rag_results = self.og_rag.query(question, top_k=5)
            
            # Filter out current plant
            rag_results = [r for r in rag_results 
                          if r['plant_name'] != plant_info['vietnamese_name']][:3]
            
            # Build combined context
            combined_context = f"""## Cây từ ảnh: {plant_info['vietnamese_name']}

{full_context}

## Thông tin các cây khác liên quan:

"""
            for result in rag_results:
                combined_context += f"\n- **{result['plant_name']}** - {result['key']}: {result['value']}\n"
            
            answer = self.llm_client.answer_question(
                question=question,
                context=combined_context
            )
            
            return {
                "identified_plant": plant_info,
                "needs_rag": True,
                "answer": answer,
                "rag_context": rag_results
            }
        else:
            # Answer using only plant from image
            answer = self.llm_client.answer_question(
                question=question,
                context=full_context
            )
            
            return {
                "identified_plant": plant_info,
                "needs_rag": False,
                "answer": answer
            }
    
    def answer_question(
        self,
        question: str,
        image_path: Optional[str] = None,
        image_url: Optional[str] = None
    ) -> Dict:
        """
        One-step answer (legacy - for backward compatibility)
        
        Returns:
            {
                "identified_plant": {...},
                "needs_rag": bool,
                "answer": str,
                "rag_context": [...] (if needs_rag)
            }
        """
        # Step 1: Identify plant from image
        predictions = self.cv_client.classify_image(
            image_path=image_path,
            image_url=image_url
        )
        
        top_prediction = predictions[0]
        plant_class = top_prediction['class_name']
        
        # Step 2: Answer with top prediction
        result = self.answer_with_plant(question, plant_class, use_rag=True)
        
        # Add confidence to identified_plant
        if "identified_plant" in result:
            result["identified_plant"]["confidence"] = top_prediction['confidence']
        
        return result
    
    def _build_full_context(self, plant_data: Dict) -> str:
        """Build complete plant context string"""
        context_parts = []
        
        # Basic info
        context_parts.append(f"**Tên:** {plant_data.get('ten', '')}")
        context_parts.append(f"**Tên khoa học:** {plant_data.get('ten_khoa_hoc', '')}")
        context_parts.append(f"**Họ:** {plant_data.get('ho', '')}")
        
        # Sections
        sections = [
            ("Mô tả", "Mô tả"),
            ("Phân bố", "Phân bố"),
            ("Công dụng", "Công dụng"),
            ("Cách dùng", "Cách dùng"),
            ("Thành phần", "Thành phần"),
            ("Tính vị", "Tính vị"),
            ("Bộ phận dùng", "Bộ phận dùng"),
            ("luu_y", "Lưu ý")
        ]
        
        for key, title in sections:
            if key in plant_data and plant_data[key]:
                section_data = plant_data[key]
                if isinstance(section_data, dict):
                    context_parts.append(f"\n### {title}")
                    for sub_key, sub_value in section_data.items():
                        if sub_value:
                            context_parts.append(f"- **{sub_key}**: {sub_value}")
        
        return "\n".join(context_parts)
    
    def _llm_routing(self, question: str, plant_name: str) -> Dict:
        """
        LLM decides if RAG is needed
        
        Returns:
            {"needs_rag": bool, "reasoning": str}
        """
        routing_prompt = f"""Bạn là hệ thống phân tích câu hỏi về dược liệu.

Câu hỏi: "{question}"
Cây được nhận diện: "{plant_name}"

Hãy quyết định xem câu hỏi này:
- CẦN so sánh/tìm kiếm thông tin từ các cây KHÁC → trả về {{"needs_rag": true}}
- CHỈ cần thông tin về cây "{plant_name}" → trả về {{"needs_rag": false}}

CHỈ trả về JSON, không giải thích."""

        try:
            response = self.llm_client.chat(
                messages=[{"role": "user", "content": routing_prompt}],
                temperature=0.1,
                max_tokens=100
            )
            
            # Parse JSON response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                # Fallback: assume no RAG needed
                return {"needs_rag": False}
        except:
            # On error, fallback to no RAG
            return {"needs_rag": False}


def get_flow2_service(
    cv_client: CVAPIClient,
    llm_client: MegLLMClient,
    og_rag: OGRAGQueryEngine,
    data_loader: PlantDataLoader
) -> Flow2Service:
    """Factory for Flow 2"""
    return Flow2Service(cv_client, llm_client, og_rag, data_loader)
