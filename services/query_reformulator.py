"""
Smart Query Reformulation Service
Intelligently reformulates user queries based on conversation context using LLM
"""
from typing import List, Dict, Optional, Any
import json
from services.llm_client import MegLLMClient


# Master reformulation system prompt
REFORMULATION_SYSTEM_PROMPT = """You are a query reformulation expert for a medicinal plant Q&A system.

Your task: Analyze user queries and conversation context to create optimal RAG search queries.

INPUTS:
- current_query: User's current question
- conversation_history: Recent conversation (max 6 turns)
- selected_plant: Plant user selected from image classification (if any)

OUTPUT FORMAT (JSON):
{
  "intent": "specific_plant | generic | comparison | plant_switch | chitchat | generic_excluding",
  "target_plants": ["plant1", "plant2"],
  "excluded_plants": ["plant3"],
  "reformulated_query": "explicit, complete query" OR ["query1", "query2"] for comparison,
  "needs_rag": true/false,
  "reasoning": "brief explanation of decision"
}

RULES:

1. PLANT CONTEXT:
   - If query is follow-up about selected_plant ("công dụng", "cách dùng"), add plant name
   - If new plant mentioned, that becomes new target
   - Detect pronouns ("nó", "cây này") and resolve to specific plant

2. GENERIC QUERIES:
   - "Cây nào...", "Loại cây...", "Dược liệu gì..." → Keep generic, target_plants = []
   - Don't force plant context on exploratory questions

3. COMPARISON:
   - "So sánh A và B" → Create separate queries for each plant
   - "Khác nhau gì" → Same as comparison

4. EXCLUSION:
   - "Ngoài X ra", "trừ X" → Mark X in excluded_plants, make query generic

5. CHITCHAT:
   - Greetings, thanks, acknowledgments → needs_rag = false
   - "Ok", "được", "cảm ơn" → chitchat intent

6. QUERY QUALITY:
   - Always make reformulated query complete and explicit
   - Remove pronouns, add context
   - Keep natural Vietnamese phrasing

7. AMBIGUITY:
   - If unclear, prefer specific_plant with selected_plant over generic
   - But if user explicitly broadens ("cây khác", "cây nào"), go generic

EXAMPLES:

Input: {"current_query": "công dụng", "selected_plant": "Đậu bắp", "history": ["User chọn Đậu bắp"]}
Output: {"intent": "specific_plant", "target_plants": ["Đậu bắp"], "reformulated_query": "công dụng của Đậu bắp", "needs_rag": true, "reasoning": "Follow-up about selected plant"}

Input: {"current_query": "Còn Rau má thì sao?", "selected_plant": "Đậu bắp", "history": ["công dụng của Đậu bắp"]}
Output: {"intent": "plant_switch", "target_plants": ["Rau má"], "reformulated_query": "công dụng của Rau má", "needs_rag": true, "reasoning": "User switched to new plant, inherit topic"}

Input: {"current_query": "Cây nào chữa ho?", "selected_plant": "Đậu bắp"}
Output: {"intent": "generic", "target_plants": [], "reformulated_query": "Cây nào chữa ho", "needs_rag": true, "reasoning": "Generic search query"}

Input: {"current_query": "Cảm ơn bạn!", "selected_plant": "Đậu bắp"}
Output: {"intent": "chitchat", "needs_rag": false, "reasoning": "Acknowledgment, no information needed"}

Input: {"current_query": "So sánh Đậu bắp và Rau má", "selected_plant": "Đậu bắp"}
Output: {"intent": "comparison", "target_plants": ["Đậu bắp", "Rau má"], "reformulated_query": ["công dụng của Đậu bắp", "công dụng của Rau má"], "needs_rag": true, "reasoning": "Comparison query"}
"""


class SmartQueryReformulator:
    """LLM-based intelligent query reformulation"""
    
    def __init__(self, llm_client: MegLLMClient):
        """
        Initialize reformulator
        
        Args:
            llm_client: LLM client for reformulation
        """
        self.llm_client = llm_client
    
    def reformulate(
        self,
        current_query: str,
        conversation_history: List[Dict[str, str]] = None,
        selected_plant: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Reformulate query based on context
        
        Args:
            current_query: User's current question
            conversation_history: Previous conversation messages
            selected_plant: Currently selected plant (from modal)
            
        Returns:
            {
                "intent": str,
                "target_plants": List[str],
                "excluded_plants": List[str],
                "reformulated_query": str or List[str],
                "needs_rag": bool,
                "reasoning": str
            }
        """
        # Build reformulation request
        user_content = self._build_reformulation_request(
            current_query,
            conversation_history or [],
            selected_plant
        )
        
        # Call LLM with reformulation prompt
        try:
            response = self.llm_client.chat(
                messages=[
                    {"role": "system", "content": REFORMULATION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.1,  # Low temperature for consistency
                max_tokens=500
            )
            
            # Parse JSON response
            result = self._parse_reformulation_response(response)
            
            # Validate and set defaults
            result = self._validate_reformulation(result, current_query, selected_plant)
            
            return result
            
        except Exception as e:
            # Fallback to simple heuristic
            print(f"Reformulation failed: {e}, using fallback")
            return self._fallback_reformulation(current_query, selected_plant)
    
    def _build_reformulation_request(
        self,
        query: str,
        history: List[Dict[str, str]],
        selected_plant: Optional[str]
    ) -> str:
        """Build the reformulation request context"""
        
        # Extract plant-relevant history (last 6 turns max)
        plant_context = self._extract_plant_context(history[-6:] if history else [])
        
        # Build request
        request = {
            "current_query": query,
            "selected_plant": selected_plant or None,
            "conversation_summary": plant_context
        }
        
        return f"""Reformulate this query:

Current query: "{query}"
Selected plant: {selected_plant or "None"}
Recent conversation: {plant_context}

Analyze and output JSON reformulation."""
    
    def _extract_plant_context(self, history: List[Dict[str, str]]) -> str:
        """Extract plant-related context from history"""
        if not history:
            return "No previous conversation"
        
        # Summarize last few messages
        context_parts = []
        for msg in history:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            
            # Keep it short
            if len(content) > 100:
                content = content[:100] + "..."
            
            context_parts.append(f"{role}: {content}")
        
        return " | ".join(context_parts[-4:])  # Last 4 messages
    
    def _parse_reformulation_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM JSON response"""
        # Try to extract JSON from response
        try:
            # Remove markdown code blocks if present
            cleaned = response.strip()
            if cleaned.startswith("```"):
                # Extract content between ```json and ```
                lines = cleaned.split("\n")
                json_lines = []
                in_json = False
                for line in lines:
                    if line.startswith("```"):
                        in_json = not in_json
                        continue
                    if in_json:
                        json_lines.append(line)
                cleaned = "\n".join(json_lines)
            
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON response: {e}\nResponse: {response}")
    
    def _validate_reformulation(
        self,
        result: Dict[str, Any],
        original_query: str,
        selected_plant: Optional[str]
    ) -> Dict[str, Any]:
        """Validate and set defaults for reformulation result"""
        
        # Set defaults for missing fields
        result.setdefault("intent", "specific_plant" if selected_plant else "generic")
        result.setdefault("target_plants", [])
        result.setdefault("excluded_plants", [])
        result.setdefault("reformulated_query", original_query)
        result.setdefault("needs_rag", True)
        result.setdefault("reasoning", "Default reformulation")
        
        # Ensure target_plants is list
        if not isinstance(result["target_plants"], list):
            result["target_plants"] = []
        
        # Ensure excluded_plants is list
        if not isinstance(result["excluded_plants"], list):
            result["excluded_plants"] = []
        
        return result
    
    def _fallback_reformulation(
        self,
        query: str,
        selected_plant: Optional[str]
    ) -> Dict[str, Any]:
        """Fallback reformulation using simple heuristics"""
        
        # Check if query is very short (likely follow-up)
        is_short = len(query.split()) <= 3
        
        # Check for chitchat patterns
        chitchat_patterns = ["cảm ơn", "thanks", "ok", "được", "tốt", "hay", "xin chào"]
        is_chitchat = any(pattern in query.lower() for pattern in chitchat_patterns)
        
        if is_chitchat:
            return {
                "intent": "chitchat",
                "target_plants": [],
                "excluded_plants": [],
                "reformulated_query": query,
                "needs_rag": False,
                "reasoning": "Chitchat detected (fallback)"
            }
        
        # If short query and have selected plant, add context
        if is_short and selected_plant:
            reformulated = f"{query} của {selected_plant}"
            return {
                "intent": "specific_plant",
                "target_plants": [selected_plant],
                "excluded_plants": [],
                "reformulated_query": reformulated,
                "needs_rag": True,
                "reasoning": "Added plant context to short query (fallback)"
            }
        
        # Otherwise keep as-is
        return {
            "intent": "generic",
            "target_plants": [],
            "excluded_plants": [],
            "reformulated_query": query,
            "needs_rag": True,
            "reasoning": "Kept original query (fallback)"
        }


def get_query_reformulator(llm_client: MegLLMClient) -> SmartQueryReformulator:
    """Factory function for query reformulator"""
    return SmartQueryReformulator(llm_client)
