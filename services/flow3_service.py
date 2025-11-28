"""
Flow 3 Service: Pure RAG (Text-Only Q&A) with Query Reformulation
"""
from typing import Dict, List
from services.llm_client import MegLLMClient
from services.ograg_engine import OGRAGQueryEngine
from services.query_reformulator import SmartQueryReformulator


class Flow3Service:
    """Service for Flow 3: Text-only RAG with Query Reformulation"""
    
    def __init__(
        self,
        llm_client: MegLLMClient,
        og_rag: OGRAGQueryEngine,
        reformulator: SmartQueryReformulator
    ):
        self.llm_client = llm_client
        self.og_rag = og_rag
        self.reformulator = reformulator
    
    
    def answer_question(
        self,
        question: str,
        top_k: int = 10,
        conversation_history: List[Dict] = None,
        selected_plant: str = None  # NEW: Track selected plant from frontend
    ) -> Dict:
        """
        Answer question using RAG with conversation context and query reformulation
        
        Args:
            question: User question
            top_k: Number of RAG results
            conversation_history: Previous conversation messages
            selected_plant: Plant user selected from modal (if any)
            
        Returns:
            {
                "question": str,
                "answer": str,
                "sources": [...],
                "used_rag": bool,
                "reformulation": {...}  # Reformulation metadata
            }
        """
        # Step 1: Reformulate query using LLM
        reformulation = self.reformulator.reformulate(
            current_query=question,
            conversation_history=conversation_history,
            selected_plant=selected_plant
        )
        
        # DEBUG: Print reformulation
        print(f"\n🔍 REFORMULATION DEBUG:")
        print(f"  Original: {question}")
        print(f"  Reformulated: {reformulation.get('reformulated_query')}")
        print(f"  Intent: {reformulation.get('intent')}")
        print(f"  Target plants: {reformulation.get('target_plants')}")
        print(f"  Reasoning: {reformulation.get('reasoning')}\n")
        
        # Step 2: Handle based on intent
        if reformulation["intent"] == "chitchat" or not reformulation["needs_rag"]:
            # Direct reply without RAG
            return self._handle_chitchat(question, conversation_history, reformulation)
        
        if reformulation["intent"] == "comparison":
            # Handle comparison queries (multiple RAG calls)
            return self._handle_comparison(reformulation, conversation_history, top_k)
        
        # Step 3: Use reformulated query for RAG
        reformulated_query = reformulation["reformulated_query"]
        
        # Determine plant filter
        plant_filter = None
        if reformulation["target_plants"]:
            # If specific plants mentioned, filter to first one
            # (comparison handled separately above)
            plant_filter = reformulation["target_plants"][0]
        
        print(f"🔎 RAG QUERY:")
        print(f"  Query text: {reformulated_query}")
        print(f"  Plant filter: {plant_filter}")
        print(f"  Top-K: {top_k}\n")
        
        # Query RAG
        rag_results = self.og_rag.query(
            query_text=reformulated_query,
            top_k=top_k,
            plant_filter=plant_filter
        )
        
        # DEBUG: Print RAG results
        print(f"📊 RAG RESULTS: {len(rag_results)} documents")
        for i, result in enumerate(rag_results[:5]):  # Show top 5
            print(f"  {i+1}. {result.get('plant_name')} (similarity: {result.get('similarity', 0):.3f})")
            print(f"     {result.get('key')}: {result.get('value')[:100]}...")
        print()
        
        # Filter out excluded plants if any
        if reformulation.get("excluded_plants"):
            rag_results = [
                r for r in rag_results
                if r['plant_name'] not in reformulation["excluded_plants"]
            ]
        
        if not rag_results:
            context = "Không tìm thấy thông tin dược liệu trực tiếp liên quan."
        else:
            context = self.og_rag.build_rag_context(rag_results, max_context_length=2000)
        
        # Generate answer with history
        answer = self.llm_client.answer_with_history(
            question=question,  # Keep original question for natural flow
            context=context,
            conversation_history=conversation_history
        )
        
        # Extract sources
        sources = self._extract_sources(rag_results) if rag_results else []
        
        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "used_rag": True,
            "reformulation": {
                "original_query": question,
                "reformulated_query": reformulated_query,
                "intent": reformulation["intent"],
                "reasoning": reformulation.get("reasoning", "")
            }
        }
    
    def _handle_chitchat(
        self,
        question: str,
        conversation_history: List[Dict],
        reformulation: Dict
    ) -> Dict:
        """Handle chitchat without RAG"""
        system_prompt = """Bạn là trợ lý AI thân thiện về dược liệu Việt Nam.
Trả lời câu hỏi một cách tự nhiên, lịch sự."""
        
        messages = [{"role": "system", "content": system_prompt}]
        if conversation_history:
            messages.extend(conversation_history[-4:])  # Last 4 turns
        messages.append({"role": "user", "content": question})
        
        answer = self.llm_client.chat(messages, temperature=0.8)
        
        return {
            "question": question,
            "answer": answer,
            "sources": [],
            "used_rag": False,
            "reformulation": {
                "original_query": question,
                "intent": reformulation["intent"],
                "reasoning": reformulation.get("reasoning", "")
            }
        }
    
    def _handle_comparison(
        self,
        reformulation: Dict,
        conversation_history: List[Dict],
        top_k: int
    ) -> Dict:
        """Handle comparison queries by querying multiple plants"""
        target_plants = reformulation["target_plants"]
        queries = reformulation["reformulated_query"]
        
        # If queries is string, split by plants
        if isinstance(queries, str):
            queries = [f"{queries} {plant}" for plant in target_plants]
        
        # Query each plant separately
        all_results = []
        for plant, query in zip(target_plants, queries):
            results = self.og_rag.query(
                query_text=query,
                top_k=top_k // len(target_plants),  # Split top_k
                plant_filter=plant
            )
            all_results.extend(results)
        
        # Build context
        if not all_results:
            context = "Không tìm thấy thông tin về các cây để so sánh."
        else:
            context = self.og_rag.build_rag_context(all_results, max_context_length=3000)
        
        # Generate comparison answer
        comparison_prompt = f"""So sánh các cây sau: {', '.join(target_plants)}

Thông tin:
{context}

Hãy so sánh các điểm giống và khác nhau một cách rõ ràng."""
        
        messages = [{"role": "system", "content": "Bạn là chuyên gia dược liệu."}]
        if conversation_history:
            messages.extend(conversation_history[-2:])
        messages.append({"role": "user", "content": comparison_prompt})
        
        answer = self.llm_client.chat(messages, temperature=0.3)
        
        sources = self._extract_sources(all_results) if all_results else []
        
        return {
            "question": reformulation["reformulated_query"],
            "answer": answer,
            "sources": sources,
            "used_rag": True,
            "reformulation": {
                "original_query": reformulation.get("original_query", ""),
                "intent": "comparison",
                "target_plants": target_plants,
                "reasoning": reformulation.get("reasoning", "")
            }
        }
    
    def _extract_sources(self, rag_results: List[Dict]) -> List[Dict]:
        """Extract unique plant sources"""
        seen_plants = set()
        sources = []
        
        for result in rag_results:
            plant_name = result['plant_name']
            if plant_name not in seen_plants:
                seen_plants.add(plant_name)
                sources.append({
                    "plant_name": plant_name,
                    "relevance": result['similarity']
                })
        
        return sources[:5]  # Top 5 sources


def get_flow3_service(
    llm_client: MegLLMClient,
    og_rag: OGRAGQueryEngine,
    reformulator: SmartQueryReformulator
) -> Flow3Service:
    """Factory for Flow 3"""
    return Flow3Service(llm_client, og_rag, reformulator)
