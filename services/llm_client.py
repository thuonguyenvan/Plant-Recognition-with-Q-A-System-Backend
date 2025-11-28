"""
MegLLM API Client
OpenAI-compatible API using openai SDK
"""
import re
from typing import List, Dict, Optional
from openai import OpenAI


def strip_html_tags(text: str) -> str:
    """
    Remove HTML tags from text while preserving line breaks
    
    Converts <br> to newlines, then removes other HTML tags
    """
    if not text:
        return ""
    
    # Replace <br> and <br/> with newlines (preserve line breaks)
    clean = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
    
    # Remove other HTML tags
    clean = re.sub(r'<[^>]+>', '', clean)
    
    # Clean up excessive whitespace (but preserve newlines)
    lines = [line.strip() for line in clean.split('\n')]
    clean = '\n'.join(line for line in lines if line)
    
    return clean.strip()


class MegLLMClient:
    """Client for MegLLM API (OpenAI-compatible)"""
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://ai.megallm.io/v1",
        model: str = "qwen/qwen3-next-80b-a3b-instruct",
        timeout: int = 60
    ):
        """
        Initialize MegLLM client
        
        Args:
            api_key: MegLLM API key
            base_url: API base URL
            model: Model name
            timeout: Request timeout
        """
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout
        )
        self.model = model
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> str:
        """
        Chat completion
        
        Args:
            messages: List of {role: "system"|"user"|"assistant", content: str}
            temperature: Sampling temperature
            max_tokens: Max response tokens
            
        Returns:
            Response text (HTML tags removed)
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        content = response.choices[0].message.content
        
        # Strip HTML tags that LLM sometimes generates
        return strip_html_tags(content)
    
    def answer_question(
        self,
        question: str,
        context: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Answer question with context
        
        Args:
            question: User question
            context: Context information
            system_prompt: Optional system prompt
            
        Returns:
            Answer text
        """
        if not system_prompt:
            system_prompt = """Bạn là trợ lý AI chuyên về dược liệu Việt Nam.
Nhiệm vụ của bạn là trả lời câu hỏi dựa trên thông tin được cung cấp.
Trả lời chính xác, ngắn gọn, dễ hiểu bằng tiếng Việt."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"""Dựa vào thông tin sau:

{context}

Hãy trả lời câu hỏi: {question}"""}
        ]
        
        return self.chat(messages)
    
    def answer_with_history(
        self,
        question: str,
        context: str,
        conversation_history: List[Dict[str, str]] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Answer question with context and conversation history
        
        Args:
            question: Current question
            context: RAG context
            conversation_history: Previous messages [{role, content}, ...]
            system_prompt: Optional system prompt
            
        Returns:
            Answer text
        """
        if not system_prompt:
            system_prompt = """Bạn là trợ lý AI chuyên về dược liệu Việt Nam.

NHIỆM VỤ:
- Trả lời câu hỏi CHÍNH XÁC dựa HOÀN TOÀN trên thông tin được cung cấp
- Nhớ và sử dụng thông tin từ cuộc trò chuyện trước đó

QUY TẮC QUAN TRỌNG:
1. CHỈ sử dụng thông tin có trong "Thông tin dược liệu liên quan" để trả lời
2. KHÔNG đưa ra thông tin bạn không chắc chắn hoặc không có trong context
3. NẾU thông tin không có trong context, hãy thẳng thắn nói: "Xin lỗi, tôi không tìm thấy thông tin về [tên cây/câu hỏi] trong cơ sở dữ liệu."
4. KHÔNG bịa đặt, suy luận, hoặc đưa ra thông tin từ kiến thức chung
5. NẾU context trống hoặc không liên quan đến câu hỏi, hãy nói: "Tôi không có thông tin về câu hỏi này."

PHONG CÁCH:
- Thân thiện, lịch sự
- Chính xác, có căn cứ từ context
- Ngắn gọn, dễ hiểu bằng tiếng Việt
- Thẳng thắn thừa nhận khi không biết"""
        
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history
        if conversation_history:
            messages.extend(conversation_history)
        
        # Add current question with context
        user_content = f"""Thông tin dược liệu liên quan:

{context}

Câu hỏi hiện tại: {question}

LƯU Ý: Chỉ trả lời dựa trên "Thông tin dược liệu liên quan" ở trên. Nếu thông tin không đủ hoặc không liên quan, hãy thành thật nói không biết."""
        
        messages.append({"role": "user", "content": user_content})
        
        return self.chat(messages, temperature=0.3)  # Lower temperature for less hallucination
    
    def route_query(self, question: str, conversation_history: List[Dict[str, str]] = None) -> Dict[str, str]:
        """
        Determine if query needs RAG search or can be answered directly
        
        Args:
            question: User question
            conversation_history: Previous messages
            
        Returns:
            {
                "route": "direct" | "rag",
                "reason": str
            }
        """
        system_prompt = """Bạn là chuyên gia phân loại câu hỏi về dược liệu Việt Nam.

NHIỆM VỤ: Xác định câu hỏi cần tra cứu database (RAG) hay có thể trả lời trực tiếp.

CẦN RAG khi:
- Hỏi về cây cụ thể (tên, công dụng, cách dùng)
- Tìm cây chữa bệnh cụ thể
- So sánh nhiều loại cây
- Câu hỏi chi tiết về dược liệu

TRẢ LỜI TRỰC TIẾP khi:
- Chào hỏi, giới thiệu
- Hỏi về khả năng/chức năng của AI
- Câu hỏi general (không cần tra cứu)
- Xác nhận thông tin từ lịch sử chat

OUTPUT: Chỉ trả về JSON format:
{"route": "direct", "reason": "..."} HOẶC {"route": "rag", "reason": "..."}"""

        messages = [{"role": "system", "content": system_prompt}]
        
        # Add history for context
        if conversation_history:
            messages.extend(conversation_history[-4:])  # Last 4 messages
        
        messages.append({
            "role": "user",
            "content": f"Phân loại câu hỏi: {question}"
        })
        
        try:
            response = self.chat(messages, temperature=0.1, max_tokens=100)
            # Parse JSON
            import json
            result = json.loads(response)
            return result
        except:
            # Default to RAG on error
            return {"route": "rag", "reason": "Unable to classify, using RAG for safety"}


# Singleton
_megllm_client = None

def get_megllm_client() -> MegLLMClient:
    """Get cached MegLLM client"""
    global _megllm_client
    if _megllm_client is None:
        from config import get_settings
        settings = get_settings()
        _megllm_client = MegLLMClient(api_key=settings.megllm_api_key)
    return _megllm_client

