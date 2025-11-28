"""
Intelligent Value Chunker for Vietnamese Text
Sentence-level chunking with semantic preservation
"""
import re
from typing import List, Tuple


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for Vietnamese text
    Vietnamese: ~1.3 tokens per word
    
    Args:
        text: Input Vietnamese text
        
    Returns:
        Estimated token count
    """
    if not text:
        return 0
    words = text.split()
    return int(len(words) * 1.3)


def split_into_sentences(text: str) -> List[str]:
    """
    Split Vietnamese text into sentences
    Handles:
    - Period, exclamation, question marks
    - Abbreviations (Dr., Jr., etc.)
    - Numbers with decimals
    
    Args:
        text: Input Vietnamese text
        
    Returns:
        List of sentences
    """
    if not text:
        return []
    
    # Vietnamese sentence delimiters
    text = re.sub(r'([.!?])\s+', r'\1<SPLIT>', text)
    
    # Don't split on abbreviations
    text = re.sub(r'([A-Z]\.<SPLIT>)', r'\1', text)
    
    # Don't split on numbers
    text = re.sub(r'(\d+\.<SPLIT>\d+)', r'\1', text)
    
    sentences = text.split('<SPLIT>')
    return [s.strip() for s in sentences if s.strip()]


def chunk_long_value(
    key: str,
    value: str,
    max_tokens: int = 250,
    min_tokens: int = 30
) -> List[Tuple[str, str, int]]:
    """
    Intelligent sentence-level chunking
    
    Args:
        key: Field key (Vietnamese)
        value: Field value to chunk
        max_tokens: Maximum tokens per chunk (default: 250)
        min_tokens: Minimum tokens per chunk (default: 30)
        
    Returns:
        List of (key, chunk_text, chunk_id) tuples
    """
    # Check if chunking is needed
    if estimate_tokens(value) <= max_tokens:
        return [(key, value, 0)]
    
    sentences = split_into_sentences(value)
    
    # If only one long sentence, split by comma
    if len(sentences) == 1:
        parts = value.split(',')
        if len(parts) > 1:
            sentences = [p.strip() + ',' for p in parts[:-1]] + [parts[-1].strip()]
    
    chunks = []
    current_chunk = []
    current_tokens = 0
    chunk_id = 0
    
    for sentence in sentences:
        sent_tokens = estimate_tokens(sentence)
        
        # If single sentence is too long, force split by comma
        if sent_tokens > max_tokens:
            if current_chunk:
                chunks.append((
                    key,
                    ' '.join(current_chunk),
                    chunk_id
                ))
                chunk_id += 1
                current_chunk = []
                current_tokens = 0
            
            # Split long sentence by comma
            sub_parts = [p.strip() for p in sentence.split(',') if p.strip()]
            for part in sub_parts:
                part_tokens = estimate_tokens(part)
                if current_tokens + part_tokens <= max_tokens:
                    current_chunk.append(part)
                    current_tokens += part_tokens
                else:
                    if current_chunk:
                        chunks.append((
                            key,
                            ', '.join(current_chunk),
                            chunk_id
                        ))
                        chunk_id += 1
                    current_chunk = [part]
                    current_tokens = part_tokens
        
        # Normal sentence processing
        elif current_tokens + sent_tokens <= max_tokens:
            current_chunk.append(sentence)
            current_tokens += sent_tokens
        else:
            # Save current chunk
            if current_chunk:
                chunks.append((
                    key,
                    ' '.join(current_chunk),
                    chunk_id
                ))
                chunk_id += 1
            
            # Start new chunk
            current_chunk = [sentence]
            current_tokens = sent_tokens
    
    # Save remaining
    if current_chunk:
        chunks.append((
            key,
            ' '.join(current_chunk),
            chunk_id
        ))
    
    return chunks if chunks else [(key, value, 0)]


# Test if run directly
if __name__ == "__main__":
    test_value = """Ở Ấn Độ, Nepal và Philipin, thân rễ sâm cau được dùng làm thuốc lợi tiểu và kích dục chữa bệnh ngoài da, loét dạ dày tá tràng, trĩ, lậu, bạch đới, hen, vàng da, tiêu chảy và nhức đầu. Ở Ấn Độ, người ta còn dùng thân rễ sâm cau để gây sẩy thai dưới dạng thuốc sắc, hoặc thuốc bột uống với đường trong một cốc sữa. Rễ sâm cau là một thành phần trong bài thuốc cổ truyền Ấn Độ gồm 10 dược liệu trị sỏi niệu."""
    
    print(f"Original text: {len(test_value)} chars, ~{estimate_tokens(test_value)} tokens\n")
    
    chunks = chunk_long_value("Làm thuốc", test_value, max_tokens=250)
    
    print(f"Generated {len(chunks)} chunks:\n")
    for key, text, chunk_id in chunks:
        print(f"Chunk {chunk_id}: {estimate_tokens(text)} tokens")
        print(f"{text[:100]}...\n")
