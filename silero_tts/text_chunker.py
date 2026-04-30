import re
from typing import List

def split_text(
    text: str,
    max_chars: int = 140,
    sentence_ends: str = ".!?"
) -> List[str]:
    """
    Split long text into chunks ≤ max_chars.
    Priority: paragraphs → lines → sentences → words.
    Merges short chunks for efficiency.
    """
    raw_chunks: List[str] = []
    
    # 1. Split by paragraphs
    paragraphs = re.split(r'\n\s*\n', text)
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        if len(para) <= max_chars:
            raw_chunks.append(para)
            continue
        
        # 2. Split by lines
        lines = para.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if len(line) <= max_chars:
                raw_chunks.append(line)
                continue
            
            # 3. Split by sentences (keeping punctuation)
            # Match sentence + following punctuation
            sentence_pattern = f'([^{re.escape(sentence_ends)}]+[{re.escape(sentence_ends)}]|[^{re.escape(sentence_ends)}]+$)'
            sentences = re.findall(sentence_pattern, line)
            
            if not sentences:
                sentences = [line]
            
            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue
                if len(sent) <= max_chars:
                    raw_chunks.append(sent)
                else:
                    # 4. Split by words
                    words = sent.split()
                    chunk = ""
                    for word in words:
                        if len(chunk) + len(word) + 1 <= max_chars:
                            chunk += " " + word if chunk else word
                        else:
                            if chunk:
                                raw_chunks.append(chunk.strip())
                            chunk = word
                    if chunk:
                        raw_chunks.append(chunk.strip())
    
    # Merge short consecutive chunks
    merged: List[str] = []
    buf = ""
    for chunk in raw_chunks:
        if len(buf) + len(chunk) + 1 <= max_chars:
            buf += " " + chunk if buf else chunk
        else:
            if buf:
                merged.append(buf.strip())
            buf = chunk
    if buf:
        merged.append(buf.strip())
    
    return [c for c in merged if c]
