"""
Document processing utilities for RAG pipeline
"""
import re
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """Clean and normalize text."""
    logger.debug(f"Cleaning text of length {len(text)}")
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\-\:\;\(\)]', '', text)
    cleaned = text.strip()
    logger.debug(f"Cleaned text length: {len(cleaned)}")
    return cleaned

def chunk_by_sentences(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Chunk text by sentences to preserve semantic meaning.
    """
    logger.debug(f"Chunking text of length {len(text)} with chunk_size={chunk_size}, overlap={overlap}")
    # Split by sentence endings
    sentences = re.split(r'(?<=[.!?])\s+', text)
    logger.debug(f"Split into {len(sentences)} sentences")
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence)
        
        if current_length + sentence_length > chunk_size and current_chunk:
            # Save current chunk
            chunks.append(' '.join(current_chunk))
            
            # Start new chunk with overlap
            overlap_sentences = []
            overlap_length = 0
            for s in reversed(current_chunk):
                if overlap_length + len(s) <= overlap:
                    overlap_sentences.insert(0, s)
                    overlap_length += len(s)
                else:
                    break
            
            current_chunk = overlap_sentences
            current_length = overlap_length
        
        current_chunk.append(sentence)
        current_length += sentence_length
    
    # Add remaining chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    logger.info(f"Created {len(chunks)} chunks from text")
    return chunks

def create_chunk_with_metadata(chunk: str, source_file: str, chunk_index: int, page_number: int = None) -> Dict:
    """
    Create a chunk dictionary with metadata including page numbers.
    """
    metadata = {
        "source": source_file,
        "chunk_index": chunk_index,
        "length": len(chunk)
    }
    
    if page_number is not None:
        metadata["page_number"] = page_number
    
    return {
        "text": chunk,
        "metadata": metadata
    }
