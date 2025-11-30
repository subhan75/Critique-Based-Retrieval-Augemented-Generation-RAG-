"""
Document chunking module with sliding window approach.
"""
from typing import List, Dict, Any, Tuple
import re

from ..utils.logger import get_logger

logger = get_logger(__name__)


def count_tokens(text: str) -> int:
    """
    Approximate token count (simple word-based estimation).
    
    For more accurate counting, could use tiktoken library.
    This approximation: 1 token ≈ 0.75 words
    
    Args:
        text: Input text
        
    Returns:
        Estimated token count
    """
    words = text.split()
    return int(len(words) * 1.33)  # Rough approximation


def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 50,
    min_chunk_size: int = 100
) -> List[str]:
    """
    Split text into overlapping chunks using sliding window.
    
    Args:
        text: Input text to chunk
        chunk_size: Target chunk size in tokens
        overlap: Number of overlapping tokens between chunks
        min_chunk_size: Minimum chunk size (discard smaller chunks)
        
    Returns:
        List of text chunks
    """
    # Split into sentences for better chunk boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        sentence_tokens = count_tokens(sentence)
        
        # If adding this sentence exceeds chunk_size, save current chunk
        if current_size + sentence_tokens > chunk_size and current_chunk:
            chunk_str = ' '.join(current_chunk)
            if count_tokens(chunk_str) >= min_chunk_size:
                chunks.append(chunk_str)
            
            # Start new chunk with overlap
            # Keep last few sentences for overlap
            overlap_sentences = []
            overlap_size = 0
            for sent in reversed(current_chunk):
                sent_tokens = count_tokens(sent)
                if overlap_size + sent_tokens <= overlap:
                    overlap_sentences.insert(0, sent)
                    overlap_size += sent_tokens
                else:
                    break
            
            current_chunk = overlap_sentences
            current_size = overlap_size
        
        current_chunk.append(sentence)
        current_size += sentence_tokens
    
    # Add final chunk
    if current_chunk:
        chunk_str = ' '.join(current_chunk)
        if count_tokens(chunk_str) >= min_chunk_size:
            chunks.append(chunk_str)
    
    return chunks


def chunk_documents(
    documents: List[str],
    chunk_size: int = 512,
    overlap: int = 50,
    add_metadata: bool = True
) -> List[Dict[str, Any]]:
    """
    Chunk multiple documents with metadata.
    
    Args:
        documents: List of document texts
        chunk_size: Target chunk size in tokens
        overlap: Overlap size in tokens
        add_metadata: Whether to add metadata to chunks
        
    Returns:
        List of chunk dictionaries with text and metadata
    """
    logger.info(f"Chunking {len(documents)} documents (size={chunk_size}, overlap={overlap})...")
    
    all_chunks = []
    chunk_id = 0
    
    for doc_id, document in enumerate(documents):
        # Skip empty documents
        if not document or not document.strip():
            continue
        
        # Chunk the document
        doc_chunks = chunk_text(document, chunk_size, overlap)
        
        # Add metadata if requested
        for position, chunk_content in enumerate(doc_chunks):
            if add_metadata:
                chunk_dict = {
                    'chunk_id': chunk_id,
                    'text': chunk_content,
                    'metadata': {
                        'source_doc_id': doc_id,
                        'position': position,
                        'total_chunks': len(doc_chunks),
                        'token_count': count_tokens(chunk_content)
                    }
                }
            else:
                chunk_dict = {
                    'chunk_id': chunk_id,
                    'text': chunk_content
                }
            
            all_chunks.append(chunk_dict)
            chunk_id += 1
    
    logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
    logger.info(f"Average chunks per document: {len(all_chunks) / max(len(documents), 1):.2f}")
    
    return all_chunks


def get_chunk_texts(chunks: List[Dict[str, Any]]) -> List[str]:
    """
    Extract just the text from chunk dictionaries.
    
    Args:
        chunks: List of chunk dictionaries
        
    Returns:
        List of chunk texts
    """
    return [chunk['text'] for chunk in chunks]


def get_chunk_statistics(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute statistics about chunks.
    
    Args:
        chunks: List of chunk dictionaries
        
    Returns:
        Dictionary with statistics
    """
    if not chunks:
        return {
            'total_chunks': 0,
            'avg_token_count': 0,
            'min_token_count': 0,
            'max_token_count': 0
        }
    
    token_counts = [chunk['metadata']['token_count'] for chunk in chunks if 'metadata' in chunk]
    
    if not token_counts:
        token_counts = [count_tokens(chunk['text']) for chunk in chunks]
    
    stats = {
        'total_chunks': len(chunks),
        'avg_token_count': sum(token_counts) / len(token_counts),
        'min_token_count': min(token_counts),
        'max_token_count': max(token_counts),
        'total_tokens': sum(token_counts)
    }
    
    logger.info(f"Chunk statistics: {stats}")
    return stats
