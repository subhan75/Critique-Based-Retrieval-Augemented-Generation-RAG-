"""
FAISS index building and management module.
"""
from typing import List, Tuple, Optional
import numpy as np
import faiss
from pathlib import Path

from ..utils.logger import get_logger

logger = get_logger(__name__)


def build_faiss_index(
    embeddings: np.ndarray,
    index_type: str = "flat"
) -> faiss.Index:
    """
    Build FAISS index from embeddings.
    
    Args:
        embeddings: Numpy array of embeddings (shape: [num_docs, embedding_dim])
        index_type: Type of FAISS index ("flat" for exact search)
        
    Returns:
        FAISS index
    """
    logger.info(f"Building FAISS index (type={index_type})...")
    logger.info(f"Embeddings shape: {embeddings.shape}")
    
    dimension = embeddings.shape[1]
    num_vectors = embeddings.shape[0]
    
    # Ensure embeddings are float32
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype('float32')
    
    # Create index based on type
    if index_type == "flat":
        # IndexFlatL2: Exact search using L2 distance
        index = faiss.IndexFlatL2(dimension)
    else:
        raise ValueError(f"Unsupported index type: {index_type}")
    
    # Add vectors to index
    index.add(embeddings)
    
    logger.info(f"FAISS index built: {index.ntotal} vectors, dimension={dimension}")
    
    return index


def save_index(
    index: faiss.Index,
    output_path: str
) -> None:
    """
    Save FAISS index to disk.
    
    Args:
        index: FAISS index
        output_path: Path to save index file
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    faiss.write_index(index, str(output_file))
    logger.info(f"FAISS index saved to {output_path}")


def load_index(input_path: str) -> faiss.Index:
    """
    Load FAISS index from disk.
    
    Args:
        input_path: Path to index file
        
    Returns:
        Loaded FAISS index
    """
    index = faiss.read_index(input_path)
    logger.info(f"FAISS index loaded from {input_path}")
    logger.info(f"Index contains {index.ntotal} vectors")
    
    return index


def search_index(
    index: faiss.Index,
    query_embedding: np.ndarray,
    k: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Search FAISS index for nearest neighbors.
    
    Args:
        index: FAISS index
        query_embedding: Query embedding (shape: [1, embedding_dim] or [embedding_dim])
        k: Number of nearest neighbors to retrieve
        
    Returns:
        Tuple of (distances, indices)
        - distances: Array of L2 distances (shape: [1, k])
        - indices: Array of document indices (shape: [1, k])
    """
    # Ensure query is 2D
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)
    
    # Ensure float32
    if query_embedding.dtype != np.float32:
        query_embedding = query_embedding.astype('float32')
    
    # Search
    distances, indices = index.search(query_embedding, k)
    
    return distances, indices


def test_retrieval(
    index: faiss.Index,
    documents: List[str],
    query_embedding: np.ndarray,
    k: int = 5
) -> List[Tuple[int, float, str]]:
    """
    Test retrieval and return results with documents.
    
    Args:
        index: FAISS index
        documents: List of document texts
        query_embedding: Query embedding
        k: Number of results
        
    Returns:
        List of tuples (doc_index, distance, doc_text)
    """
    distances, indices = search_index(index, query_embedding, k)
    
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        if idx < len(documents):
            results.append((int(idx), float(dist), documents[idx]))
    
    return results


def get_index_stats(index: faiss.Index) -> dict:
    """
    Get statistics about FAISS index.
    
    Args:
        index: FAISS index
        
    Returns:
        Dictionary with index statistics
    """
    stats = {
        'num_vectors': index.ntotal,
        'dimension': index.d,
        'is_trained': index.is_trained,
    }
    
    logger.info(f"Index stats: {stats}")
    return stats


def build_and_save_index(
    embeddings: np.ndarray,
    output_path: str,
    index_type: str = "flat"
) -> faiss.Index:
    """
    Build and save FAISS index in one step.
    
    Args:
        embeddings: Numpy array of embeddings
        output_path: Path to save index
        index_type: Type of FAISS index
        
    Returns:
        Built FAISS index
    """
    # Build index
    index = build_faiss_index(embeddings, index_type)
    
    # Save index
    save_index(index, output_path)
    
    return index
