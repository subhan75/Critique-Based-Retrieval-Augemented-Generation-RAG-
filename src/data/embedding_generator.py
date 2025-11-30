"""
Embedding generation module using OpenAI API.
"""
from typing import List, Dict, Any, Optional
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json

from ..utils.logger import get_logger
from ..utils.api_utils import create_embedding, get_cost_summary

logger = get_logger(__name__)


def generate_embeddings_batch(
    texts: List[str],
    model: str = "text-embedding-3-small",
    batch_size: int = 100,
    show_progress: bool = True
) -> np.ndarray:
    """
    Generate embeddings for texts using OpenAI API with batching.
    
    Args:
        texts: List of texts to embed
        model: OpenAI embedding model name
        batch_size: Number of texts per API call
        show_progress: Whether to show progress bar
        
    Returns:
        Numpy array of embeddings (shape: [num_texts, embedding_dim])
    """
    logger.info(f"Generating embeddings for {len(texts)} texts using {model}...")
    logger.info(f"Batch size: {batch_size}")
    
    all_embeddings = []
    
    # Create progress bar
    iterator = range(0, len(texts), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Generating embeddings", unit="batch")
    
    for i in iterator:
        batch = texts[i:i + batch_size]
        
        try:
            # Call OpenAI API
            batch_embeddings = create_embedding(batch, model=model)
            all_embeddings.extend(batch_embeddings)
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings for batch {i//batch_size}: {e}")
            raise
    
    # Convert to numpy array
    embeddings_array = np.array(all_embeddings, dtype='float32')
    
    logger.info(f"Generated embeddings shape: {embeddings_array.shape}")
    
    # Log cost
    cost_summary = get_cost_summary()
    logger.info(f"Embedding cost: ${cost_summary['total_cost']:.4f}")
    
    return embeddings_array


def save_embeddings(
    embeddings: np.ndarray,
    output_path: str,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save embeddings to disk.
    
    Args:
        embeddings: Numpy array of embeddings
        output_path: Path to save embeddings (.npy file)
        metadata: Optional metadata to save alongside embeddings
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save embeddings as numpy array
    np.save(output_file, embeddings)
    logger.info(f"Embeddings saved to {output_path}")
    
    # Save metadata if provided
    if metadata:
        metadata_path = output_file.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Metadata saved to {metadata_path}")


def load_embeddings(
    input_path: str,
    load_metadata: bool = True
) -> tuple[np.ndarray, Optional[Dict[str, Any]]]:
    """
    Load embeddings from disk.
    
    Args:
        input_path: Path to embeddings file (.npy)
        load_metadata: Whether to load metadata
        
    Returns:
        Tuple of (embeddings array, metadata dict or None)
    """
    embeddings = np.load(input_path)
    logger.info(f"Loaded embeddings from {input_path}, shape: {embeddings.shape}")
    
    metadata = None
    if load_metadata:
        metadata_path = Path(input_path).with_suffix('.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            logger.info(f"Loaded metadata from {metadata_path}")
    
    return embeddings, metadata


def embed_documents(
    documents: List[str],
    output_path: str,
    model: str = "text-embedding-3-small",
    batch_size: int = 100,
    save_to_disk: bool = True
) -> np.ndarray:
    """
    Generate and optionally save embeddings for documents.
    
    Args:
        documents: List of document texts
        output_path: Path to save embeddings
        model: OpenAI embedding model
        batch_size: Batch size for API calls
        save_to_disk: Whether to save embeddings to disk
        
    Returns:
        Numpy array of embeddings
    """
    # Generate embeddings
    embeddings = generate_embeddings_batch(
        documents,
        model=model,
        batch_size=batch_size,
        show_progress=True
    )
    
    # Save if requested
    if save_to_disk:
        metadata = {
            'num_documents': len(documents),
            'embedding_model': model,
            'embedding_dim': embeddings.shape[1],
            'batch_size': batch_size
        }
        save_embeddings(embeddings, output_path, metadata)
    
    return embeddings


def embed_query(
    query: str,
    model: str = "text-embedding-3-small"
) -> np.ndarray:
    """
    Generate embedding for a single query.
    
    Args:
        query: Query text
        model: OpenAI embedding model
        
    Returns:
        Numpy array of query embedding (shape: [1, embedding_dim])
    """
    embedding = create_embedding(query, model=model)
    return np.array(embedding, dtype='float32')
