"""
Test script for data pipeline components.
"""
import os
# Disable torch to avoid DLL issues on Windows
os.environ['HF_DATASETS_DISABLE_TORCH'] = '1'

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.dataset_loader import load_squad, prepare_dataset_for_rag
from src.data.chunk_builder import chunk_documents, get_chunk_texts, get_chunk_statistics
from src.data.vector_store import build_vector_store, search_vector_store
from src.utils.logger import setup_logger
from src.utils.api_utils import setup_openai_client, get_cost_summary
import yaml

# Setup logging
logger = setup_logger(__name__, log_file="logs/test_data_pipeline.log")


def test_dataset_loading():
    """Test dataset loading."""
    logger.info("=" * 50)
    logger.info("TEST 1: Dataset Loading")
    logger.info("=" * 50)
    
    try:
        # Load small sample for testing
        # Loads 10 queries from SQuAD v1.1 validation set
        dataset = load_squad(
            num_samples=10
        )
        
        logger.info(f"[OK] Loaded {len(dataset['queries'])} queries")
        logger.info(f"[OK] Sample query: {dataset['queries'][0]}")
        logger.info(f"[OK] Sample answer: {dataset['answers'][0]}")
        logger.info(f"[OK] Number of documents: {len(dataset['documents'])}")
        
        return dataset
        
    except Exception as e:
        logger.error(f"[FAIL] Dataset loading failed: {e}")
        raise


def test_chunking(dataset):
    """Test document chunking."""
    logger.info("\n" + "=" * 50)
    logger.info("TEST 2: Document Chunking")
    logger.info("=" * 50)
    
    try:
        # Chunk documents
        chunks = chunk_documents(
            dataset['documents'],
            chunk_size=512,
            overlap=50
        )
        
        logger.info(f"[OK] Created {len(chunks)} chunks")
        
        # Get statistics
        stats = get_chunk_statistics(chunks)
        logger.info(f"[OK] Chunk statistics: {stats}")
        
        # Get chunk texts
        chunk_texts = get_chunk_texts(chunks)
        if len(chunk_texts) > 0:
            logger.info(f"[OK] Sample chunk: {chunk_texts[0][:100]}...")
        else:
            logger.warning("No chunks created - documents may be too short")
        
        return chunk_texts
        
    except Exception as e:
        logger.error(f"[FAIL] Chunking failed: {e}")
        raise


def test_vector_store_building(chunk_texts):
    """Test ChromaDB vector store building."""
    logger.info("\n" + "=" * 50)
    logger.info("TEST 3: ChromaDB Vector Store Building")
    logger.info("=" * 50)
    
    try:
        # Setup OpenAI client
        setup_openai_client()
        
        # Use only first 10 chunks for testing
        sample_chunks = chunk_texts[:10]
        
        # Create chunk IDs
        chunk_ids = [f"chunk_{i}" for i in range(len(sample_chunks))]
        
        # Create metadata
        metadatas = [{"chunk_index": i, "source": "squad"} for i in range(len(sample_chunks))]
        
        # Build vector store
        vector_store = build_vector_store(
            chunks=sample_chunks,
            chunk_ids=chunk_ids,
            metadatas=metadatas,
            collection_name="test_squad_chunks",
            reset=True  # Reset for clean test
        )
        
        logger.info(f"[OK] Vector store built successfully")
        
        # Get stats
        stats = vector_store.get_collection_stats()
        logger.info(f"[OK] Collection stats: {stats}")
        
        # Check cost
        cost = get_cost_summary()
        logger.info(f"[OK] Embedding cost: ${cost['total_cost']:.6f}")
        
        return vector_store, sample_chunks
        
    except Exception as e:
        logger.error(f"[FAIL] Vector store building failed: {e}")
        raise


def test_retrieval(vector_store, chunk_texts):
    """Test retrieval from ChromaDB."""
    logger.info("\n" + "=" * 50)
    logger.info("TEST 4: ChromaDB Retrieval")
    logger.info("=" * 50)
    
    try:
        # Test query
        test_query = "What is the capital of France?"
        logger.info(f"Test query: {test_query}")
        
        # Search vector store
        results = search_vector_store(
            store=vector_store,
            query=test_query,
            top_k=3
        )
        
        logger.info(f"[OK] Retrieved {len(results['chunks'])} results")
        logger.info("\nTop 3 results:")
        for i, (chunk, distance, metadata) in enumerate(zip(
            results['chunks'],
            results['distances'],
            results['metadatas']
        )):
            logger.info(f"  {i+1}. Distance: {distance:.4f}")
            logger.info(f"     Metadata: {metadata}")
            logger.info(f"     Text: {chunk[:100]}...")
        
        # Check cost
        cost = get_cost_summary()
        logger.info(f"\n[OK] Total cost so far: ${cost['total_cost']:.6f}")
        
    except Exception as e:
        logger.error(f"[FAIL] Retrieval failed: {e}")
        raise


def main():
    """Run all tests."""
    logger.info("Starting Data Pipeline Tests with ChromaDB")
    logger.info("=" * 50)
    
    try:
        # Test 1: Dataset loading
        dataset = test_dataset_loading()
        
        # Test 2: Chunking
        chunk_texts = test_chunking(dataset)
        
        # Test 3: Vector store building (includes embedding)
        vector_store, sample_chunks = test_vector_store_building(chunk_texts)
        
        # Test 4: Retrieval
        test_retrieval(vector_store, sample_chunks)
        
        logger.info("\n" + "=" * 50)
        logger.info("[OK] ALL TESTS PASSED!")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"\n[FAIL] TESTS FAILED: {e}")
        raise


if __name__ == "__main__":
    main()
