"""
ChromaDB Vector Store for RAG System.
Handles document storage, embedding, and retrieval using ChromaDB.
"""
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from chromadb.config import Settings

from ..utils.logger import get_logger

logger = get_logger(__name__)


class VectorStore:
    """
    ChromaDB-based vector store for document retrieval.
    """
    
    def __init__(
        self,
        collection_name: str = "rag_documents",
        persist_directory: str = "data/chroma",
        embedding_model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        use_manual_embeddings: bool = True
    ):
        """
        Initialize ChromaDB vector store.
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist ChromaDB data
            embedding_model: OpenAI embedding model to use
            api_key: OpenAI API key (defaults to env variable)
            use_manual_embeddings: If True, manually generate embeddings (more reliable on Windows)
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        self.use_manual_embeddings = use_manual_embeddings
        
        # Get API key
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        # Initialize OpenAI client for manual embeddings
        if use_manual_embeddings:
            from openai import OpenAI
            self.openai_client = OpenAI(api_key=self.api_key)
            logger.info("Using manual embeddings (more reliable on Windows)")
        
        # Create persist directory
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client with telemetry disabled
        logger.info(f"Initializing ChromaDB with persistent storage at: {persist_directory}")
        settings = Settings(
            anonymized_telemetry=False,
            allow_reset=True
        )
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=settings
        )
        
        # Create embedding function (only if not using manual)
        if not use_manual_embeddings:
            self.embedding_function = OpenAIEmbeddingFunction(
                api_key=self.api_key,
                model_name=embedding_model
            )
        else:
            self.embedding_function = None
        
        self.collection = None
        logger.info(f"ChromaDB initialized successfully")
    
    def create_collection(self, reset: bool = False) -> None:
        """
        Create or get ChromaDB collection.
        
        Args:
            reset: If True, delete existing collection and create new one
        """
        try:
            if reset:
                # Delete existing collection
                try:
                    self.client.delete_collection(name=self.collection_name)
                    logger.info(f"Deleted existing collection: {self.collection_name}")
                except Exception:
                    pass  # Collection doesn't exist
            
            # Create or get collection
            if self.use_manual_embeddings:
                # No embedding function - we'll provide embeddings manually
                self.collection = self.client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}  # Use cosine similarity
                )
            else:
                self.collection = self.client.get_or_create_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_function,
                    metadata={"hnsw:space": "cosine"}  # Use cosine similarity
                )
            
            logger.info(f"Collection '{self.collection_name}' ready")
            logger.info(f"Collection count: {self.collection.count()} documents")
            
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            raise
    
    def add_documents(
        self,
        documents: List[str],
        ids: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Add documents to ChromaDB collection.
        ChromaDB will automatically embed the documents using OpenAI.
        
        Args:
            documents: List of document texts
            ids: List of unique document IDs
            metadatas: Optional list of metadata dicts for each document
        """
        if self.collection is None:
            raise ValueError("Collection not initialized. Call create_collection() first.")
        
        if len(documents) != len(ids):
            raise ValueError("Number of documents must match number of IDs")
        
        if metadatas and len(metadatas) != len(documents):
            raise ValueError("Number of metadatas must match number of documents")
        
        try:
            logger.info(f"Adding {len(documents)} documents to collection...")
            
            if self.use_manual_embeddings:
                print("[DEBUG] Using manual embeddings")
                logger.info("Generating embeddings manually via OpenAI API...")
                
                # Generate embeddings in batches
                batch_size = 10
                total_batches = (len(documents) - 1) // batch_size + 1
                print(f"[DEBUG] Will process {total_batches} batches")
                
                for i in range(0, len(documents), batch_size):
                    batch_docs = documents[i:i+batch_size]
                    batch_ids = ids[i:i+batch_size]
                    batch_metas = metadatas[i:i+batch_size] if metadatas else None
                    
                    batch_num = i // batch_size + 1
                    print(f"[DEBUG] Batch {batch_num}/{total_batches}: Embedding {len(batch_docs)} documents...")
                    logger.info(f"Batch {batch_num}/{total_batches}: Embedding {len(batch_docs)} documents...")
                    
                    # Generate embeddings
                    print(f"[DEBUG] Calling OpenAI API...")
                    response = self.openai_client.embeddings.create(
                        input=batch_docs,
                        model=self.embedding_model
                    )
                    print(f"[DEBUG] Got embeddings from OpenAI")
                    batch_embeddings = [item.embedding for item in response.data]
                    
                    # Add to collection with embeddings
                    print(f"[DEBUG] Adding to ChromaDB collection...")
                    import sys
                    sys.stdout.flush()
                    
                    try:
                        # Add one document at a time to avoid hanging
                        for j, (doc_id, embedding, doc, meta) in enumerate(zip(batch_ids, batch_embeddings, batch_docs, batch_metas or [None]*len(batch_ids))):
                            self.collection.add(
                                ids=[doc_id],
                                embeddings=[embedding],
                                documents=[doc],
                                metadatas=[meta] if meta else None
                            )
                            if (j + 1) % 5 == 0:
                                print(f"[DEBUG]   Added {j+1}/{len(batch_ids)} documents...")
                                sys.stdout.flush()
                    except Exception as e:
                        print(f"[DEBUG] Error adding to collection: {e}")
                        raise
                    
                    print(f"[DEBUG] ✓ Batch {batch_num}/{total_batches} complete")
                    logger.info(f"✓ Batch {batch_num}/{total_batches} complete")
            else:
                logger.info("Using ChromaDB automatic embeddings...")
                
                # Add documents in batches (ChromaDB will embed automatically)
                batch_size = 10
                total_batches = (len(documents) - 1) // batch_size + 1
                
                for i in range(0, len(documents), batch_size):
                    batch_docs = documents[i:i+batch_size]
                    batch_ids = ids[i:i+batch_size]
                    batch_metas = metadatas[i:i+batch_size] if metadatas else None
                    
                    batch_num = i // batch_size + 1
                    logger.info(f"Processing batch {batch_num}/{total_batches}...")
                    
                    self.collection.add(
                        documents=batch_docs,
                        ids=batch_ids,
                        metadatas=batch_metas
                    )
                    
                    logger.info(f"✓ Batch {batch_num}/{total_batches} complete")
            
            logger.info(f"Successfully added {len(documents)} documents")
            logger.info(f"Total collection count: {self.collection.count()}")
            
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise
    
    def query(
        self,
        query_text: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        include: List[str] = None
    ) -> Dict[str, Any]:
        """
        Query the collection for similar documents.
        
        Args:
            query_text: Query text to search for
            n_results: Number of results to return
            where: Optional metadata filter
            include: What to include in results (documents, distances, metadatas)
        
        Returns:
            Dictionary with query results
        """
        if self.collection is None:
            raise ValueError("Collection not initialized. Call create_collection() first.")
        
        if include is None:
            include = ['documents', 'distances', 'metadatas']
        
        try:
            if self.use_manual_embeddings:
                # Generate query embedding manually
                response = self.openai_client.embeddings.create(
                    input=[query_text],
                    model=self.embedding_model
                )
                query_embedding = response.data[0].embedding
                
                # Query with embedding
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results,
                    where=where,
                    include=include
                )
            else:
                # Let ChromaDB embed the query automatically
                results = self.collection.query(
                    query_texts=[query_text],
                    n_results=n_results,
                    where=where,
                    include=include
                )
            
            return results
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.
        
        Returns:
            Dictionary with collection statistics
        """
        if self.collection is None:
            return {"error": "Collection not initialized"}
        
        try:
            count = self.collection.count()
            
            # Get a sample to check metadata
            sample = self.collection.peek(limit=1)
            
            stats = {
                "collection_name": self.collection_name,
                "document_count": count,
                "embedding_model": self.embedding_model,
                "persist_directory": self.persist_directory,
                "has_documents": count > 0
            }
            
            if sample and sample.get('metadatas'):
                stats["sample_metadata"] = sample['metadatas'][0] if sample['metadatas'] else None
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}
    
    def delete_collection(self) -> None:
        """
        Delete the collection.
        """
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = None
            logger.info(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            raise


def build_vector_store(
    chunks: List[str],
    chunk_ids: List[str],
    metadatas: Optional[List[Dict[str, Any]]] = None,
    collection_name: str = "rag_documents",
    reset: bool = False
) -> VectorStore:
    """
    Build a ChromaDB vector store from document chunks.
    
    Args:
        chunks: List of text chunks
        chunk_ids: List of unique IDs for each chunk
        metadatas: Optional metadata for each chunk
        collection_name: Name for the collection
        reset: Whether to reset existing collection
    
    Returns:
        Initialized VectorStore with documents added
    """
    logger.info("="*60)
    logger.info("BUILDING CHROMADB VECTOR STORE")
    logger.info("="*60)
    
    # Initialize vector store with manual embeddings (more reliable on Windows)
    print(f"[DEBUG] Initializing VectorStore with manual_embeddings=True")
    store = VectorStore(
        collection_name=collection_name,
        use_manual_embeddings=True  # Use manual embeddings to avoid hanging
    )
    print(f"[DEBUG] VectorStore initialized")
    
    # Create collection
    print(f"[DEBUG] Creating collection...")
    store.create_collection(reset=reset)
    print(f"[DEBUG] Collection created")
    
    # Add documents
    store.add_documents(
        documents=chunks,
        ids=chunk_ids,
        metadatas=metadatas
    )
    
    # Show stats
    stats = store.get_collection_stats()
    logger.info("\nVector Store Statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    
    logger.info("="*60)
    logger.info("VECTOR STORE BUILD COMPLETE")
    logger.info("="*60)
    
    return store


def search_vector_store(
    store: VectorStore,
    query: str,
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Search the vector store for relevant chunks.
    
    Args:
        store: VectorStore instance
        query: Query text
        top_k: Number of results to return
    
    Returns:
        Dictionary with search results
    """
    logger.info(f"\nSearching for: '{query}'")
    logger.info(f"Retrieving top {top_k} results...")
    
    results = store.query(
        query_text=query,
        n_results=top_k
    )
    
    # Format results
    formatted_results = {
        'query': query,
        'chunks': [],
        'distances': results['distances'][0] if results.get('distances') else [],
        'metadatas': results['metadatas'][0] if results.get('metadatas') else []
    }
    
    if results.get('documents'):
        formatted_results['chunks'] = results['documents'][0]
    
    logger.info(f"Found {len(formatted_results['chunks'])} results")
    
    return formatted_results
