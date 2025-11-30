"""
Simple in-memory vector store using numpy (ChromaDB alternative for Windows).
"""
import os
import numpy as np
from typing import List, Dict, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv

from ..utils.logger import get_logger

load_dotenv()
logger = get_logger(__name__)


class SimpleVectorStore:
    """
    Simple numpy-based vector store (no ChromaDB dependency).
    Works reliably on Windows.
    """
    
    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        api_key: Optional[str] = None
    ):
        """
        Initialize simple vector store.
        
        Args:
            embedding_model: OpenAI embedding model
            api_key: OpenAI API key
        """
        self.embedding_model = embedding_model
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        
        if not self.api_key:
            raise ValueError("OpenAI API key not found")
        
        self.client = OpenAI(api_key=self.api_key)
        
        # Storage
        self.documents = []
        self.embeddings = None
        self.ids = []
        self.metadatas = []
        
        logger.info("Simple vector store initialized")
    
    def add_documents(
        self,
        documents: List[str],
        ids: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ):
        """Add documents to vector store."""
        print(f"Adding {len(documents)} documents...")
        
        # Generate embeddings
        print("Generating embeddings via OpenAI...")
        response = self.client.embeddings.create(
            input=documents,
            model=self.embedding_model
        )
        
        new_embeddings = np.array([item.embedding for item in response.data])
        print(f"✓ Generated {len(new_embeddings)} embeddings")
        
        # Store
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
        
        self.documents.extend(documents)
        self.ids.extend(ids)
        self.metadatas.extend(metadatas or [{}] * len(documents))
        
        print(f"✓ Total documents in store: {len(self.documents)}")
    
    def query(
        self,
        query_text: str,
        n_results: int = 5
    ) -> Dict[str, Any]:
        """Query for similar documents."""
        # Generate query embedding
        response = self.client.embeddings.create(
            input=[query_text],
            model=self.embedding_model
        )
        query_embedding = np.array(response.data[0].embedding)
        
        # Compute cosine similarity
        similarities = np.dot(self.embeddings, query_embedding)
        similarities = similarities / (np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding))
        
        # Get top k
        top_indices = np.argsort(similarities)[::-1][:n_results]
        
        # Format results
        results = {
            'documents': [[self.documents[i] for i in top_indices]],
            'distances': [[1 - similarities[i] for i in top_indices]],  # Convert similarity to distance
            'metadatas': [[self.metadatas[i] for i in top_indices]]
        }
        
        return results
    
    def count(self) -> int:
        """Get document count."""
        return len(self.documents)


def build_simple_vector_store(
    chunks: List[str],
    chunk_ids: List[str],
    metadatas: Optional[List[Dict[str, Any]]] = None
) -> SimpleVectorStore:
    """Build simple vector store."""
    print("="*60)
    print("BUILDING SIMPLE VECTOR STORE (NumPy-based)")
    print("="*60)
    
    store = SimpleVectorStore()
    store.add_documents(chunks, chunk_ids, metadatas)
    
    print("="*60)
    print("VECTOR STORE BUILD COMPLETE")
    print(f"Total documents: {store.count()}")
    print("="*60)
    
    return store


def search_simple_vector_store(
    store: SimpleVectorStore,
    query: str,
    top_k: int = 5
) -> Dict[str, Any]:
    """Search simple vector store."""
    results = store.query(query, n_results=top_k)
    
    return {
        'query': query,
        'chunks': results['documents'][0],
        'distances': results['distances'][0],
        'metadatas': results['metadatas'][0]
    }
