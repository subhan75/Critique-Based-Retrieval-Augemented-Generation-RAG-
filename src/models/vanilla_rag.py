"""
Vanilla RAG Model Implementation.
Standard Retrieval-Augmented Generation without critique mechanism.
"""
import os
from typing import List, Dict, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv

from ..utils.logger import get_logger
from ..utils.api_utils import track_api_call

# Load environment variables
load_dotenv()

logger = get_logger(__name__)


class VanillaRAG:
    """
    Vanilla RAG implementation using ChromaDB for retrieval and OpenAI for generation.
    """
    
    def __init__(
        self,
        vector_store,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 500,
        top_k: int = 5
    ):
        """
        Initialize Vanilla RAG model.
        
        Args:
            vector_store: ChromaDB VectorStore instance
            model: OpenAI model to use for generation
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens in response
            top_k: Number of chunks to retrieve
        """
        self.vector_store = vector_store
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_k = top_k
        
        # Initialize OpenAI client
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.client = OpenAI(api_key=api_key)
        
        logger.info(f"Initialized Vanilla RAG with model: {model}")
        logger.info(f"Parameters: temperature={temperature}, max_tokens={max_tokens}, top_k={top_k}")
    
    def retrieve(self, query: str) -> Dict[str, Any]:
        """
        Retrieve relevant chunks from vector store.
        
        Args:
            query: User query
            
        Returns:
            Dictionary with retrieved chunks and metadata
        """
        logger.info(f"Retrieving top {self.top_k} chunks for query: '{query}'")
        
        try:
            results = self.vector_store.query(
                query_text=query,
                n_results=self.top_k
            )
            
            # Extract results
            chunks = results['documents'][0] if results.get('documents') else []
            distances = results['distances'][0] if results.get('distances') else []
            metadatas = results['metadatas'][0] if results.get('metadatas') else []
            
            logger.info(f"Retrieved {len(chunks)} chunks")
            
            return {
                'chunks': chunks,
                'distances': distances,
                'metadatas': metadatas,
                'query': query
            }
            
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            raise
    
    def generate(
        self,
        query: str,
        retrieved_chunks: List[str],
        return_prompt: bool = False
    ) -> Dict[str, Any]:
        """
        Generate answer using retrieved chunks.
        
        Args:
            query: User query
            retrieved_chunks: List of retrieved text chunks
            return_prompt: Whether to return the prompt used
            
        Returns:
            Dictionary with generated answer and metadata
        """
        logger.info(f"Generating answer for query: '{query}'")
        
        # Construct context from retrieved chunks
        context = "\n\n".join([
            f"[Passage {i+1}]\n{chunk}"
            for i, chunk in enumerate(retrieved_chunks)
        ])
        
        # Create prompt
        prompt = self._create_prompt(query, context)
        
        try:
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Extract answer
            answer = response.choices[0].message.content.strip()
            
            # Track API usage
            usage = response.usage
            track_api_call(
                model=self.model,
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens
            )
            
            logger.info(f"Generated answer ({usage.completion_tokens} tokens)")
            logger.info(f"API usage: {usage.prompt_tokens} input + {usage.completion_tokens} output tokens")
            
            result = {
                'answer': answer,
                'query': query,
                'model': self.model,
                'usage': {
                    'prompt_tokens': usage.prompt_tokens,
                    'completion_tokens': usage.completion_tokens,
                    'total_tokens': usage.total_tokens
                }
            }
            
            if return_prompt:
                result['prompt'] = prompt
            
            return result
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
    
    def _create_prompt(self, query: str, context: str) -> str:
        """
        Create prompt for answer generation.
        
        Args:
            query: User query
            context: Retrieved context passages
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""Answer the following question based on the provided context passages. 
Be concise and accurate. If the answer cannot be found in the context, say "I cannot answer based on the provided context."

Context:
{context}

Question: {query}

Answer:"""
        
        return prompt
    
    def answer(
        self,
        query: str,
        return_details: bool = False
    ) -> Dict[str, Any]:
        """
        Complete RAG pipeline: retrieve and generate answer.
        
        Args:
            query: User query
            return_details: Whether to return retrieval details
            
        Returns:
            Dictionary with answer and optional details
        """
        logger.info("="*60)
        logger.info(f"VANILLA RAG - Processing query: '{query}'")
        logger.info("="*60)
        
        # Step 1: Retrieve relevant chunks
        retrieval_results = self.retrieve(query)
        
        # Step 2: Generate answer
        generation_results = self.generate(
            query=query,
            retrieved_chunks=retrieval_results['chunks']
        )
        
        # Combine results
        result = {
            'query': query,
            'answer': generation_results['answer'],
            'model': self.model
        }
        
        if return_details:
            result['retrieval'] = {
                'chunks': retrieval_results['chunks'],
                'distances': retrieval_results['distances'],
                'metadatas': retrieval_results['metadatas'],
                'top_k': self.top_k
            }
            result['generation'] = {
                'usage': generation_results['usage'],
                'temperature': self.temperature,
                'max_tokens': self.max_tokens
            }
        
        logger.info("="*60)
        logger.info("VANILLA RAG - Complete")
        logger.info("="*60)
        
        return result
    
    def batch_answer(
        self,
        queries: List[str],
        return_details: bool = False,
        verbose: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Process multiple queries in batch.
        
        Args:
            queries: List of user queries
            return_details: Whether to return retrieval details
            verbose: Whether to log progress
            
        Returns:
            List of result dictionaries
        """
        logger.info(f"Processing batch of {len(queries)} queries")
        
        results = []
        for i, query in enumerate(queries):
            if verbose:
                logger.info(f"\nProcessing query {i+1}/{len(queries)}")
            
            try:
                result = self.answer(query, return_details=return_details)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process query {i+1}: {e}")
                results.append({
                    'query': query,
                    'answer': None,
                    'error': str(e)
                })
        
        logger.info(f"\nBatch processing complete: {len(results)} results")
        return results


def create_vanilla_rag(
    vector_store,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    max_tokens: int = 500,
    top_k: int = 5
) -> VanillaRAG:
    """
    Factory function to create a Vanilla RAG instance.
    
    Args:
        vector_store: ChromaDB VectorStore instance
        model: OpenAI model to use
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response
        top_k: Number of chunks to retrieve
        
    Returns:
        Initialized VanillaRAG instance
    """
    return VanillaRAG(
        vector_store=vector_store,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        top_k=top_k
    )
