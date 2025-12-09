"""
Reranker RAG Model Implementation using OpenAI.
Uses robust Pointwise Scoring to rerank candidates by relevance.
"""
import os
import json
from typing import List, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv

from ..utils.logger import get_logger
from ..utils.api_utils import track_api_call

# Load environment variables
load_dotenv()

logger = get_logger(__name__)


class RerankerRAG:
    """
    Reranker RAG implementation using OpenAI Pointwise Scoring.
    """
    
    def __init__(
        self,
        vector_store,
        model: str = "gpt-4o-mini",
        reranker_model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 500,
        initial_k: int = 20,
        top_k: int = 5
    ):
        self.vector_store = vector_store
        self.model = model
        self.reranker_model = reranker_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.initial_k = initial_k
        self.top_k = top_k
        
        # Initialize OpenAI client
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.client = OpenAI(api_key=api_key)
        
        logger.info(f"Initialized Pointwise Reranker RAG")
        logger.info(f"Config: initial_k={initial_k}, top_k={top_k}, reranker={reranker_model}")

    def rerank_chunks(self, query: str, chunks: List[str]) -> Dict[str, Any]:
        """
        Rerank chunks using Pointwise scoring (0-10).
        """
        if not chunks:
            return {"indices": [], "scores": []}

        # Construct a structured prompt for batch scoring
        passages_text = ""
        for i, chunk in enumerate(chunks):
            # FIX: Removed the [:400] truncation. We now use the full text.
            clean_chunk = chunk.replace('\n', ' ')
            passages_text += f"ID_{i}: {clean_chunk}\n"
        
        prompt = f"""You are an expert relevance evaluator.
Query: {query}

Below are {len(chunks)} text passages. Rate each passage's relevance to the query on a scale of 0-10.
0.0 = Completely Irrelevant
5.0 = Partially Relevant (Topic match but no answer)
10.0 = Highly Relevant (Contains the specific answer)

Return a JSON object mapping the ID (e.g., "ID_0") to the score (number).
Example: {{"ID_0": 2.5, "ID_1": 9.0, ...}}

Passages:
{passages_text}"""

        try:
            response = self.client.chat.completions.create(
                model=self.reranker_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"}
            )

            # Track cost
            track_api_call(self.reranker_model, response.usage.prompt_tokens, response.usage.completion_tokens)

            content = response.choices[0].message.content
            scores_map = json.loads(content)
            
            # Parse scores and keep track of original indices
            scored_items = []
            for i in range(len(chunks)):
                key = f"ID_{i}"
                score = float(scores_map.get(key, 0.0))
                scored_items.append({
                    'index': i,
                    'score': score,
                    'chunk': chunks[i]
                })
            
            # Sort by score descending
            scored_items.sort(key=lambda x: x['score'], reverse=True)
            
            # Extract sorted indices and scores
            sorted_indices = [item['index'] for item in scored_items]
            sorted_scores = [item['score'] for item in scored_items]
            sorted_chunks = [item['chunk'] for item in scored_items]
            
            return {
                "chunks": sorted_chunks,
                "indices": sorted_indices,
                "scores": sorted_scores
            }

        except Exception as e:
            logger.warning(f"Reranking failed: {e}. Falling back to original order.")
            return {
                "chunks": chunks, 
                "indices": list(range(len(chunks))), 
                "scores": [0.0] * len(chunks)
            }

    def retrieve(self, query: str) -> Dict[str, Any]:
        """
        Retrieve and Rerank pipeline.
        """
        logger.info(f"1. Retrieving top {self.initial_k} candidates...")
        
        # 1. Initial Retrieval (Vector Search)
        results = self.vector_store.query(
            query_text=query,
            n_results=self.initial_k
        )
        
        candidates = results['documents'][0] if results.get('documents') else []
        metadatas = results['metadatas'][0] if results.get('metadatas') else []
        
        if not candidates:
            return {'chunks': [], 'metadatas': []}

        # 2. Reranking (Pointwise)
        logger.info(f"2. Reranking {len(candidates)} candidates...")
        rerank_result = self.rerank_chunks(query, candidates)
        
        reranked_chunks = rerank_result['chunks']
        sorted_indices = rerank_result['indices']
        
        # Reorder metadata to match chunks
        reranked_metadatas = [metadatas[i] for i in sorted_indices]

        # 3. Top-K Selection
        final_chunks = reranked_chunks[:self.top_k]
        final_metadatas = reranked_metadatas[:self.top_k]
        
        logger.info(f"3. Selected top {len(final_chunks)} chunks after reranking")
        
        return {
            'chunks': final_chunks,
            'metadatas': final_metadatas,
            'all_candidates': len(candidates)
        }

    def generate(self, query: str, chunks: List[str]) -> Dict[str, Any]:
        """
        Generate answer using the refined chunks.
        """
        context = "\n\n".join([f"[Passage {i+1}]\n{chunk}" for i, chunk in enumerate(chunks)])
        
        prompt = f"""Answer the following question based on the provided context passages. 
If the answer cannot be found in the context, say "I cannot answer based on the provided context."

Context:
{context}

Question: {query}

Answer:"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        track_api_call(self.model, response.usage.prompt_tokens, response.usage.completion_tokens)
        
        return {
            "answer": response.choices[0].message.content.strip(),
            "usage": response.usage
        }

    def answer(self, query: str, return_details: bool = False) -> Dict[str, Any]:
        """
        End-to-end Reranker RAG pipeline.
        """
        # Retrieve & Rerank
        retrieval = self.retrieve(query)
        
        # Generate
        generation = self.generate(query, retrieval['chunks'])
        
        result = {
            'query': query,
            'answer': generation['answer'],
            'model': self.model
        }
        
        if return_details:
            result['retrieval'] = retrieval
            result['generation'] = generation
            
        return result

def create_reranker_rag(
    vector_store,
    model: str = "gpt-4o-mini",
    initial_k: int = 20,
    top_k: int = 5
) -> RerankerRAG:
    """Factory function."""
    return RerankerRAG(
        vector_store=vector_store,
        model=model,
        initial_k=initial_k,
        top_k=top_k
    )