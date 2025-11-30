"""
Critique-Enhanced RAG with single refinement.

This module implements a RAG system that uses LLM-based critique to evaluate
and refine generated answers.
"""
import os
import json
from typing import Dict, Any, List, Optional
from openai import OpenAI
from dotenv import load_dotenv

from ..utils.logger import get_logger
from ..utils.api_utils import track_api_call

load_dotenv()
logger = get_logger(__name__)


class CritiqueRAG:
    """
    Critique-Enhanced RAG with answer evaluation and refinement.
    
    Pipeline:
    1. Retrieve relevant chunks (same as Vanilla RAG)
    2. Generate initial answer (same as Vanilla RAG)
    3. Critique answer quality
    4. Refine if quality_score < 8.0
    5. Return best answer with metadata
    """
    
    def __init__(
        self,
        vector_store,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 500,
        quality_threshold: float = 9.0
    ):
        """
        Initialize Critique RAG.
        
        Args:
            vector_store: SimpleVectorStore instance for retrieval
            model: OpenAI model for generation and critique
            temperature: Sampling temperature
            max_tokens: Maximum tokens for generation
            quality_threshold: Minimum quality score to accept without refinement
        """
        self.vector_store = vector_store
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.quality_threshold = quality_threshold
        
        # Initialize OpenAI client
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.client = OpenAI(api_key=api_key)
        
        logger.info(f"CritiqueRAG initialized with model={model}, threshold={quality_threshold}")
    
    def retrieve(self, query: str, k: int = 5) -> List[str]:
        """
        Retrieve top-k relevant chunks.
        
        Args:
            query: User query
            k: Number of chunks to retrieve
            
        Returns:
            List of retrieved chunk texts
        """
        results = self.vector_store.query(query, n_results=k)
        chunks = results['documents'][0]
        
        logger.info(f"Retrieved {len(chunks)} chunks for query")
        return chunks
    
    def generate(self, query: str, chunks: List[str]) -> str:
        """
        Generate answer from query and context chunks.
        
        Args:
            query: User query
            chunks: Retrieved context chunks
            
        Returns:
            Generated answer
        """
        context = "\n\n".join(chunks)
        
        prompt = f"""Context:
{context}

Question: {query}

Based ONLY on the context above, answer the question. If you cannot answer based on the context, say "I cannot answer based on the given context."

Answer:"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        # Track API call
        track_api_call(
            self.model,
            response.usage.prompt_tokens,
            response.usage.completion_tokens
        )
        
        answer = response.choices[0].message.content.strip()
        logger.info(f"Generated answer: {answer[:100]}...")
        
        return answer
    
    def critique_answer(
        self,
        query: str,
        chunks: List[str],
        answer: str
    ) -> Dict[str, Any]:
        """
        Critique answer quality using LLM.
        
        Evaluates:
        - Factual correctness based on context
        - Hallucinations (unsupported claims)
        - Overall quality score (0-10)
        
        Args:
            query: User query
            chunks: Context chunks used for generation
            answer: Generated answer to critique
            
        Returns:
            Critique dictionary with:
                - is_correct: bool
                - has_hallucinations: bool
                - hallucinations: list of unsupported claims
                - quality_score: float (0-10)
                - reasoning: str (explanation)
        """
        context = "\n\n".join(chunks)
        
        critique_prompt = f"""You are a STRICT expert evaluator for question-answering systems.

Question: {query}

Context provided:
{context}

Generated Answer: {answer}

Evaluate this answer using the following rubric (BE STRICT - most answers should score 5-7):

1. FACTUAL ACCURACY (0-3 points):
   - 3: All facts are correct and verifiable from context
   - 2: Mostly correct, one minor unsupported detail
   - 1: Some correct facts, but contains unsupported claims
   - 0: Contains hallucinations or major factual errors

2. COMPLETENESS (0-3 points):
   - 3: Fully answers the question with all key details
   - 2: Answers the question but missing some details
   - 1: Partially answers, significant gaps
   - 0: Does not answer the question or too brief

3. RELEVANCE (0-2 points):
   - 2: Directly addresses the question, no irrelevant info
   - 1: Mostly relevant but includes some off-topic content
   - 0: Off-topic or doesn't address the question

4. CLARITY (0-2 points):
   - 2: Clear, well-structured, easy to understand
   - 1: Understandable but could be clearer
   - 0: Confusing or poorly structured

IMPORTANT SCORING GUIDELINES:
- Be STRICT: Average answers should score 5-6, not 8-9
- Score below 7 if there are ANY issues
- Only score 9-10 for truly exceptional answers
- Check EVERY claim against the context

Respond with ONLY this JSON format:
{{
  "factual_accuracy": <0-3>,
  "completeness": <0-3>,
  "relevance": <0-2>,
  "clarity": <0-2>,
  "quality_score": <sum of above, 0-10>,
  "is_correct": <true if factual_accuracy >= 2>,
  "has_hallucinations": <true if any unsupported claims>,
  "hallucinations": ["specific unsupported claim 1", "claim 2"] or [],
  "issues": ["specific issue 1", "issue 2"] or [],
  "strengths": ["strength 1", "strength 2"] or [],
  "reasoning": "<brief explanation of score>"
}}"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": critique_prompt}],
            temperature=0,
            response_format={"type": "json_object"}
        )
        
        # Track API call
        track_api_call(
            self.model,
            response.usage.prompt_tokens,
            response.usage.completion_tokens
        )
        
        try:
            critique = json.loads(response.choices[0].message.content)
            logger.info(f"Critique: score={critique.get('quality_score', 0)}, "
                       f"hallucinations={critique.get('has_hallucinations', False)}")
            return critique
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse critique JSON: {e}")
            # Return default critique on parse failure
            return {
                'is_correct': False,
                'has_hallucinations': True,
                'hallucinations': ['JSON parse failed'],
                'quality_score': 0.0,
                'reasoning': f'Critique parsing failed: {str(e)}'
            }
    
    def refine_answer(
        self,
        query: str,
        chunks: List[str],
        answer: str,
        critique: Dict[str, Any]
    ) -> str:
        """
        Refine answer based on critique feedback.
        
        Args:
            query: User query
            chunks: Context chunks
            answer: Original answer
            critique: Critique feedback
            
        Returns:
            Refined answer
        """
        context = "\n\n".join(chunks)
        
        # Extract problems from critique
        problems = []
        
        if critique.get('hallucinations'):
            for hallucination in critique['hallucinations']:
                problems.append(f"Hallucination: {hallucination}")
        
        if not critique.get('is_correct'):
            problems.append("Answer may be factually incorrect based on context")
        
        if critique.get('reasoning'):
            problems.append(f"Issue: {critique['reasoning']}")
        
        problems_text = "\n".join(f"- {p}" for p in problems)
        
        refine_prompt = f"""You are refining an answer based on expert critique feedback.

Question: {query}

Context:
{context}

Original Answer: {answer}

Critique Feedback:
- Quality Score: {critique.get('quality_score', 0)}/10
- Factual Accuracy: {critique.get('factual_accuracy', 0)}/3
- Completeness: {critique.get('completeness', 0)}/3
- Relevance: {critique.get('relevance', 0)}/2
- Clarity: {critique.get('clarity', 0)}/2

Specific Issues to Address:
{problems_text}

Your task: Generate an IMPROVED answer that:
1. Addresses EACH specific issue mentioned above
2. Maintains similar length to the original (do not make it unnecessarily long)
3. Uses ONLY information explicitly stated in the context
4. Directly and completely answers the question
5. Is clear, concise, and well-structured
6. If context lacks information, state "Based on the provided context, [answer what you can]"

CRITICAL RULES:
- Fix the identified issues WITHOUT adding unnecessary details
- Stay grounded in the context - NO hallucinations
- Keep the answer focused and relevant
- Aim for a quality score of 9-10

Improved Answer:"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": refine_prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        # Track API call
        track_api_call(
            self.model,
            response.usage.prompt_tokens,
            response.usage.completion_tokens
        )
        
        refined_answer = response.choices[0].message.content.strip()
        logger.info(f"Refined answer: {refined_answer[:100]}...")
        
        return refined_answer
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Main query method with critique and single refinement.
        
        Pipeline:
        1. Retrieve relevant chunks
        2. Generate initial answer
        3. Critique answer
        4. If quality_score < threshold, refine once
        5. Re-critique refined answer
        6. Return best answer with metadata
        
        Args:
            question: User question
            
        Returns:
            Dictionary with:
                - answer: Final answer
                - quality_score: Final quality score
                - has_hallucinations: bool
                - hallucinations: list
                - is_high_quality: bool (score >= threshold)
                - was_refined: bool
                - initial_score: Initial quality score (if refined)
                - chunks: Retrieved chunks
                - critique: Full critique object
        """
        logger.info(f"Processing query: {question}")
        
        # Step 1: Retrieve
        chunks = self.retrieve(question, k=5)
        
        # Step 2: Generate initial answer
        answer = self.generate(question, chunks)
        
        # Step 3: Critique
        critique = self.critique_answer(question, chunks, answer)
        initial_score = critique['quality_score']
        
        # Step 4: Refine if needed (single refinement)
        refined = False
        if critique['quality_score'] < self.quality_threshold:
            logger.info(f"Quality score {critique['quality_score']} < {self.quality_threshold}, refining...")
            
            # Refine answer
            answer = self.refine_answer(question, chunks, answer, critique)
            
            # Re-critique refined answer
            critique = self.critique_answer(question, chunks, answer)
            refined = True
            
            logger.info(f"After refinement: score {initial_score} → {critique['quality_score']}")
        else:
            logger.info(f"Quality score {critique['quality_score']} >= {self.quality_threshold}, accepting answer")
        
        # Return result with metadata
        result = {
            'answer': answer,
            'quality_score': critique['quality_score'],
            'has_hallucinations': critique.get('has_hallucinations', False),
            'hallucinations': critique.get('hallucinations', []),
            'is_high_quality': critique['quality_score'] >= self.quality_threshold,
            'is_correct': critique.get('is_correct', False),
            'reasoning': critique.get('reasoning', ''),
            'was_refined': refined,
            'chunks': chunks,
            'critique': critique
        }
        
        # Add initial score if refined
        if refined:
            result['initial_score'] = initial_score
            result['score_improvement'] = critique['quality_score'] - initial_score
        
        return result


def create_critique_rag(
    vector_store,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    max_tokens: int = 500,
    quality_threshold: float = 9.0
) -> CritiqueRAG:
    """
    Factory function to create CritiqueRAG instance.
    
    Args:
        vector_store: SimpleVectorStore instance
        model: OpenAI model name
        temperature: Sampling temperature
        max_tokens: Maximum tokens for generation
        quality_threshold: Minimum quality score to accept
        
    Returns:
        CritiqueRAG instance
    """
    return CritiqueRAG(
        vector_store=vector_store,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        quality_threshold=quality_threshold
    )
