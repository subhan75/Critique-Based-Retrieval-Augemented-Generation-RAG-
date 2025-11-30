"""
Compute Oracle Quality Scores for RAG evaluation results.

Oracle scores use an LLM to evaluate answer quality against ground truth,
providing a human-like assessment of correctness, completeness, and clarity.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any
from openai import OpenAI
from tqdm import tqdm
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
api_key = os.environ.get('OPENAI_API_KEY')
if not api_key:
    raise ValueError("OPENAI_API_KEY not found. Please set it in your .env file or environment variables.")

client = OpenAI(api_key=api_key)

def compute_oracle_score(
    prediction: str,
    reference: str,
    question: str,
    model: str = "gpt-4o-mini"
) -> Dict[str, Any]:
    """
    Use LLM to evaluate answer quality against ground truth.
    
    Args:
        prediction: Generated answer
        reference: Ground truth answer
        question: Original question
        model: OpenAI model to use for evaluation
        
    Returns:
        Dictionary with:
            - correctness: 0-4
            - completeness: 0-3
            - clarity: 0-2
            - conciseness: 0-1
            - oracle_score: 0-10 (sum of above)
            - reasoning: explanation
    """
    
    prompt = f"""You are evaluating the quality of an AI-generated answer against a ground truth answer.

Question: {question}

Ground Truth Answer: {reference}

Generated Answer: {prediction}

Evaluate the generated answer on a scale of 0-10 using the following rubric:

1. CORRECTNESS (0-4 points):
   - Does it provide the correct information?
   - Does it match the ground truth semantically (not necessarily word-for-word)?
   - 4: Perfectly correct, matches ground truth
   - 3: Mostly correct, minor differences
   - 2: Partially correct, some errors
   - 1: Mostly incorrect
   - 0: Completely incorrect

2. COMPLETENESS (0-3 points):
   - Does it cover all key points from the ground truth?
   - 3: Covers all key information
   - 2: Covers most key information
   - 1: Covers some key information
   - 0: Missing most key information

3. CLARITY (0-2 points):
   - Is it clear, well-structured, and easy to understand?
   - 2: Very clear and well-structured
   - 1: Acceptable clarity
   - 0: Confusing or poorly structured

4. CONCISENESS (0-1 point):
   - Is it appropriately concise (not too verbose, not too brief)?
   - 1: Good length, appropriate detail
   - 0: Too verbose or too brief

IMPORTANT:
- Focus on semantic correctness, not exact word matching
- A longer answer that's correct should not be penalized heavily
- The generated answer may provide additional context - that's okay if it's correct

Return ONLY this JSON format:
{{
  "correctness": <0-4>,
  "completeness": <0-3>,
  "clarity": <0-2>,
  "conciseness": <0-1>,
  "oracle_score": <sum of above, 0-10>,
  "reasoning": "<brief 1-2 sentence explanation>"
}}"""
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        
        # Validate and ensure oracle_score is the sum
        oracle_score = (
            result.get('correctness', 0) +
            result.get('completeness', 0) +
            result.get('clarity', 0) +
            result.get('conciseness', 0)
        )
        result['oracle_score'] = oracle_score
        
        return result
        
    except Exception as e:
        print(f"Error computing oracle score: {e}")
        return {
            'correctness': 0,
            'completeness': 0,
            'clarity': 0,
            'conciseness': 0,
            'oracle_score': 0,
            'reasoning': f'Error: {str(e)}'
        }


def compute_oracle_scores_for_results(
    results_file: Path,
    output_file: Path = None,
    model: str = "gpt-4o-mini"
) -> Dict[str, Any]:
    """
    Compute oracle scores for all results in a file.
    
    Args:
        results_file: Path to evaluation results JSON
        output_file: Path to save oracle scores (optional)
        model: OpenAI model to use
        
    Returns:
        Dictionary with oracle scores and statistics
    """
    
    print(f"\n{'='*70}")
    print(f"COMPUTING ORACLE SCORES: {results_file.name}")
    print(f"{'='*70}\n")
    
    # Load results
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = data.get('results', [])
    print(f"Loaded {len(results)} results\n")
    
    # Compute oracle scores
    oracle_scores = []
    
    for i, result in enumerate(tqdm(results, desc="Computing oracle scores")):
        # Extract fields (handle different key names)
        question = result.get('question') or result.get('query', '')
        prediction = result.get('answer') or result.get('predicted_answer', '')
        reference = result.get('ground_truth', '')
        
        if not prediction or not reference:
            print(f"Warning: Skipping result {i} - missing prediction or reference")
            continue
        
        # Compute oracle score
        oracle = compute_oracle_score(prediction, reference, question, model)
        
        # Add to result
        result['oracle'] = oracle
        oracle_scores.append(oracle['oracle_score'])
    
    # Compute statistics
    oracle_stats = {
        'mean': float(np.mean(oracle_scores)),
        'std': float(np.std(oracle_scores)),
        'min': float(np.min(oracle_scores)),
        'max': float(np.max(oracle_scores)),
        'median': float(np.median(oracle_scores)),
        'count': len(oracle_scores)
    }
    
    # Component statistics
    component_stats = {
        'correctness': {
            'mean': float(np.mean([r['oracle']['correctness'] for r in results if 'oracle' in r])),
            'max': 4
        },
        'completeness': {
            'mean': float(np.mean([r['oracle']['completeness'] for r in results if 'oracle' in r])),
            'max': 3
        },
        'clarity': {
            'mean': float(np.mean([r['oracle']['clarity'] for r in results if 'oracle' in r])),
            'max': 2
        },
        'conciseness': {
            'mean': float(np.mean([r['oracle']['conciseness'] for r in results if 'oracle' in r])),
            'max': 1
        }
    }
    
    # Print summary
    print(f"\n{'='*70}")
    print("ORACLE SCORE SUMMARY")
    print(f"{'='*70}")
    print(f"Mean:   {oracle_stats['mean']:.4f}/10")
    print(f"Std:    {oracle_stats['std']:.4f}")
    print(f"Min:    {oracle_stats['min']:.4f}")
    print(f"Max:    {oracle_stats['max']:.4f}")
    print(f"Median: {oracle_stats['median']:.4f}")
    print(f"\nComponent Breakdown:")
    print(f"  Correctness:  {component_stats['correctness']['mean']:.2f}/4")
    print(f"  Completeness: {component_stats['completeness']['mean']:.2f}/3")
    print(f"  Clarity:      {component_stats['clarity']['mean']:.2f}/2")
    print(f"  Conciseness:  {component_stats['conciseness']['mean']:.2f}/1")
    print(f"{'='*70}\n")
    
    # Save results with oracle scores
    if output_file:
        data['oracle_stats'] = oracle_stats
        data['oracle_component_stats'] = component_stats
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        print(f"✅ Saved results with oracle scores to: {output_file}\n")
    
    return {
        'oracle_stats': oracle_stats,
        'component_stats': component_stats,
        'results': results
    }


def main():
    """Main function to compute oracle scores for evaluation results."""
    
    print("\n" + "="*70)
    print("ORACLE QUALITY SCORE COMPUTATION")
    print("="*70 + "\n")
    
    # Find latest evaluation results
    results_dir = Path("results")
    
    # Find vanilla and critique results
    vanilla_files = sorted(results_dir.glob("vanilla_rag_evaluation_*.json"))
    critique_files = sorted(results_dir.glob("critique_rag_evaluation_*.json"))
    
    if not vanilla_files or not critique_files:
        print("❌ Error: Could not find evaluation result files")
        print(f"   Looking in: {results_dir.absolute()}")
        print(f"   Vanilla files found: {len(vanilla_files)}")
        print(f"   Critique files found: {len(critique_files)}")
        return
    
    # Use latest files
    vanilla_file = vanilla_files[-1]
    critique_file = critique_files[-1]
    
    print(f"Vanilla RAG results:  {vanilla_file.name}")
    print(f"Critique RAG results: {critique_file.name}\n")
    
    # Compute oracle scores for vanilla
    print("Step 1: Computing oracle scores for Vanilla RAG...")
    vanilla_output = results_dir / f"vanilla_rag_with_oracle_{vanilla_file.stem.split('_')[-1]}.json"
    vanilla_oracle = compute_oracle_scores_for_results(
        vanilla_file,
        vanilla_output,
        model="gpt-4o-mini"
    )
    
    # Compute oracle scores for critique
    print("Step 2: Computing oracle scores for Critique RAG...")
    critique_output = results_dir / f"critique_rag_with_oracle_{critique_file.stem.split('_')[-1]}.json"
    critique_oracle = compute_oracle_scores_for_results(
        critique_file,
        critique_output,
        model="gpt-4o-mini"
    )
    
    # Compare
    print("\n" + "="*70)
    print("ORACLE SCORE COMPARISON")
    print("="*70)
    
    vanilla_mean = vanilla_oracle['oracle_stats']['mean']
    critique_mean = critique_oracle['oracle_stats']['mean']
    improvement = ((critique_mean - vanilla_mean) / vanilla_mean * 100) if vanilla_mean > 0 else 0
    
    print(f"\nOverall Oracle Score:")
    print(f"  Vanilla RAG:  {vanilla_mean:.4f}/10")
    print(f"  Critique RAG: {critique_mean:.4f}/10")
    print(f"  Improvement:  {improvement:+.1f}%")
    
    print(f"\nComponent Scores:")
    for component in ['correctness', 'completeness', 'clarity', 'conciseness']:
        v_score = vanilla_oracle['component_stats'][component]['mean']
        c_score = critique_oracle['component_stats'][component]['mean']
        max_score = vanilla_oracle['component_stats'][component]['max']
        
        print(f"  {component.capitalize():13s}: {v_score:.2f}/{max_score} → {c_score:.2f}/{max_score}")
    
    print("\n" + "="*70)
    print("✅ ORACLE SCORES COMPUTED SUCCESSFULLY!")
    print("="*70)
    print(f"\nNext step: Run 'python compare_results.py' to see updated comparison\n")


if __name__ == "__main__":
    main()
