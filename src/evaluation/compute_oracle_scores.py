"""
Compute Oracle Quality Scores for ALL RAG models.
Updates: Now includes Reranker RAG support.
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
    raise ValueError("OPENAI_API_KEY not found. Please set it in your .env file.")

client = OpenAI(api_key=api_key)

def compute_oracle_score(
    prediction: str,
    reference: str,
    question: str,
    model: str = "gpt-4o-mini"
) -> Dict[str, Any]:
    """Use LLM to evaluate answer quality against ground truth."""
    
    prompt = f"""You are evaluating the quality of an AI-generated answer against a ground truth answer.

Question: {question}
Ground Truth Answer: {reference}
Generated Answer: {prediction}

Evaluate the generated answer on a scale of 0-10 using the following rubric:

1. CORRECTNESS (0-4 points):
   - 4: Perfectly correct, matches ground truth semantically
   - 0: Completely incorrect

2. COMPLETENESS (0-3 points):
   - 3: Covers all key information
   - 0: Missing most key information

3. CLARITY (0-2 points):
   - 2: Very clear and well-structured
   - 0: Confusing

4. CONCISENESS (0-1 point):
   - 1: Good length
   - 0: Too verbose or too brief

Return ONLY this JSON format:
{{
  "correctness": <0-4>,
  "completeness": <0-3>,
  "clarity": <0-2>,
  "conciseness": <0-1>,
  "oracle_score": <sum of above, 0-10>,
  "reasoning": "<brief 1 sentence explanation>"
}}"""
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error computing oracle score: {e}")
        return {'oracle_score': 0}

def compute_scores_for_file(results_file: Path, model_name: str):
    """Compute scores for a specific results file."""
    output_file = results_file.parent / f"{model_name}_with_oracle_{results_file.stem.split('_')[-1]}.json"
    
    print(f"\nProcessing {model_name.upper()}: {results_file.name}")
    
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = data.get('results', [])
    scores = []
    
    # Check if already computed
    if data.get('oracle_stats'):
        print(f"   ℹ️  Oracle scores already exist. Skipping.")
        return

    for result in tqdm(results, desc=f"   Computing {model_name} scores"):
        # Handle key differences
        pred = result.get('answer') or result.get('predicted_answer', '')
        ref = result.get('ground_truth', '')
        q = result.get('question') or result.get('query', '')
        
        if pred and ref:
            oracle = compute_oracle_score(pred, ref, q)
            result['oracle'] = oracle
            scores.append(oracle.get('oracle_score', 0))
    
    # Add stats to data object
    if scores:
        data['oracle_stats'] = {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'min': float(np.min(scores)),
            'max': float(np.max(scores))
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        print(f"   ✅ Saved to {output_file.name} (Mean Score: {np.mean(scores):.2f})")
    else:
        print("   ⚠️  No valid results to score.")

def main():
    print("\n" + "="*60)
    print("COMPUTING ORACLE SCORES FOR ALL MODELS")
    print("="*60)
    
    results_dir = Path("results")
    
    # Helper to find latest non-oracle file
    def get_latest(pattern):
        files = list(results_dir.glob(pattern))
        # Filter out files that are already oracle outputs to avoid re-processing outputs
        files = [f for f in files if "with_oracle" not in f.name]
        return max(files, key=lambda p: p.stat().st_mtime) if files else None

    # 1. Vanilla
    vanilla_file = get_latest("vanilla_rag_evaluation_*.json")
    if vanilla_file: compute_scores_for_file(vanilla_file, "vanilla_rag")
    
    # 2. Reranker (This is the new part!)
    reranker_file = get_latest("reranker_rag_evaluation_*.json")
    if reranker_file: compute_scores_for_file(reranker_file, "reranker_rag")
    else: print("\n❌ No Reranker results found. Run evaluate_reranker_rag.py first.")

    # 3. Critique
    critique_file = get_latest("critique_rag_evaluation_*.json")
    if critique_file: compute_scores_for_file(critique_file, "critique_rag")

    print("\n" + "="*60)
    print("Done! Now run: python src/evaluation/compare_all.py")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()