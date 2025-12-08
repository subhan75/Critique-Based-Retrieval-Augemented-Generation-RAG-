"""
Comprehensive Comparison: Vanilla vs Reranker vs Critique.
"""
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
from rouge_score import rouge_scorer
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("RAG MODEL SHOWDOWN: VANILLA vs RERANKER vs CRITIQUE")
print("="*80 + "\n")

# ============================================================
# 1. LOAD RESULTS
# ============================================================

def load_results(results_dir: str = "results") -> Dict[str, Any]:
    """Load results for all three models."""
    results_path = Path(results_dir)
    
    # Helper to find latest file
    def find_latest(pattern):
        files = list(results_path.glob(pattern))
        return max(files, key=lambda p: p.stat().st_mtime) if files else None

    # Load files (prefer oracle versions if available)
    vanilla_file = find_latest("vanilla_rag_with_oracle_*.json") or find_latest("vanilla_rag_evaluation_*.json")
    reranker_file = find_latest("reranker_rag_with_oracle_*.json") or find_latest("reranker_rag_evaluation_*.json")
    critique_file = find_latest("critique_rag_with_oracle_*.json") or find_latest("critique_rag_evaluation_*.json")
    
    data = {}
    
    if vanilla_file:
        print(f"📂 Vanilla:  {vanilla_file.name}")
        with open(vanilla_file, 'r', encoding='utf-8') as f: data['vanilla'] = json.load(f)
    else:
        print("⚠️  Vanilla RAG results not found")

    if reranker_file:
        print(f"📂 Reranker: {reranker_file.name}")
        with open(reranker_file, 'r', encoding='utf-8') as f: data['reranker'] = json.load(f)
    else:
        print("⚠️  Reranker RAG results not found (Run evaluate_reranker_rag.py first)")

    if critique_file:
        print(f"📂 Critique: {critique_file.name}")
        with open(critique_file, 'r', encoding='utf-8') as f: data['critique'] = json.load(f)
    else:
        print("⚠️  Critique RAG results not found")
        
    print("")
    return data

# ============================================================
# 2. COMPUTE METRICS
# ============================================================

def compute_rouge_l(results: List[Dict]) -> Dict[str, float]:
    """Compute ROUGE-L stats."""
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = []
    
    for r in results:
        # Handle different key names across files
        pred = r.get('answer') or r.get('predicted_answer', '')
        ref = r.get('ground_truth', '')
        if pred and ref:
            scores.append(scorer.score(ref, pred)['rougeL'].fmeasure)
            
    if not scores: return {'mean': 0.0, 'std': 0.0}
    
    return {
        'mean': float(np.mean(scores)),
        'std': float(np.std(scores))
    }

def compute_cost_per_query(data: Dict[str, Any], num_queries: int) -> float:
    """Extract cost per query safely."""
    summary = data.get('cost_summary', {}) or data.get('summary', {})
    total = summary.get('total_cost', 0)
    return total / num_queries if num_queries > 0 else 0

def get_oracle_score(results: List[Dict]) -> float:
    """Extract mean oracle score if available."""
    scores = [r.get('oracle', {}).get('oracle_score', 0) for r in results if 'oracle' in r]
    return float(np.mean(scores)) if scores else 0.0

# ============================================================
# 3. GENERATE 3-WAY TABLE
# ============================================================

def generate_3way_table(metrics: Dict[str, Any]) -> str:
    """Generate ASCII table for 3 models."""
    
    # Helper to format cells safely
    def fmt(val, is_pct=False, is_cost=False):
        if val is None: return "N/A".center(12)
        if is_cost: return f"${val:.4f}".center(12)
        if is_pct: return f"{val:+.1f}%".center(12)
        return f"{val:.4f}".center(12)

    m = metrics # Alias for brevity
    
    # Calculate Improvements vs Vanilla
    def get_imp(model_key, metric_key):
        base = m['vanilla'][metric_key]['mean']
        target = m[model_key][metric_key]['mean']
        return ((target - base) / base * 100) if base > 0 else 0

    rerank_rouge_imp = get_imp('reranker', 'rouge')
    critique_rouge_imp = get_imp('critique', 'rouge')
    
    rerank_oracle = m['reranker']['oracle']
    critique_oracle = m['critique']['oracle']
    vanilla_oracle = m['vanilla']['oracle']

    table = f"""
╔══════════════════════════════════════════════════════════════════════════╗
║                  RAG MODEL COMPARISON (N={m['count']})                         ║
╠═════════════════════════╦══════════════╦══════════════╦══════════════════╣
║ Metric                  ║ Vanilla      ║ Reranker     ║ Critique         ║
╠═════════════════════════╬══════════════╬══════════════╬══════════════════╣
║ ROUGE-L (Mean)          ║ {fmt(m['vanilla']['rouge']['mean'])} ║ {fmt(m['reranker']['rouge']['mean'])} ║ {fmt(m['critique']['rouge']['mean'])}     ║
║ ROUGE-L (Improvement)   ║      -       ║ {fmt(rerank_rouge_imp, True)} ║ {fmt(critique_rouge_imp, True)}     ║
╠═════════════════════════╬══════════════╬══════════════╬══════════════════╣
║ Oracle Score (0-10)     ║ {fmt(vanilla_oracle)} ║ {fmt(rerank_oracle)} ║ {fmt(critique_oracle)}     ║
╠═════════════════════════╬══════════════╬══════════════╬══════════════════╣
║ Avg Latency (Est.)      ║    ~2.0s     ║    ~3.0s     ║    ~5.0s+        ║
║ Cost Per Query          ║ {fmt(m['vanilla']['cost'], is_cost=True)} ║ {fmt(m['reranker']['cost'], is_cost=True)} ║ {fmt(m['critique']['cost'], is_cost=True)}     ║
╚═════════════════════════╩══════════════╩══════════════╩══════════════════╝
"""
    return table

# ============================================================
# 4. MAIN EXECUTION
# ============================================================

def main():
    # 1. Load Data
    data = load_results()
    
    # Ensure we have data
    if not all(k in data for k in ['vanilla', 'reranker', 'critique']):
        print("\n❌ Missing data files. Please run evaluations for all 3 models first.")
        return

    # 2. Extract Results Lists
    v_res = data['vanilla']['results']
    r_res = data['reranker']['results']
    c_res = data['critique']['results']
    
    # Use the smallest dataset size for fair comparison
    min_len = min(len(v_res), len(r_res), len(c_res))
    v_res, r_res, c_res = v_res[:min_len], r_res[:min_len], c_res[:min_len]
    
    print(f"Comparing on common set of {min_len} queries...")

    # 3. Compute Metrics
    metrics = {
        'count': min_len,
        'vanilla': {
            'rouge': compute_rouge_l(v_res),
            'cost': compute_cost_per_query(data['vanilla'], min_len),
            'oracle': get_oracle_score(v_res)
        },
        'reranker': {
            'rouge': compute_rouge_l(r_res),
            'cost': compute_cost_per_query(data['reranker'], min_len),
            'oracle': get_oracle_score(r_res)
        },
        'critique': {
            'rouge': compute_rouge_l(c_res),
            'cost': compute_cost_per_query(data['critique'], min_len),
            'oracle': get_oracle_score(c_res)
        }
    }

    # 4. Display
    print(generate_3way_table(metrics))
    
    # 5. Reranker Specific Analysis
    print("\n🔎 RERANKER DEEP DIVE")
    print(f"   Initial Retrieval: {data['reranker']['metadata']['initial_k']} chunks")
    print(f"   Final Selected:    {data['reranker']['metadata']['top_k']} chunks")
    
    # Calculate how often the top-ranked chunk changed (proxy for reranker effectiveness)
    # Note: This requires detailed retrieval logs which might not be in the basic json
    # simplified check:
    print(f"   Avg ROUGE-L: {metrics['reranker']['rouge']['mean']:.4f}")
    if metrics['reranker']['rouge']['mean'] > metrics['vanilla']['rouge']['mean']:
        print("   ✅ Reranker is outperforming Vanilla baseline.")
    else:
        print("   ⚠️ Reranker is performing similar to or worse than baseline.")

if __name__ == "__main__":
    main()