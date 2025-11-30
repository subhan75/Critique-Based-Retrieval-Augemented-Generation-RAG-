"""
Comparison script for Vanilla RAG vs Critique RAG.

Computes:
- ROUGE-L scores
- BERTScore
- Hallucination rate (Critique only)
- Quality scores (Critique only)
- Refinement metrics (Critique only)
- Latency estimates
- Cost analysis
- Statistical significance tests
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
from rouge_score import rouge_scorer
from scipy.stats import ttest_rel
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

print("\n" + "="*70)
print("VANILLA RAG vs CRITIQUE RAG - COMPREHENSIVE COMPARISON")
print("="*70 + "\n")


# ============================================================
# 1. LOAD RESULTS
# ============================================================

def load_results(results_dir: str = "results") -> Dict[str, Any]:
    """Load both Vanilla and Critique RAG results, preferring oracle-enhanced versions."""
    results_path = Path(results_dir)
    
    # Check for oracle-enhanced files first
    vanilla_oracle_files = list(results_path.glob("vanilla_rag_with_oracle_*.json"))
    critique_oracle_files = list(results_path.glob("critique_rag_with_oracle_*.json"))
    
    # Fall back to regular files if oracle files don't exist
    vanilla_files = vanilla_oracle_files if vanilla_oracle_files else list(results_path.glob("vanilla_rag_evaluation_*.json"))
    critique_files = critique_oracle_files if critique_oracle_files else list(results_path.glob("critique_rag_evaluation_*.json"))
    
    if not vanilla_files:
        raise FileNotFoundError("No Vanilla RAG results found in results/ directory")
    if not critique_files:
        raise FileNotFoundError("No Critique RAG results found in results/ directory")
    
    # Load most recent files
    vanilla_file = max(vanilla_files, key=lambda p: p.stat().st_mtime)
    critique_file = max(critique_files, key=lambda p: p.stat().st_mtime)
    
    has_oracle = 'with_oracle' in vanilla_file.name and 'with_oracle' in critique_file.name
    
    print(f"📂 Loading Vanilla RAG: {vanilla_file.name}")
    print(f"📂 Loading Critique RAG: {critique_file.name}")
    if has_oracle:
        print(f"   ✅ Oracle scores detected!\n")
    else:
        print(f"   ℹ️  No oracle scores (run compute_oracle_scores.py to add them)\n")
    
    with open(vanilla_file, 'r', encoding='utf-8') as f:
        vanilla_data = json.load(f)
    
    with open(critique_file, 'r', encoding='utf-8') as f:
        critique_data = json.load(f)
    
    return {
        'vanilla': vanilla_data,
        'critique': critique_data,
        'vanilla_file': vanilla_file.name,
        'critique_file': critique_file.name,
        'has_oracle': has_oracle
    }


# ============================================================
# 2. COMPUTE ROUGE-L
# ============================================================

def compute_rouge_l(results: List[Dict]) -> Dict[str, Any]:
    """Compute ROUGE-L scores for all results."""
    print("   Computing ROUGE-L scores...")
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    scores = []
    for result in results:
        # Handle different key names (Vanilla uses 'predicted_answer', Critique uses 'answer')
        answer = result.get('answer') or result.get('predicted_answer', '')
        ground_truth = result.get('ground_truth', '')
        
        if not answer or not ground_truth:
            continue
            
        score = scorer.score(ground_truth, answer)
        scores.append(score['rougeL'].fmeasure)
    
    return {
        'scores': scores,
        'mean': float(np.mean(scores)),
        'std': float(np.std(scores)),
        'median': float(np.median(scores)),
        'min': float(np.min(scores)),
        'max': float(np.max(scores))
    }


# ============================================================
# 3. COMPUTE BERTSCORE
# ============================================================

def compute_bertscore(results: List[Dict]) -> Dict[str, Any]:
    """Compute BERTScore for all results."""
    try:
        print("   Computing BERTScore (this may take a few minutes)...")
        from bert_score import score as bert_score
        
        # Handle different key names and ensure strings
        predictions = []
        references = []
        
        for r in results:
            pred = r.get('answer') or r.get('predicted_answer', '')
            ref = r.get('ground_truth', '')
            
            # Convert to string and strip whitespace
            pred = str(pred).strip() if pred else ''
            ref = str(ref).strip() if ref else ''
            
            # Only include pairs where both are non-empty
            if pred and ref and len(pred) > 0 and len(ref) > 0:
                predictions.append(pred)
                references.append(ref)
        
        if not predictions:
            print("   ⚠️  No valid prediction-reference pairs found")
            return None
        
        print(f"   Computing BERTScore for {len(predictions)} valid pairs...")
        
        # Use roberta-large without rescaling first to see raw scores
        P, R, F1 = bert_score(
            predictions,
            references,
            model_type='roberta-large',
            num_layers=17,
            lang='en',
            verbose=False,
            device='cpu',
            rescale_with_baseline=False  # Try without rescaling first
        )
        
        # Convert to numpy
        P = P.numpy()
        R = R.numpy()
        F1 = F1.numpy()
        
        return {
            'precision': {
                'scores': P.tolist(),
                'mean': float(np.mean(P)),
                'std': float(np.std(P))
            },
            'recall': {
                'scores': R.tolist(),
                'mean': float(np.mean(R)),
                'std': float(np.std(R))
            },
            'f1': {
                'scores': F1.tolist(),
                'mean': float(np.mean(F1)),
                'std': float(np.std(F1))
            }
        }
    except ImportError as e:
        print(f"   ⚠️  BERTScore not available. Skipping. (Error: {str(e)[:50]}...)")
        return None
    except AttributeError as e:
        print(f"   ⚠️  BERTScore failed (NumPy compatibility issue). Skipping.")
        print(f"   💡 Tip: Try 'pip install numpy<2' if you need BERTScore")
        return None
    except Exception as e:
        print(f"   ⚠️  BERTScore computation failed: {str(e)[:100]}")
        return None


# ============================================================
# 4. COMPUTE HALLUCINATION RATE (Critique only)
# ============================================================

def compute_hallucination_rate(results: List[Dict]) -> Dict[str, Any]:
    """Compute hallucination rate from critique results."""
    total = len(results)
    has_hallucinations = sum(1 for r in results if r.get('has_hallucinations', False))
    
    # Get all hallucination examples
    hallucination_examples = []
    for r in results:
        if r.get('has_hallucinations') and r.get('hallucinations'):
            # Handle different key names
            question = r.get('question') or r.get('query', '')
            answer = r.get('answer') or r.get('predicted_answer', '')
            
            hallucination_examples.append({
                'question': question,
                'answer': answer,
                'hallucinations': r['hallucinations'],
                'quality_score': r.get('quality_score', 0)
            })
    
    return {
        'rate': has_hallucinations / total if total > 0 else 0,
        'count': has_hallucinations,
        'total': total,
        'percentage': (has_hallucinations / total * 100) if total > 0 else 0,
        'examples': hallucination_examples[:5]  # Top 5 examples
    }


# ============================================================
# 4.5. EXTRACT ORACLE SCORES (if available)
# ============================================================

def extract_oracle_scores(results: List[Dict]) -> Dict[str, Any]:
    """Extract oracle scores if they exist in results."""
    oracle_scores = []
    correctness_scores = []
    completeness_scores = []
    clarity_scores = []
    conciseness_scores = []
    
    for r in results:
        if 'oracle' in r:
            oracle = r['oracle']
            oracle_scores.append(oracle.get('oracle_score', 0))
            correctness_scores.append(oracle.get('correctness', 0))
            completeness_scores.append(oracle.get('completeness', 0))
            clarity_scores.append(oracle.get('clarity', 0))
            conciseness_scores.append(oracle.get('conciseness', 0))
    
    if not oracle_scores:
        return None
    
    return {
        'mean': float(np.mean(oracle_scores)),
        'std': float(np.std(oracle_scores)),
        'min': float(np.min(oracle_scores)),
        'max': float(np.max(oracle_scores)),
        'scores': oracle_scores,
        'components': {
            'correctness': {
                'mean': float(np.mean(correctness_scores)),
                'max': 4
            },
            'completeness': {
                'mean': float(np.mean(completeness_scores)),
                'max': 3
            },
            'clarity': {
                'mean': float(np.mean(clarity_scores)),
                'max': 2
            },
            'conciseness': {
                'mean': float(np.mean(conciseness_scores)),
                'max': 1
            }
        }
    }


# ============================================================
# 5. COMPUTE QUALITY METRICS (Critique only)
# ============================================================

def compute_quality_metrics(results: List[Dict]) -> Dict[str, Any]:
    """Compute quality score metrics from critique results."""
    quality_scores = [r['quality_score'] for r in results]
    
    high_quality = sum(1 for s in quality_scores if s >= 8.0)
    medium_quality = sum(1 for s in quality_scores if 5.0 <= s < 8.0)
    low_quality = sum(1 for s in quality_scores if s < 5.0)
    
    total = len(quality_scores)
    
    return {
        'mean': float(np.mean(quality_scores)),
        'std': float(np.std(quality_scores)),
        'median': float(np.median(quality_scores)),
        'min': float(np.min(quality_scores)),
        'max': float(np.max(quality_scores)),
        'high_quality_count': high_quality,
        'high_quality_rate': high_quality / total if total > 0 else 0,
        'medium_quality_count': medium_quality,
        'medium_quality_rate': medium_quality / total if total > 0 else 0,
        'low_quality_count': low_quality,
        'low_quality_rate': low_quality / total if total > 0 else 0,
        'distribution': {
            'high (≥8.0)': high_quality,
            'medium (5.0-7.9)': medium_quality,
            'low (<5.0)': low_quality
        }
    }


# ============================================================
# 6. COMPUTE REFINEMENT METRICS (Critique only)
# ============================================================

def compute_refinement_metrics(results: List[Dict]) -> Dict[str, Any]:
    """Compute refinement metrics from critique results."""
    total = len(results)
    refined = sum(1 for r in results if r.get('was_refined', False))
    
    # Analyze refinement impact
    refined_cases = [r for r in results if r.get('was_refined')]
    
    if refined_cases:
        improvements = [r.get('score_improvement', 0) for r in refined_cases]
        initial_scores = [r.get('initial_score', 0) for r in refined_cases]
        final_scores = [r.get('quality_score', 0) for r in refined_cases]
        
        positive_improvements = [i for i in improvements if i > 0]
        negative_improvements = [i for i in improvements if i < 0]
        neutral_improvements = [i for i in improvements if i == 0]
        
        return {
            'rate': refined / total if total > 0 else 0,
            'percentage': (refined / total * 100) if total > 0 else 0,
            'count': refined,
            'total': total,
            'avg_improvement': float(np.mean(improvements)) if improvements else 0,
            'avg_initial_score': float(np.mean(initial_scores)) if initial_scores else 0,
            'avg_final_score': float(np.mean(final_scores)) if final_scores else 0,
            'positive_improvements': len(positive_improvements),
            'negative_improvements': len(negative_improvements),
            'neutral_improvements': len(neutral_improvements),
            'best_improvement': float(max(improvements)) if improvements else 0,
            'worst_improvement': float(min(improvements)) if improvements else 0,
            'success_rate': len(positive_improvements) / len(improvements) if improvements else 0
        }
    else:
        return {
            'rate': 0,
            'percentage': 0,
            'count': 0,
            'total': total,
            'avg_improvement': 0
        }


# ============================================================
# 7. COMPUTE LATENCY (Estimated)
# ============================================================

def compute_latency(data: Dict[str, Any], is_critique: bool = False) -> Dict[str, float]:
    """Estimate latency based on API calls and typical response times."""
    total_queries = len(data['results'])
    
    # Estimate based on typical API latency
    if is_critique:
        refinement_rate = data['summary'].get('refinement_rate', 0.35)
        # retrieve (0.5s) + generate (1.5s) + critique (1.0s) + refine (2.0s * rate)
        avg_latency = 0.5 + 1.5 + 1.0 + (2.0 * refinement_rate)
    else:
        # retrieve (0.5s) + generate (1.5s)
        avg_latency = 0.5 + 1.5
    
    return {
        'avg_per_query': avg_latency,
        'total_estimated': avg_latency * total_queries,
        'note': 'Estimated based on typical API latency'
    }


# ============================================================
# 8. STATISTICAL COMPARISON
# ============================================================

def statistical_comparison(
    vanilla_scores: List[float],
    critique_scores: List[float],
    metric_name: str = "ROUGE-L"
) -> Dict[str, Any]:
    """Perform paired t-test to check statistical significance."""
    
    # Ensure same length
    min_len = min(len(vanilla_scores), len(critique_scores))
    vanilla_scores = vanilla_scores[:min_len]
    critique_scores = critique_scores[:min_len]
    
    # Paired t-test
    t_stat, p_value = ttest_rel(critique_scores, vanilla_scores)
    
    # Effect size (Cohen's d)
    diff = np.array(critique_scores) - np.array(vanilla_scores)
    cohens_d = np.mean(diff) / (np.std(diff) + 1e-10)  # Add small epsilon to avoid division by zero
    
    # Determine significance
    is_significant = p_value < 0.05
    mean_improvement = float(np.mean(diff))
    
    if is_significant:
        if mean_improvement > 0:
            interpretation = f"Statistically significant improvement in {metric_name}"
        else:
            interpretation = f"Statistically significant degradation in {metric_name}"
    else:
        interpretation = f"No statistically significant difference in {metric_name}"
    
    return {
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'is_significant': is_significant,
        'significance_level': 'p < 0.05' if is_significant else 'p ≥ 0.05',
        'cohens_d': float(cohens_d),
        'effect_size': (
            'Large' if abs(cohens_d) >= 0.8
            else 'Medium' if abs(cohens_d) >= 0.5
            else 'Small' if abs(cohens_d) >= 0.2
            else 'Negligible'
        ),
        'mean_improvement': mean_improvement,
        'interpretation': interpretation
    }


# ============================================================
# 9. GENERATE COMPARISON TABLE
# ============================================================

def generate_comparison_table(metrics: Dict[str, Any]) -> str:
    """Generate formatted comparison table."""
    
    # Calculate improvements (with safe division)
    vanilla_rouge = metrics['vanilla']['rouge_l']['mean']
    rouge_improvement = ((metrics['critique']['rouge_l']['mean'] - vanilla_rouge) / 
                         vanilla_rouge * 100) if vanilla_rouge > 0 else 0
    
    vanilla_latency = metrics['vanilla']['latency']['avg_per_query']
    latency_overhead = ((metrics['critique']['latency']['avg_per_query'] - vanilla_latency) / 
                        vanilla_latency * 100) if vanilla_latency > 0 else 0
    
    vanilla_cost = metrics['vanilla']['cost_per_query']
    cost_overhead = ((metrics['critique']['cost_per_query'] - vanilla_cost) / 
                     vanilla_cost * 100) if vanilla_cost > 0 else 0
    
    # BERTScore improvement (if available)
    bert_section = ""
    if metrics['vanilla'].get('bertscore') and metrics['critique'].get('bertscore'):
        bert_improvement = ((metrics['critique']['bertscore']['f1']['mean'] - 
                            metrics['vanilla']['bertscore']['f1']['mean']) / 
                           metrics['vanilla']['bertscore']['f1']['mean'] * 100)
        bert_section = f"""║ BERTScore F1 (mean)       │ {metrics['vanilla']['bertscore']['f1']['mean']:.4f}         │ {metrics['critique']['bertscore']['f1']['mean']:.4f}            ║
║ BERTScore improvement     │ -              │ {bert_improvement:+.1f}%             ║
║                           │                │                     ║"""
    
    # Oracle score section (if available)
    oracle_section = ""
    if metrics['vanilla'].get('oracle') and metrics['critique'].get('oracle'):
        oracle_improvement = ((metrics['critique']['oracle']['mean'] - 
                              metrics['vanilla']['oracle']['mean']) / 
                             metrics['vanilla']['oracle']['mean'] * 100)
        oracle_section = f"""║ Oracle Score (mean)       │ {metrics['vanilla']['oracle']['mean']:.4f}/10      │ {metrics['critique']['oracle']['mean']:.4f}/10         ║
║ Oracle Score (std)        │ {metrics['vanilla']['oracle']['std']:.4f}         │ {metrics['critique']['oracle']['std']:.4f}            ║
║ Oracle improvement        │ -              │ {oracle_improvement:+.1f}%             ║
║                           │                │                     ║
║ Oracle Correctness        │ {metrics['vanilla']['oracle']['components']['correctness']['mean']:.2f}/4          │ {metrics['critique']['oracle']['components']['correctness']['mean']:.2f}/4             ║
║ Oracle Completeness       │ {metrics['vanilla']['oracle']['components']['completeness']['mean']:.2f}/3          │ {metrics['critique']['oracle']['components']['completeness']['mean']:.2f}/3             ║
║ Oracle Clarity            │ {metrics['vanilla']['oracle']['components']['clarity']['mean']:.2f}/2          │ {metrics['critique']['oracle']['components']['clarity']['mean']:.2f}/2             ║
║ Oracle Conciseness        │ {metrics['vanilla']['oracle']['components']['conciseness']['mean']:.2f}/1          │ {metrics['critique']['oracle']['components']['conciseness']['mean']:.2f}/1             ║
║                           │                │                     ║"""
    
    table = f"""
╔══════════════════════════════════════════════════════════════════╗
║           VANILLA RAG vs CRITIQUE RAG COMPARISON                 ║
╠══════════════════════════════════════════════════════════════════╣
║ Metric                    │ Vanilla RAG    │ Critique RAG        ║
╠═══════════════════════════╪════════════════╪═════════════════════╣
║ ROUGE-L (mean)            │ {metrics['vanilla']['rouge_l']['mean']:.4f}         │ {metrics['critique']['rouge_l']['mean']:.4f}            ║
║ ROUGE-L (std)             │ {metrics['vanilla']['rouge_l']['std']:.4f}         │ {metrics['critique']['rouge_l']['std']:.4f}            ║
║ ROUGE-L improvement       │ -              │ {rouge_improvement:+.1f}%             ║
║                           │                │                     ║
{bert_section}{oracle_section}║ Hallucination Rate        │ N/A            │ {metrics['critique']['hallucination']['percentage']:.1f}%              ║
║ Hallucinations Detected   │ N/A            │ {metrics['critique']['hallucination']['count']}/{metrics['critique']['hallucination']['total']}               ║
║                           │                │                     ║
║ Avg Quality Score         │ N/A            │ {metrics['critique']['quality']['mean']:.2f}/10           ║
║ High Quality (≥8.0)       │ N/A            │ {metrics['critique']['quality']['high_quality_rate']*100:.1f}% ({metrics['critique']['quality']['high_quality_count']})       ║
║ Medium Quality (5.0-7.9)  │ N/A            │ {metrics['critique']['quality']['medium_quality_rate']*100:.1f}% ({metrics['critique']['quality']['medium_quality_count']})       ║
║ Low Quality (<5.0)        │ N/A            │ {metrics['critique']['quality']['low_quality_rate']*100:.1f}% ({metrics['critique']['quality']['low_quality_count']})        ║
║                           │                │                     ║
║ Refinement Rate           │ N/A            │ {metrics['critique']['refinement']['percentage']:.1f}%              ║
║ Refinements Applied       │ N/A            │ {metrics['critique']['refinement']['count']}/{metrics['critique']['refinement']['total']}               ║
║ Avg Score Improvement     │ N/A            │ {metrics['critique']['refinement']['avg_improvement']:+.2f}              ║
║ Refinement Success Rate   │ N/A            │ {metrics['critique']['refinement']['success_rate']*100:.1f}%              ║
║                           │                │                     ║
║ Avg Latency (per query)   │ {metrics['vanilla']['latency']['avg_per_query']:.2f}s          │ {metrics['critique']['latency']['avg_per_query']:.2f}s             ║
║ Latency Overhead          │ -              │ +{latency_overhead:.1f}%             ║
║                           │                │                     ║
║ Cost per Query            │ ${metrics['vanilla']['cost_per_query']:.4f}        │ ${metrics['critique']['cost_per_query']:.4f}           ║
║ Cost Overhead             │ -              │ +{cost_overhead:.1f}%             ║
║ Total Cost                │ ${metrics['vanilla']['total_cost']:.2f}         │ ${metrics['critique']['total_cost']:.2f}            ║
╠═══════════════════════════╧════════════════╧═════════════════════╣
║ Statistical Test (ROUGE-L):                                      ║
║   t-statistic = {metrics['statistical']['rouge_l']['t_statistic']:.3f}                                            ║
║   p-value = {metrics['statistical']['rouge_l']['p_value']:.4f}                                               ║
║   Effect size (Cohen's d) = {metrics['statistical']['rouge_l']['cohens_d']:.3f} ({metrics['statistical']['rouge_l']['effect_size']})                    ║
║   Result: {metrics['statistical']['rouge_l']['interpretation']:<56} ║
╚══════════════════════════════════════════════════════════════════╝
"""
    return table


# ============================================================
# 10. MAIN EXECUTION
# ============================================================

def main():
    # Load results
    print("Step 1: Loading results...")
    data = load_results()
    
    vanilla_results = data['vanilla']['results']
    critique_results = data['critique']['results']
    
    print(f"   ✅ Vanilla RAG: {len(vanilla_results)} results")
    print(f"   ✅ Critique RAG: {len(critique_results)} results\n")
    
    # Initialize metrics dictionary
    metrics = {
        'vanilla': {},
        'critique': {},
        'statistical': {}
    }
    
    # Compute metrics
    print("Step 2: Computing metrics...")
    
    # ROUGE-L
    metrics['vanilla']['rouge_l'] = compute_rouge_l(vanilla_results)
    metrics['critique']['rouge_l'] = compute_rouge_l(critique_results)
    print(f"   ✅ ROUGE-L computed")
    
    # BERTScore (optional)
    print()
    metrics['vanilla']['bertscore'] = compute_bertscore(vanilla_results)
    metrics['critique']['bertscore'] = compute_bertscore(critique_results)
    if metrics['vanilla']['bertscore']:
        print(f"   ✅ BERTScore computed")
    
    # Oracle scores (if available)
    print()
    if data.get('has_oracle', False):
        print("Step 2.5: Extracting Oracle scores...")
        metrics['vanilla']['oracle'] = extract_oracle_scores(vanilla_results)
        metrics['critique']['oracle'] = extract_oracle_scores(critique_results)
        if metrics['vanilla']['oracle'] and metrics['critique']['oracle']:
            print(f"   ✅ Oracle scores extracted")
        else:
            print(f"   ⚠️  Oracle scores not found in results")
    
    # Critique-specific metrics
    print()
    print("Step 3: Computing Critique-specific metrics...")
    metrics['critique']['hallucination'] = compute_hallucination_rate(critique_results)
    print(f"   ✅ Hallucination rate computed")
    
    metrics['critique']['quality'] = compute_quality_metrics(critique_results)
    print(f"   ✅ Quality metrics computed")
    
    metrics['critique']['refinement'] = compute_refinement_metrics(critique_results)
    print(f"   ✅ Refinement metrics computed")
    
    # Latency
    print()
    print("Step 4: Computing latency estimates...")
    metrics['vanilla']['latency'] = compute_latency(data['vanilla'], is_critique=False)
    metrics['critique']['latency'] = compute_latency(data['critique'], is_critique=True)
    print(f"   ✅ Latency estimated")
    
    # Cost
    print()
    print("Step 5: Computing cost metrics...")
    vanilla_summary = data['vanilla'].get('summary', {})
    critique_summary = data['critique'].get('summary', {})
    
    metrics['vanilla']['total_cost'] = vanilla_summary.get('total_cost', 0)
    metrics['critique']['total_cost'] = critique_summary.get('total_cost', 0)
    
    metrics['vanilla']['cost_per_query'] = metrics['vanilla']['total_cost'] / len(vanilla_results) if vanilla_results else 0
    metrics['critique']['cost_per_query'] = metrics['critique']['total_cost'] / len(critique_results) if critique_results else 0
    print(f"   ✅ Cost computed")
    
    # Statistical tests
    print()
    print("Step 6: Performing statistical tests...")
    metrics['statistical']['rouge_l'] = statistical_comparison(
        metrics['vanilla']['rouge_l']['scores'],
        metrics['critique']['rouge_l']['scores'],
        metric_name="ROUGE-L"
    )
    print(f"   ✅ Statistical tests completed")
    
    # Generate table
    print()
    print("Step 7: Generating comparison table...")
    table = generate_comparison_table(metrics)
    
    # Display table
    print("\n" + "="*70)
    print(table)
    
    # Save results
    print("Step 8: Saving results...")
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    # Save metrics (with custom JSON encoder for numpy types)
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.bool_):
                return bool(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NumpyEncoder, self).default(obj)
    
    with open(output_dir / "comparison_metrics.json", 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    
    # Save table
    with open(output_dir / "comparison_table.txt", 'w', encoding='utf-8') as f:
        f.write(table)
    
    print(f"   ✅ Metrics saved to: results/comparison_metrics.json")
    print(f"   ✅ Table saved to: results/comparison_table.txt")
    
    # Summary (with safe division)
    print("\n" + "="*70)
    print("KEY FINDINGS:")
    print("="*70)
    
    # ROUGE-L improvement
    vanilla_rouge = metrics['vanilla']['rouge_l']['mean']
    rouge_imp = ((metrics['critique']['rouge_l']['mean'] - vanilla_rouge) / vanilla_rouge * 100) if vanilla_rouge > 0 else 0
    print(f"📊 ROUGE-L Improvement: {rouge_imp:+.1f}%")
    
    # Oracle improvement (if available)
    if metrics['vanilla'].get('oracle') and metrics['critique'].get('oracle'):
        vanilla_oracle = metrics['vanilla']['oracle']['mean']
        oracle_imp = ((metrics['critique']['oracle']['mean'] - vanilla_oracle) / vanilla_oracle * 100) if vanilla_oracle > 0 else 0
        print(f"📊 Oracle Score Improvement: {oracle_imp:+.1f}% ({metrics['vanilla']['oracle']['mean']:.2f} → {metrics['critique']['oracle']['mean']:.2f})")
    
    print(f"📊 Statistical Significance: {metrics['statistical']['rouge_l']['interpretation']}")
    print(f"📊 Hallucination Rate: {metrics['critique']['hallucination']['percentage']:.1f}%")
    print(f"📊 Refinement Rate: {metrics['critique']['refinement']['percentage']:.1f}%")
    print(f"📊 Avg Quality Score: {metrics['critique']['quality']['mean']:.2f}/10")
    
    # Cost overhead
    vanilla_cost = metrics['vanilla']['cost_per_query']
    cost_ovh = ((metrics['critique']['cost_per_query'] - vanilla_cost) / vanilla_cost * 100) if vanilla_cost > 0 else 0
    print(f"💰 Cost Overhead: +{cost_ovh:.1f}%")
    
    # Latency overhead
    vanilla_lat = metrics['vanilla']['latency']['avg_per_query']
    lat_ovh = ((metrics['critique']['latency']['avg_per_query'] - vanilla_lat) / vanilla_lat * 100) if vanilla_lat > 0 else 0
    print(f"⏱️  Latency Overhead: +{lat_ovh:.1f}%")
    
    print("\n" + "="*70)
    print("✅ COMPARISON COMPLETE!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
