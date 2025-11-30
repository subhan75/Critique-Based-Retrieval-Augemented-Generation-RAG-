"""
Debug script to check BERTScore calculation and data quality.
"""
import json
from pathlib import Path

print("="*70)
print("DEBUGGING BERTSCORE CALCULATION")
print("="*70 + "\n")

# Load results
vanilla_file = Path("results/vanilla_rag_evaluation_20251126_151034.json")
critique_file = Path("results/critique_rag_evaluation_20251126_184901.json")

print("Step 1: Loading results...")
with open(vanilla_file, 'r', encoding='utf-8') as f:
    vanilla_data = json.load(f)
    vanilla_results = vanilla_data['results']

with open(critique_file, 'r', encoding='utf-8') as f:
    critique_data = json.load(f)
    critique_results = critique_data['results']

print(f"   Vanilla: {len(vanilla_results)} results")
print(f"   Critique: {len(critique_results)} results\n")

# Check first few examples
print("Step 2: Checking data quality...")
print("\n" + "="*70)
print("VANILLA RAG - First 3 Examples:")
print("="*70)

for i in range(min(3, len(vanilla_results))):
    r = vanilla_results[i]
    pred = r.get('predicted_answer', 'N/A')
    ref = r.get('ground_truth', 'N/A')
    
    print(f"\nExample {i+1}:")
    print(f"  Prediction: {pred[:100]}..." if len(str(pred)) > 100 else f"  Prediction: {pred}")
    print(f"  Reference:  {ref[:100]}..." if len(str(ref)) > 100 else f"  Reference:  {ref}")
    print(f"  Pred length: {len(str(pred))}, Ref length: {len(str(ref))}")

print("\n" + "="*70)
print("CRITIQUE RAG - First 3 Examples:")
print("="*70)

for i in range(min(3, len(critique_results))):
    r = critique_results[i]
    pred = r.get('answer', 'N/A')
    ref = r.get('ground_truth', 'N/A')
    
    print(f"\nExample {i+1}:")
    print(f"  Prediction: {pred[:100]}..." if len(str(pred)) > 100 else f"  Prediction: {pred}")
    print(f"  Reference:  {ref[:100]}..." if len(str(ref)) > 100 else f"  Reference:  {ref}")
    print(f"  Pred length: {len(str(pred))}, Ref length: {len(str(ref))}")

# Test BERTScore on a few examples
print("\n" + "="*70)
print("Step 3: Testing BERTScore on sample data...")
print("="*70 + "\n")

try:
    from bert_score import score as bert_score
    
    # Get first 5 valid pairs from vanilla
    predictions = []
    references = []
    
    for r in vanilla_results[:5]:
        pred = str(r.get('predicted_answer', '')).strip()
        ref = str(r.get('ground_truth', '')).strip()
        if pred and ref:
            predictions.append(pred)
            references.append(ref)
    
    if predictions:
        print(f"Testing BERTScore on {len(predictions)} vanilla examples...")
        print("Using model: roberta-large, layer 17, NO rescaling\n")
        
        P, R, F1 = bert_score(
            predictions,
            references,
            model_type='roberta-large',
            num_layers=17,
            lang='en',
            verbose=True,
            device='cpu',
            rescale_with_baseline=False
        )
        
        print(f"\nResults WITHOUT rescaling:")
        print(f"  Precision: mean={P.mean():.4f}, min={P.min():.4f}, max={P.max():.4f}")
        print(f"  Recall:    mean={R.mean():.4f}, min={R.min():.4f}, max={R.max():.4f}")
        print(f"  F1:        mean={F1.mean():.4f}, min={F1.min():.4f}, max={F1.max():.4f}")
        
        print(f"\n  Individual F1 scores: {F1.tolist()}")
        
        # Now try WITH rescaling
        print("\n" + "-"*70)
        print("Testing WITH rescaling...")
        
        P2, R2, F12 = bert_score(
            predictions,
            references,
            model_type='roberta-large',
            num_layers=17,
            lang='en',
            verbose=False,
            device='cpu',
            rescale_with_baseline=True
        )
        
        print(f"\nResults WITH rescaling:")
        print(f"  Precision: mean={P2.mean():.4f}, min={P2.min():.4f}, max={P2.max():.4f}")
        print(f"  Recall:    mean={R2.mean():.4f}, min={R2.min():.4f}, max={R2.max():.4f}")
        print(f"  F1:        mean={F12.mean():.4f}, min={F12.min():.4f}, max={F12.max():.4f}")
        
        print(f"\n  Individual F1 scores: {F12.tolist()}")
        
    else:
        print("❌ No valid prediction-reference pairs found!")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("DEBUGGING COMPLETE")
print("="*70)
