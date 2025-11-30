"""
Full evaluation script for Critique-Enhanced RAG.

Runs Critique RAG on 150 SQuAD validation samples and saves results.
"""
import sys
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
project_root = Path(__file__).resolve().parents[2]  # Go up 2 levels to project root
sys.path.insert(0, str(project_root))

from src.data.dataset_loader import load_squad
from src.data.chunk_builder import chunk_documents, get_chunk_texts
from src.data.simple_vector_store import build_simple_vector_store
from src.models.critique_rag import create_critique_rag
from src.utils.logger import setup_logger
from src.utils.api_utils import get_cost_summary

# Setup logging
logger = setup_logger(__name__, log_file="logs/evaluate_critique_rag.log")

print("\n" + "="*60)
print("CRITIQUE RAG - FULL EVALUATION (150 SAMPLES)")
print("="*60 + "\n")

# Configuration
NUM_SAMPLES = 150
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
TOP_K = 5
MODEL = "gpt-4o-mini"
TEMPERATURE = 0.0
MAX_TOKENS = 500
QUALITY_THRESHOLD = 8.0

print("Configuration:")
print(f"  Dataset: SQuAD v1.1")
print(f"  Samples: {NUM_SAMPLES}")
print(f"  Chunk size: {CHUNK_SIZE}")
print(f"  Chunk overlap: {CHUNK_OVERLAP}")
print(f"  Top-K retrieval: {TOP_K}")
print(f"  Model: {MODEL}")
print(f"  Temperature: {TEMPERATURE}")
print(f"  Max tokens: {MAX_TOKENS}")
print(f"  Quality threshold: {QUALITY_THRESHOLD}")

# Step 1: Load dataset
print(f"\n1. Loading SQuAD dataset ({NUM_SAMPLES} samples)...")
try:
    dataset = load_squad(num_samples=NUM_SAMPLES)
    print(f"✅ Loaded {len(dataset['queries'])} questions")
    print(f"   Documents: {len(dataset['documents'])}")
except Exception as e:
    print(f"❌ Failed to load dataset: {e}")
    exit(1)

# Step 2: Chunk documents
print("\n2. Chunking documents...")
try:
    chunks = chunk_documents(
        documents=dataset['contexts'],
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP
    )
    chunk_texts = get_chunk_texts(chunks)
    print(f"✅ Created {len(chunk_texts)} chunks")
except Exception as e:
    print(f"❌ Failed to chunk documents: {e}")
    exit(1)

# Step 3: Build vector store
print("\n3. Building vector store...")
print("   (Using simple numpy-based store - more reliable on Windows)")
print("   (This will embed all chunks using OpenAI API)")
try:
    chunk_ids = [f"chunk_{i}" for i in range(len(chunk_texts))]
    metadatas = [{"chunk_index": i, "source": "squad"} for i in range(len(chunk_texts))]
    
    vector_store = build_simple_vector_store(
        chunks=chunk_texts,
        chunk_ids=chunk_ids,
        metadatas=metadatas
    )
    print(f"✅ Vector store built successfully")
    
    # Show embedding cost so far
    cost = get_cost_summary()
    print(f"   Embedding cost: ${cost['total_cost']:.4f}")
    
except Exception as e:
    print(f"❌ Failed to build vector store: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Step 4: Initialize Critique RAG
print("\n4. Initializing Critique RAG...")
try:
    critique_rag = create_critique_rag(
        vector_store=vector_store,
        model=MODEL,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        quality_threshold=QUALITY_THRESHOLD
    )
    print(f"✅ Critique RAG initialized")
except Exception as e:
    print(f"❌ Failed to initialize Critique RAG: {e}")
    exit(1)

# Step 5: Run evaluation
print(f"\n5. Running evaluation on {NUM_SAMPLES} queries...")
print("   (This will take 2-3 hours - progress will be shown)")
print("="*60 + "\n")

results = []
errors = []

for i in range(len(dataset['queries'])):
    question = dataset['queries'][i]
    ground_truth = dataset['answers'][i]
    
    # Progress indicator
    if (i + 1) % 10 == 0:
        print(f"Progress: {i+1}/{NUM_SAMPLES} queries processed...")
        cost = get_cost_summary()
        print(f"  Current cost: ${cost['total_cost']:.4f}")
    
    try:
        # Query Critique RAG
        result = critique_rag.query(question)
        
        # Store result
        results.append({
            'query_id': i,
            'question': question,
            'ground_truth': ground_truth,
            'answer': result['answer'],
            'quality_score': result['quality_score'],
            'has_hallucinations': result['has_hallucinations'],
            'hallucinations': result['hallucinations'],
            'is_high_quality': result['is_high_quality'],
            'is_correct': result['is_correct'],
            'reasoning': result['reasoning'],
            'was_refined': result['was_refined'],
            'initial_score': result.get('initial_score'),
            'score_improvement': result.get('score_improvement'),
            'num_chunks': len(result['chunks'])
        })
        
    except Exception as e:
        print(f"\n❌ Error on query {i+1}: {e}")
        errors.append({
            'query_id': i,
            'question': question,
            'error': str(e)
        })

print("\n" + "="*60)
print(f"✅ Evaluation complete!")
print(f"   Successful: {len(results)}/{NUM_SAMPLES}")
print(f"   Errors: {len(errors)}")

# Step 6: Save results
print("\n6. Saving results...")
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = results_dir / f"critique_rag_evaluation_{timestamp}.json"

output_data = {
    'metadata': {
        'timestamp': timestamp,
        'num_samples': NUM_SAMPLES,
        'model': MODEL,
        'quality_threshold': QUALITY_THRESHOLD,
        'chunk_size': CHUNK_SIZE,
        'chunk_overlap': CHUNK_OVERLAP,
        'top_k': TOP_K
    },
    'results': results,
    'errors': errors,
    'summary': {
        'total_queries': NUM_SAMPLES,
        'successful': len(results),
        'failed': len(errors),
        'high_quality_answers': sum(1 for r in results if r['is_high_quality']),
        'refined_answers': sum(1 for r in results if r['was_refined']),
        'hallucinations_detected': sum(1 for r in results if r['has_hallucinations']),
        'avg_quality_score': sum(r['quality_score'] for r in results) / len(results) if results else 0,
        'refinement_rate': sum(1 for r in results if r['was_refined']) / len(results) if results else 0,
        'hallucination_rate': sum(1 for r in results if r['has_hallucinations']) / len(results) if results else 0
    }
}

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)

print(f"✅ Results saved to: {output_file}")

# Step 7: Display summary
print("\n7. Evaluation Summary:")
print(f"   Total queries: {NUM_SAMPLES}")
print(f"   Successful: {len(results)}")
print(f"   High quality answers (≥8.0): {output_data['summary']['high_quality_answers']} ({output_data['summary']['high_quality_answers']/len(results)*100:.1f}%)")
print(f"   Answers refined: {output_data['summary']['refined_answers']} ({output_data['summary']['refinement_rate']*100:.1f}%)")
print(f"   Hallucinations detected: {output_data['summary']['hallucinations_detected']} ({output_data['summary']['hallucination_rate']*100:.1f}%)")
print(f"   Average quality score: {output_data['summary']['avg_quality_score']:.2f}/10")

# Step 8: Cost summary
print("\n8. Cost Summary:")
cost = get_cost_summary()
print(f"   Total API calls: {cost['total_calls']}")
print(f"   Total tokens: {cost['total_tokens']:,}")
print(f"   Input tokens: {cost['input_tokens']:,}")
print(f"   Output tokens: {cost['output_tokens']:,}")
print(f"   Total cost: ${cost['total_cost']:.4f}")
print(f"   Cost per query: ${cost['total_cost']/NUM_SAMPLES:.4f}")

print("\n" + "="*60)
print("✅ CRITIQUE RAG EVALUATION COMPLETE!")
print("="*60)
print(f"\nResults saved to: {output_file.absolute()}")
print("\nNext steps:")
print("  1. Review results in the JSON file")
print("  2. Run comparison: python compare_results.py")
print("  3. Analyze when refinement helps vs hurts")
print("  4. Write research report\n")
