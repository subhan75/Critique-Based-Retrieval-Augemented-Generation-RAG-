"""
Full evaluation script for Vanilla RAG model on 150 SQuAD samples.
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import json
from tqdm import tqdm
from datetime import datetime

# Load environment variables
load_dotenv()

# Add project root to path
project_root = Path(__file__).resolve().parents[2]  # Go up 2 levels to project root
sys.path.insert(0, str(project_root))

from src.data.dataset_loader import load_squad
from src.data.chunk_builder import chunk_documents, get_chunk_texts
from src.data.simple_vector_store import build_simple_vector_store
from src.models.vanilla_rag import create_vanilla_rag
from src.utils.logger import setup_logger
from src.utils.api_utils import get_cost_summary

# Setup logging
logger = setup_logger(__name__, log_file="logs/evaluate_vanilla_rag.log")

print("\n" + "="*60)
print("VANILLA RAG - FULL EVALUATION (150 SAMPLES)")
print("="*60 + "\n")

# Configuration
NUM_SAMPLES = 150
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
TOP_K = 5
MODEL = "gpt-4o-mini"
TEMPERATURE = 0.7
MAX_TOKENS = 500

print("Configuration:")
print(f"  Dataset: SQuAD v1.1")
print(f"  Samples: {NUM_SAMPLES}")
print(f"  Chunk size: {CHUNK_SIZE}")
print(f"  Chunk overlap: {CHUNK_OVERLAP}")
print(f"  Top-K retrieval: {TOP_K}")
print(f"  Model: {MODEL}")
print(f"  Temperature: {TEMPERATURE}")
print(f"  Max tokens: {MAX_TOKENS}")

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
        dataset['documents'],
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
    # Create chunk IDs and metadata
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

# Step 4: Initialize Vanilla RAG
print("\n4. Initializing Vanilla RAG model...")
try:
    rag_model = create_vanilla_rag(
        vector_store=vector_store,
        model=MODEL,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        top_k=TOP_K
    )
    print(f"✅ Vanilla RAG initialized")
except Exception as e:
    print(f"❌ Failed to initialize RAG: {e}")
    exit(1)

# Step 5: Evaluate on all queries
print(f"\n5. Evaluating on {NUM_SAMPLES} queries...")
print("   (This may take 10-15 minutes)")
print("="*60)

results = []
queries = dataset['queries']
ground_truths = dataset['answers']

# Use tqdm for progress bar
for i, (query, gt_answer) in enumerate(tqdm(
    zip(queries, ground_truths),
    total=len(queries),
    desc="Processing queries"
)):
    try:
        # Get answer from RAG
        result = rag_model.answer(query, return_details=True)
        
        # Store result
        results.append({
            'query_id': i,
            'query': query,
            'ground_truth': gt_answer,
            'predicted_answer': result['answer'],
            'retrieval': {
                'top_k': TOP_K,
                'distances': result['retrieval']['distances'][:3],  # Store top 3 distances
                'num_chunks': len(result['retrieval']['chunks'])
            },
            'generation': {
                'model': MODEL,
                'usage': result['generation']['usage']
            }
        })
        
    except Exception as e:
        logger.error(f"Failed to process query {i}: {e}")
        results.append({
            'query_id': i,
            'query': query,
            'ground_truth': gt_answer,
            'predicted_answer': None,
            'error': str(e)
        })

print("\n" + "="*60)

# Step 6: Calculate statistics
print("\n6. Calculating statistics...")
successful = sum(1 for r in results if r.get('predicted_answer') is not None)
failed = len(results) - successful

print(f"   Total queries: {len(results)}")
print(f"   Successful: {successful}")
print(f"   Failed: {failed}")
print(f"   Success rate: {successful/len(results)*100:.1f}%")

# Step 7: Cost summary
print("\n7. Cost Summary:")
cost = get_cost_summary()
print(f"   Total API calls: {cost['total_calls']}")
print(f"   Total tokens: {cost['total_tokens']:,}")
print(f"   Input tokens: {cost['input_tokens']:,}")
print(f"   Output tokens: {cost['output_tokens']:,}")
print(f"   Total cost: ${cost['total_cost']:.4f}")

# Breakdown
embedding_cost = cost['total_cost'] * 0.1  # Rough estimate (embeddings are cheap)
generation_cost = cost['total_cost'] * 0.9
print(f"\n   Estimated breakdown:")
print(f"   - Embedding cost: ~${embedding_cost:.4f}")
print(f"   - Generation cost: ~${generation_cost:.4f}")

# Step 8: Save results
print("\n8. Saving results...")
output_dir = Path("results")
output_dir.mkdir(exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = output_dir / f"vanilla_rag_evaluation_{timestamp}.json"

evaluation_data = {
    'metadata': {
        'model': 'vanilla_rag',
        'dataset': 'squad_v1.1',
        'num_samples': NUM_SAMPLES,
        'timestamp': timestamp,
        'configuration': {
            'chunk_size': CHUNK_SIZE,
            'chunk_overlap': CHUNK_OVERLAP,
            'top_k': TOP_K,
            'model': MODEL,
            'temperature': TEMPERATURE,
            'max_tokens': MAX_TOKENS
        }
    },
    'statistics': {
        'total_queries': len(results),
        'successful': successful,
        'failed': failed,
        'success_rate': successful/len(results)*100
    },
    'cost_summary': cost,
    'results': results
}

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(evaluation_data, f, indent=2, ensure_ascii=False)

print(f"✅ Results saved to: {output_file}")

# Also save a summary file
summary_file = output_dir / "vanilla_rag_summary.txt"
with open(summary_file, 'w', encoding='utf-8') as f:
    f.write("VANILLA RAG EVALUATION SUMMARY\n")
    f.write("="*60 + "\n\n")
    f.write(f"Dataset: SQuAD v1.1 ({NUM_SAMPLES} samples)\n")
    f.write(f"Model: {MODEL}\n")
    f.write(f"Timestamp: {timestamp}\n\n")
    f.write("RESULTS:\n")
    f.write(f"  Total queries: {len(results)}\n")
    f.write(f"  Successful: {successful}\n")
    f.write(f"  Failed: {failed}\n")
    f.write(f"  Success rate: {successful/len(results)*100:.1f}%\n\n")
    f.write("COST:\n")
    f.write(f"  Total cost: ${cost['total_cost']:.4f}\n")
    f.write(f"  Total tokens: {cost['total_tokens']:,}\n")
    f.write(f"  Input tokens: {cost['input_tokens']:,}\n")
    f.write(f"  Output tokens: {cost['output_tokens']:,}\n")

print(f"✅ Summary saved to: {summary_file}")

print("\n" + "="*60)
print("✅ VANILLA RAG EVALUATION COMPLETE!")
print("="*60)
print("\nNext steps:")
print("  1. Review results in results/ directory")
print("  2. Implement Critique-Enhanced RAG")
print("  3. Compare both models\n")
