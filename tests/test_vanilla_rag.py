"""
Test script for Vanilla RAG model.
Tests with a small sample of SQuAD questions.
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data.dataset_loader import load_squad
from src.data.chunk_builder import chunk_documents, get_chunk_texts
from src.data.simple_vector_store import build_simple_vector_store
from src.models.vanilla_rag import create_vanilla_rag
from src.utils.logger import setup_logger
from src.utils.api_utils import get_cost_summary

# Setup logging
logger = setup_logger(__name__, log_file="logs/test_vanilla_rag.log")

print("\n" + "="*60)
print("TESTING VANILLA RAG MODEL")
print("="*60 + "\n")

# Step 1: Load dataset
print("1. Loading SQuAD dataset (10 samples)...")
try:
    dataset = load_squad(num_samples=10)
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
        chunk_size=512,
        overlap=50
    )
    chunk_texts = get_chunk_texts(chunks)
    print(f"✅ Created {len(chunk_texts)} chunks")
except Exception as e:
    print(f"❌ Failed to chunk documents: {e}")
    exit(1)

# Step 3: Build vector store
print("\n3. Building ChromaDB vector store...")
print("   NOTE: This will call OpenAI API to embed documents.")
print("   If this hangs, press Ctrl+C and we'll try an alternative approach.")
try:
    # Create chunk IDs and metadata
    chunk_ids = [f"chunk_{i}" for i in range(len(chunk_texts))]
    metadatas = [{"chunk_index": i, "source": "squad"} for i in range(len(chunk_texts))]
    
    print(f"   Using simple numpy-based vector store (ChromaDB alternative)...")
    
    vector_store = build_simple_vector_store(
        chunks=chunk_texts,
        chunk_ids=chunk_ids,
        metadatas=metadatas
    )
    print(f"✅ Vector store built successfully")
    sys.stdout.flush()
except KeyboardInterrupt:
    print("\n\n⚠️  Process interrupted by user")
    print("The embedding process was taking too long.")
    print("\nTroubleshooting:")
    print("  1. Check your internet connection")
    print("  2. Verify OpenAI API key is valid")
    print("  3. Try running: pip install --upgrade chromadb openai")
    exit(1)
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
        model="gpt-4o-mini",
        temperature=0.7,
        max_tokens=500,
        top_k=5
    )
    print(f"✅ Vanilla RAG initialized")
    print(f"   Model: gpt-4o-mini")
    print(f"   Top-K: 5 chunks")
except Exception as e:
    print(f"❌ Failed to initialize RAG: {e}")
    exit(1)

# Step 5: Test with sample queries
print("\n5. Testing with sample queries...")
print("="*60)

# Test with first 3 questions from dataset
test_queries = dataset['queries'][:3]
ground_truth_answers = dataset['answers'][:3]

results = []
for i, (query, gt_answer) in enumerate(zip(test_queries, ground_truth_answers)):
    print(f"\n--- Query {i+1}/3 ---")
    print(f"Question: {query}")
    print(f"Ground Truth: {gt_answer}")
    
    try:
        # Get answer from RAG
        result = rag_model.answer(query, return_details=True)
        
        print(f"\nVanilla RAG Answer: {result['answer']}")
        print(f"\nRetrieval Details:")
        print(f"  - Retrieved {len(result['retrieval']['chunks'])} chunks")
        print(f"  - Top distance: {result['retrieval']['distances'][0]:.4f}")
        
        print(f"\nGeneration Details:")
        print(f"  - Tokens: {result['generation']['usage']['total_tokens']}")
        print(f"  - Input: {result['generation']['usage']['prompt_tokens']}")
        print(f"  - Output: {result['generation']['usage']['completion_tokens']}")
        
        # Store result
        results.append({
            'query': query,
            'ground_truth': gt_answer,
            'predicted_answer': result['answer'],
            'retrieval_distances': result['retrieval']['distances'],
            'usage': result['generation']['usage']
        })
        
        print(f"\n✅ Query {i+1} processed successfully")
        
    except Exception as e:
        print(f"\n❌ Failed to process query {i+1}: {e}")
        results.append({
            'query': query,
            'ground_truth': gt_answer,
            'predicted_answer': None,
            'error': str(e)
        })

print("\n" + "="*60)

# Step 6: Show cost summary
print("\n6. Cost Summary:")
cost = get_cost_summary()
print(f"   Total API calls: {cost['total_calls']}")
print(f"   Total tokens: {cost['total_tokens']:,}")
print(f"   Input tokens: {cost['input_tokens']:,}")
print(f"   Output tokens: {cost['output_tokens']:,}")
print(f"   Total cost: ${cost['total_cost']:.4f}")

# Step 7: Save results
print("\n7. Saving results...")
output_dir = Path("results")
output_dir.mkdir(exist_ok=True)

output_file = output_dir / "vanilla_rag_test_results.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump({
        'model': 'vanilla_rag',
        'test_size': len(test_queries),
        'results': results,
        'cost_summary': cost
    }, f, indent=2, ensure_ascii=False)

print(f"✅ Results saved to: {output_file}")

print("\n" + "="*60)
print("✅ VANILLA RAG TEST COMPLETE!")
print("="*60)
print(f"\nResults saved to: {output_file.absolute()}")
print("\nNext steps:")
print("  1. Review results in results/vanilla_rag_test_results.json")
print("  2. Run full evaluation with 150 samples: python evaluate_vanilla_rag.py")
print("  3. Implement Critique-Enhanced RAG\n")

# Also print summary to console
print("\nTest Summary:")
print(f"  Queries tested: {len(test_queries)}")
print(f"  Successful: {sum(1 for r in results if r.get('predicted_answer'))}")
print(f"  Total cost: ${cost['total_cost']:.4f}\n")
