"""
Test script for Critique-Enhanced RAG.

Tests the critique and refinement functionality on a small sample.
"""
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data.dataset_loader import load_squad
from src.data.chunk_builder import chunk_documents, get_chunk_texts
from src.data.simple_vector_store import build_simple_vector_store
from src.models.critique_rag import create_critique_rag
from src.utils.logger import setup_logger
from src.utils.api_utils import get_cost_summary

# Setup logging
logger = setup_logger(__name__, log_file="logs/test_critique_rag.log")

print("\n" + "="*60)
print("TESTING CRITIQUE-ENHANCED RAG")
print("="*60 + "\n")

# Step 1: Load dataset (small sample)
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
        documents=dataset['contexts'],
        chunk_size=512,
        overlap=50
    )
    chunk_texts = get_chunk_texts(chunks)
    print(f"✅ Created {len(chunk_texts)} chunks")
except Exception as e:
    print(f"❌ Failed to chunk documents: {e}")
    exit(1)

# Step 3: Build vector store
print("\n3. Building vector store...")
print("   (Using simple numpy-based vector store)")
try:
    chunk_ids = [f"chunk_{i}" for i in range(len(chunk_texts))]
    metadatas = [{"chunk_index": i, "source": "squad"} for i in range(len(chunk_texts))]
    
    vector_store = build_simple_vector_store(
        chunks=chunk_texts,
        chunk_ids=chunk_ids,
        metadatas=metadatas
    )
    print(f"✅ Vector store built successfully")
except Exception as e:
    print(f"❌ Failed to build vector store: {e}")
    exit(1)

# Step 4: Initialize Critique RAG
print("\n4. Initializing Critique RAG...")
try:
    critique_rag = create_critique_rag(
        vector_store=vector_store,
        model="gpt-4o-mini",
        temperature=0.0,
        max_tokens=500,
        quality_threshold=8.0
    )
    print(f"✅ Critique RAG initialized")
    print(f"   Model: gpt-4o-mini")
    print(f"   Quality threshold: 8.0")
    print(f"   Top-K: 5 chunks")
except Exception as e:
    print(f"❌ Failed to initialize Critique RAG: {e}")
    exit(1)

# Step 5: Test with sample queries
print("\n5. Testing with sample queries...")
print("="*60 + "\n")

# Use first 3 questions
test_queries = [
    {
        'question': dataset['queries'][0],
        'ground_truth': dataset['answers'][0]
    },
    {
        'question': dataset['queries'][1],
        'ground_truth': dataset['answers'][1]
    },
    {
        'question': dataset['queries'][2],
        'ground_truth': dataset['answers'][2]
    }
]

results = []

for i, query_data in enumerate(test_queries, 1):
    question = query_data['question']
    ground_truth = query_data['ground_truth']
    
    print(f"--- Query {i}/3 ---")
    print(f"Question: {question}")
    print(f"Ground Truth: {ground_truth}")
    print()
    
    try:
        # Query Critique RAG
        result = critique_rag.query(question)
        
        # Display results
        print(f"Critique RAG Answer: {result['answer']}")
        print()
        print(f"Quality Assessment:")
        print(f"  - Quality Score: {result['quality_score']}/10")
        print(f"  - Is Correct: {result['is_correct']}")
        print(f"  - Has Hallucinations: {result['has_hallucinations']}")
        if result['hallucinations']:
            print(f"  - Hallucinations: {result['hallucinations']}")
        print(f"  - Is High Quality: {result['is_high_quality']}")
        print(f"  - Was Refined: {result['was_refined']}")
        
        if result['was_refined']:
            print(f"  - Initial Score: {result['initial_score']}/10")
            print(f"  - Score Improvement: +{result['score_improvement']:.1f}")
        
        print(f"  - Reasoning: {result['reasoning']}")
        print()
        
        print(f"Retrieval Details:")
        print(f"  - Retrieved {len(result['chunks'])} chunks")
        print()
        
        # Store result
        results.append({
            'question': question,
            'ground_truth': ground_truth,
            'answer': result['answer'],
            'quality_score': result['quality_score'],
            'has_hallucinations': result['has_hallucinations'],
            'was_refined': result['was_refined'],
            'is_high_quality': result['is_high_quality']
        })
        
        print(f"✅ Query {i} processed successfully\n")
        
    except Exception as e:
        print(f"❌ Query {i} failed: {e}\n")
        import traceback
        traceback.print_exc()

print("="*60 + "\n")

# Step 6: Summary
print("6. Test Summary:")
print(f"   Queries tested: {len(results)}")
print(f"   Successful: {len(results)}")
if results:
    print(f"   High quality answers: {sum(1 for r in results if r['is_high_quality'])}")
    print(f"   Answers refined: {sum(1 for r in results if r['was_refined'])}")
    print(f"   Hallucinations detected: {sum(1 for r in results if r['has_hallucinations'])}")
    print(f"   Average quality score: {sum(r['quality_score'] for r in results) / len(results):.1f}/10")
else:
    print("   No successful queries to summarize")

# Step 7: Cost summary
print("\n7. Cost Summary:")
cost = get_cost_summary()
print(f"   Total API calls: {cost['total_calls']}")
print(f"   Total tokens: {cost['total_tokens']:,}")
print(f"   Input tokens: {cost['input_tokens']:,}")
print(f"   Output tokens: {cost['output_tokens']:,}")
print(f"   Total cost: ${cost['total_cost']:.4f}")

print("\n" + "="*60)
print("✅ CRITIQUE RAG TEST COMPLETE!")
print("="*60)
print("\nNext steps:")
print("  1. Review the quality scores and refinement behavior")
print("  2. If results look good, run full evaluation: python evaluate_critique_rag.py")
print("  3. Compare with Vanilla RAG results\n")
