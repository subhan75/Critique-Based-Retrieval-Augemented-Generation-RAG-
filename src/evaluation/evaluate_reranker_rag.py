"""
Full evaluation script for Reranker RAG model.
"""
import sys
from pathlib import Path
import json
from tqdm import tqdm
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.data.dataset_loader import load_squad
from src.data.chunk_builder import chunk_documents, get_chunk_texts
from src.data.simple_vector_store import build_simple_vector_store
from src.models.reranker_rag import create_reranker_rag
from src.utils.logger import setup_logger
from src.utils.api_utils import get_cost_summary

# Setup logging
logger = setup_logger(__name__, log_file="logs/evaluate_reranker_rag.log")

print("\n" + "="*60)
print("RERANKER RAG - FULL EVALUATION")
print("="*60 + "\n")

# Configuration
NUM_SAMPLES = 150
INITIAL_K = 20  # Fetch 20 candidates
TOP_K = 5       # Keep top 5 after reranking
MODEL = "gpt-4o-mini"

# 1. Load Dataset
print(f"1. Loading SQuAD dataset ({NUM_SAMPLES} samples)...")
dataset = load_squad(num_samples=NUM_SAMPLES)

# 2. Process Data
print("2. Chunking documents...")
chunks = chunk_documents(dataset['documents'], chunk_size=512, overlap=50)
chunk_texts = get_chunk_texts(chunks)

# 3. Build Vector Store
print("3. Building vector store...")
chunk_ids = [f"chunk_{i}" for i in range(len(chunk_texts))]
metadatas = [{"chunk_index": i, "source": "squad"} for i in range(len(chunk_texts))]

vector_store = build_simple_vector_store(
    chunks=chunk_texts,
    chunk_ids=chunk_ids,
    metadatas=metadatas
)

# 4. Initialize Reranker RAG
print("\n4. Initializing Reranker RAG...")
rag_model = create_reranker_rag(
    vector_store=vector_store,
    model=MODEL,
    initial_k=INITIAL_K,
    top_k=TOP_K
)

# 5. Run Evaluation
print(f"\n5. Evaluating on {NUM_SAMPLES} queries...")
results = []

for i, (query, gt_answer) in enumerate(tqdm(zip(dataset['queries'], dataset['answers']), total=len(dataset['queries']))):
    try:
        result = rag_model.answer(query, return_details=True)
        
        results.append({
            'query_id': i,
            'query': query,
            'ground_truth': gt_answer,
            'predicted_answer': result['answer'],
            'retrieval': {
                'candidates_fetched': result['retrieval']['all_candidates'],
                'final_chunks': len(result['retrieval']['chunks'])
            }
        })
    except Exception as e:
        logger.error(f"Error on query {i}: {e}")
        results.append({'query_id': i, 'error': str(e)})

# 6. Save Results
output_dir = Path("results")
output_dir.mkdir(exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = output_dir / f"reranker_rag_evaluation_{timestamp}.json"

cost = get_cost_summary()

data = {
    'metadata': {
        'model': 'reranker_rag', 
        'initial_k': INITIAL_K,
        'top_k': TOP_K
    },
    'results': results,
    'cost_summary': cost
}

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2)

print(f"\n✅ Results saved to: {output_file}")
print(f"💰 Total Cost: ${cost['total_cost']:.4f}")