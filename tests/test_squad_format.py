"""
Quick test to load SQuAD v1.1 and verify structure.
First run: python download_squad.py
"""
import json
from pathlib import Path
from src.data.dataset_loader import load_squad

print("\n" + "="*60)
print("Loading SQuAD v1.1 Dataset")
print("="*60 + "\n")

# Check if dataset exists
json_path = Path("data/raw/squad_10.json")
if not json_path.exists():
    print("⚠️  Dataset not found!")
    print("Please run: python download_squad.py")
    print("This will download the dataset from Hugging Face.")
    exit(1)

dataset = load_squad(num_samples=10)

print(f"\nDataset structure:")
print(f"  - Queries: {len(dataset['queries'])}")
print(f"  - Answers: {len(dataset['answers'])}")
print(f"  - Contexts: {len(dataset['contexts'])}")
print(f"  - Documents: {len(dataset['documents'])}")

print(f"\n" + "="*60)
print("Sample Entry:")
print("="*60)
print(f"\nQuery: {dataset['queries'][0]}")
print(f"\nAnswer: {dataset['answers'][0]}")
print(f"\nContext (first 200 chars): {dataset['contexts'][0][:200]}...")
print(f"\nDocument (first 200 chars): {dataset['documents'][0][:200]}...")

# Check if JSON was saved
if json_path.exists():
    print(f"\n" + "="*60)
    print(f"✓ JSON loaded from: {json_path}")
    print("="*60)
    
    # Show first entry from JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"\nJSON Structure:")
    print(f"  Total entries: {len(data)}")
    print(f"\nFirst entry:")
    print(json.dumps(data[0], indent=2))
else:
    print(f"\n✗ JSON not found at: {json_path}")

print(f"\n" + "="*60)
print("✓ Dataset loaded successfully!")
print("="*60)
