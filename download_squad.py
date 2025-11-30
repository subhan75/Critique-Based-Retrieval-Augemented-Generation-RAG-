"""
Download SQuAD v1.1 dataset from Hugging Face.
Run this once to download and cache the dataset locally.
"""
from datasets import load_dataset
import json
from pathlib import Path


def download_squad(num_samples=150):
    """
    Download SQuAD v1.1 dataset from Hugging Face.
    
    Args:
        num_samples: Number of samples to download (default: 150)
    """
    print("="*60)
    print(f"DOWNLOADING SQuAD v1.1 ({num_samples} samples)")
    print("="*60)
    
    # Create output directory
    output_dir = Path("src/data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load SQuAD v1.1 from validation set
    print(f"\nLoading {num_samples} examples from SQuAD v1.1 validation set...")
    print("This may take 1-2 minutes on first download...\n")
    
    dataset = load_dataset("squad", split=f"validation[:{num_samples}]")
    
    print(f"Downloaded {len(dataset)} examples. Processing...\n")
    
    # Create the dataset in our format
    samples = []
    
    for idx, item in enumerate(dataset):
        samples.append({
            "id": str(idx),
            "question": item['question'],
            "answer": item['answers']['text'][0],  # Take first answer
            "context": item['context'],
            "title": item.get('title', 'Unknown')  # Document title if available
        })
    
    print(f"{'='*60}")
    print(f"Processing complete!")
    print(f"  Total samples: {len(samples)}")
    print(f"{'='*60}")
    
    # Save to JSON
    output_path = output_dir / f"squad_{num_samples}.json"
    
    print(f"\nSaving to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(samples, indent=2, fp=f)
    
    # Show file info
    file_size_kb = output_path.stat().st_size / 1024
    print(f"✓ Successfully saved!")
    print(f"  Location: {output_path.absolute()}")
    print(f"  Samples: {len(samples)}")
    print(f"  File size: {file_size_kb:.2f} KB")
    
    # Show sample entry
    if samples:
        print(f"\n{'='*60}")
        print("SAMPLE ENTRY:")
        print(f"{'='*60}")
        sample = samples[0]
        print(f"ID: {sample['id']}")
        print(f"Question: {sample['question']}")
        print(f"Answer: {sample['answer']}")
        print(f"Context (first 200 chars): {sample['context'][:200]}...")
        print(f"Title: {sample['title']}")
    
    print(f"\n{'='*60}")
    print("DOWNLOAD COMPLETE! ✓")
    print(f"{'='*60}\n")
    
    return samples


if __name__ == "__main__":
    # Download 150 samples for main experiments
    print("Downloading main dataset (150 samples)...\n")
    samples = download_squad(num_samples=150)
    
    # Also download 10 samples for quick testing
    print("\n\nDownloading test set (10 samples)...\n")
    test_samples = download_squad(num_samples=10)
