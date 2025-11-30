"""
Test script to verify BERTScore is working.
Run this AFTER fixing NumPy to confirm BERTScore is available.
"""

print("="*60)
print("TESTING BERTSCORE AVAILABILITY")
print("="*60 + "\n")

# Test 1: Check NumPy version
print("1. Checking NumPy version...")
try:
    import numpy as np
    print(f"   ✅ NumPy version: {np.__version__}")
    
    if np.__version__.startswith('2.'):
        print(f"   ⚠️  WARNING: NumPy 2.x detected!")
        print(f"   BERTScore requires NumPy 1.x")
        print(f"   Run: pip install 'numpy<2'")
    else:
        print(f"   ✅ NumPy 1.x - Compatible with BERTScore")
except Exception as e:
    print(f"   ❌ NumPy error: {e}")

print()

# Test 1.5: Check PyTorch
print("1.5. Checking PyTorch...")
try:
    import torch
    print(f"   ✅ PyTorch version: {torch.__version__}")
    print(f"   Device: {torch.device('cpu')}")
    
    # Test basic tensor operation
    x = torch.tensor([1.0, 2.0, 3.0])
    print(f"   ✅ PyTorch working correctly")
except Exception as e:
    print(f"   ❌ PyTorch error: {e}")
    print(f"   This might be the DLL issue!")
    print(f"   Fix: Run fix_bertscore_dll.bat")

print()

# Test 2: Try importing BERTScore
print("2. Testing BERTScore import...")
try:
    from bert_score import score as bert_score
    print(f"   ✅ BERTScore imported successfully!")
    bertscore_available = True
except ImportError as e:
    print(f"   ❌ BERTScore not installed")
    print(f"   Run: pip install bert-score")
    bertscore_available = False
except AttributeError as e:
    print(f"   ❌ BERTScore import failed (NumPy compatibility issue)")
    print(f"   Error: {str(e)[:100]}")
    print(f"   Fix: pip install 'numpy<2'")
    bertscore_available = False
except Exception as e:
    print(f"   ❌ Unexpected error: {str(e)[:100]}")
    bertscore_available = False

print()

# Test 3: Run a simple BERTScore test
if bertscore_available:
    print("3. Running BERTScore test...")
    try:
        # Simple test with 2 sentences
        predictions = ["The cat sat on the mat"]
        references = ["A cat was sitting on a mat"]
        
        print("   Computing BERTScore for test sentences...")
        P, R, F1 = bert_score(
            predictions,
            references,
            lang='en',
            verbose=False,
            device='cpu'
        )
        
        print(f"   ✅ BERTScore test successful!")
        print(f"   Precision: {P.mean():.4f}")
        print(f"   Recall: {R.mean():.4f}")
        print(f"   F1: {F1.mean():.4f}")
        
    except Exception as e:
        print(f"   ❌ BERTScore test failed: {str(e)[:100]}")
        bertscore_available = False
else:
    print("3. Skipping BERTScore test (import failed)")

print()
print("="*60)

# Summary
if bertscore_available:
    print("✅ BERTSCORE IS READY!")
    print("="*60)
    print("\nYou can now run: python compare_results.py")
    print("BERTScore will be computed successfully.\n")
else:
    print("❌ BERTSCORE NOT AVAILABLE")
    print("="*60)
    print("\nTo fix:")
    print("1. Run: pip uninstall -y numpy")
    print("2. Run: pip install 'numpy<2'")
    print("3. Run: pip install --force-reinstall matplotlib")
    print("4. Run this test again: python test_bertscore.py\n")
