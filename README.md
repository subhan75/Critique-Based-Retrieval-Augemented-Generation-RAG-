# 🎯 Critique-Enhanced RAG System

**Improving Retrieval-Augmented Generation through LLM-based Critique and Refinement**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📖 Overview

This project implements a **Critique-Enhanced RAG system** that uses an LLM-based critique mechanism to evaluate and refine generated answers. Unlike traditional RAG systems, this approach:

1. **Generates** an initial answer using retrieved context
2. **Critiques** the answer for quality, correctness, and hallucinations
3. **Refines** low-quality answers based on critique feedback

### 🎯 Key Results

| Metric | Vanilla RAG | Critique RAG | Improvement |
|--------|-------------|--------------|-------------|
| **Oracle Score** | 5.41/10 | 5.97/10 | **+10.4%** ✅ |
| **ROUGE-L** | 0.1615 | 0.1721 | +6.6% |
| **Hallucination Rate** | N/A | 0.0% | ✅ Zero hallucinations |
| **Refinement Rate** | N/A | 56.0% | - |
| **Quality Score** | N/A | 8.03/10 | - |

**Key Finding**: The critique mechanism achieves **+10.4% improvement** in oracle quality scores, demonstrating that LLM-based critique effectively improves answer quality even when automatic metrics (ROUGE-L, BERTScore) show minimal improvement.

---

## 📁 Project Structure

```
Critique-based-rag/
│
├── 📄 README.md                          # This file
├── 📄 requirements.txt                   # Python dependencies
├── 📄 config.yaml                        # Configuration settings
├── 📄 .env                               # Environment variables (create from .env.example)
├── 📄 .gitignore                         # Git ignore rules
│
├── 📁 src/                               # Source code
│   ├── 📁 data/                          # Data loading and processing
│   │   ├── dataset_loader.py            # SQuAD dataset loader
│   │   ├── chunk_builder.py             # Document chunking
│   │   ├── simple_vector_store.py       # Vector store implementation
│   │   └── embedding_generator.py       # Embedding generation
│   │
│   ├── 📁 models/                        # RAG implementations
│   │   ├── vanilla_rag.py               # Baseline RAG system
│   │   └── critique_rag.py              # Critique-enhanced RAG
│   │
│   ├── 📁 evaluation/                    # Evaluation scripts
│   │   ├── evaluate_vanilla_rag.py      # Evaluate baseline
│   │   ├── evaluate_critique_rag.py     # Evaluate critique RAG
│   │   ├── compute_oracle_scores.py     # Compute oracle quality scores
│   │   └── compare_results.py           # Compare both systems
│   │
│   └── 📁 utils/                         # Utility functions
│       ├── logger.py                     # Logging utilities
│       └── api_utils.py                  # API cost tracking
│
├── 📁 results/                           # Evaluation results
│   ├── comparison_table.txt             # Final comparison table
│   ├── comparison_metrics.json          # Detailed metrics
│   ├── vanilla_rag_evaluation_*.json    # Vanilla RAG results
│   ├── critique_rag_evaluation_*.json   # Critique RAG results
│   └── *_with_oracle_*.json             # Results with oracle scores
│
├── 📁 logs/                              # Log files (auto-generated)
│
└── 📁 tests/                             # Unit tests
```

---

## 🚀 Quick Start

### 1️⃣ **Prerequisites**

- Python 3.8 or higher
- OpenAI API key
- ~500MB disk space for results

### 2️⃣ **Installation**

```bash
# Clone the repository
git clone https://github.com/yourusername/Critique-based-rag.git
cd Critique-based-rag

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3️⃣ **Configuration**

Create a `.env` file in the project root and add your OpenAI API key:

```bash
OPENAI_API_KEY=your-api-key-here
```

Or copy from the example:
```bash
cp .env.example .env
# Then edit .env and add your API key
```

**Optional**: Edit `config.yaml` to customize:
- Model selection (default: `gpt-4o-mini`)
- Quality threshold (default: 9.0)
- Chunk size and overlap
- Number of retrieved chunks

---

## 📊 Running Evaluations

> **📦 Dataset Included**: The SQuAD v1.1 dataset (150 samples) is already included in `src/data/raw/squad_150.json`. **No download required!**

### **TL;DR - Just Run These 3 Commands:**

```bash
python src/evaluation/evaluate_vanilla_rag.py    # ~15 min
python src/evaluation/evaluate_critique_rag.py   # ~20 min
python src/evaluation/compare_results.py         # ~1 min
```

Then check `results/comparison_table.txt` for the results! 🎉

---

### **Complete Evaluation Pipeline** (3 Simple Steps)

Run the complete evaluation pipeline to reproduce the results:

```bash
# Step 1: Evaluate Vanilla RAG (baseline)
python src/evaluation/evaluate_vanilla_rag.py

# Step 2: Evaluate Critique RAG
python src/evaluation/evaluate_critique_rag.py

# Step 3: Compare Results
python src/evaluation/compare_results.py
```

**That's it!** The comparison will show you:
- Oracle Score improvement
- ROUGE-L scores
- BERTScore
- Hallucination rates
- Refinement statistics
- Statistical significance tests

**Time**: ~30-40 minutes total  
**Cost**: ~$1-2 (using GPT-4o-mini)

---

### **Optional: Compute Oracle Scores** (Advanced)

If you want to add oracle quality scores to your evaluation:

```bash
# After running Steps 1 & 2 above, run:
python src/evaluation/compute_oracle_scores.py

# Then compare again to see oracle scores:
python src/evaluation/compare_results.py
```

**Note**: Oracle scoring adds ~15 minutes and ~$0.50 in API costs, but provides valuable human-like quality assessment.

---

## 📈 Understanding the Results

### **1. Comparison Table**

After running evaluations, check `results/comparison_table.txt`:

```
╔══════════════════════════════════════════════════════════════════╗
║           VANILLA RAG vs CRITIQUE RAG COMPARISON                 ║
╠══════════════════════════════════════════════════════════════════╣
║ Metric                    │ Vanilla RAG    │ Critique RAG        ║
╠═══════════════════════════╪════════════════╪═════════════════════╣
║ Oracle Score (mean)       │ 5.41/10        │ 5.97/10             ║
║ Oracle improvement        │ -              │ +10.4%              ║
...
```

### **2. Metrics Explained**

#### **Oracle Score** (0-10) 🎯
- **What**: LLM-based quality evaluation against ground truth
- **Components**:
  - Correctness (0-4): Semantic match with ground truth
  - Completeness (0-3): Coverage of key information
  - Clarity (0-2): Structure and readability
  - Conciseness (0-1): Appropriate length
- **Why**: Provides human-like quality assessment, more nuanced than automatic metrics

#### **ROUGE-L** (0-1)
- **What**: Lexical overlap with ground truth
- **Limitation**: Doesn't capture semantic similarity or paraphrasing

#### **BERTScore** (0-1)
- **What**: Semantic similarity using RoBERTa embeddings
- **Limitation**: May not capture answer quality improvements

#### **Quality Score** (0-10) - Critique RAG Only
- **What**: Internal critique mechanism's assessment
- **Use**: Determines if refinement is needed (threshold: 9.0)

#### **Hallucination Rate** (%) - Critique RAG Only
- **What**: Percentage of answers with detected hallucinations
- **Result**: 0.0% (critique mechanism prevents hallucinations)

#### **Refinement Rate** (%)
- **What**: Percentage of answers that were refined
- **Result**: 56.0% (84 out of 150 answers refined)

---

## 🔬 Evaluation Methodology

### **Dataset**
- **Source**: SQuAD v1.1 validation set
- **Samples**: 150 questions
- **Task**: Extractive question answering

### **Evaluation Process**

1. **Vanilla RAG**: Standard retrieve-then-generate
   - Retrieve top-5 chunks
   - Generate answer using GPT-4o-mini

2. **Critique RAG**: Enhanced with critique and refinement
   - Retrieve top-5 chunks
   - Generate initial answer
   - Critique answer (quality, correctness, hallucinations)
   - Refine if quality score < 9.0
   - Re-critique refined answer

3. **Oracle Scoring**: LLM-based quality evaluation
   - Use GPT-4o-mini to evaluate each answer
   - Score on correctness, completeness, clarity, conciseness
   - Compare against ground truth

4. **Comparison**: Statistical analysis
   - Compute ROUGE-L, BERTScore, Oracle scores
   - Perform paired t-tests
   - Calculate effect sizes (Cohen's d)

---

## 🎓 Key Findings

### **1. Oracle Scores Validate Critique Mechanism** ✅

The **+10.4% improvement** in oracle scores demonstrates that:
- Critique mechanism effectively improves answer quality
- Automatic metrics (ROUGE-L, BERTScore) underestimate improvement
- Human-like evaluation is necessary for comprehensive assessment

### **2. Zero Hallucination Rate** ✅

The critique mechanism successfully:
- Detects potential hallucinations
- Prevents hallucinated content in final answers
- Maintains factual accuracy

### **3. High Refinement Success Rate** ✅

- **95.2%** of refinements improved quality scores
- Average improvement: **+5.80 points**
- Demonstrates effective critique-based refinement

### **4. Metric Limitations**

- ROUGE-L: +6.6% (not statistically significant)
- BERTScore: -0.7% (slight decrease)
- **Conclusion**: Lexical/semantic metrics don't capture quality improvements

---

## �️ Customization

### **Adjust Quality Threshold**

Edit `src/models/critique_rag.py`:

```python
# Lower threshold = more refinements
quality_threshold = 8.0  # Default: 9.0

# Higher threshold = fewer refinements
quality_threshold = 9.5
```

### **Change Models**

Edit `config.yaml`:

```yaml
# Use different models
llm_model: "gpt-4"           # More powerful, more expensive
critic_model: "gpt-4o-mini"  # Fast and cheap for critique
embedding_model: "text-embedding-3-small"
```

### **Adjust Retrieval**

Edit `config.yaml`:

```yaml
# Retrieve more chunks
final_k: 10  # Default: 5

# Larger chunks
chunk_size: 1024  # Default: 512
chunk_overlap: 100  # Default: 50
```

---

## 📝 Output Files

### **Evaluation Results**

- `vanilla_rag_evaluation_*.json`: Vanilla RAG results (150 samples)
- `critique_rag_evaluation_*.json`: Critique RAG results (150 samples)
- `*_with_oracle_*.json`: Results augmented with oracle scores

### **Comparison**

- `comparison_table.txt`: Formatted comparison table
- `comparison_metrics.json`: Detailed metrics in JSON format

### **Logs**

- `logs/evaluate_vanilla_rag.log`: Vanilla RAG evaluation log
- `logs/evaluate_critique_rag.log`: Critique RAG evaluation log

---

## 🧪 Testing

Run unit tests:

```bash
# Run all tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_critique_rag.py
```

---

## 📚 Documentation

For more details, see:
- **Implementation**: Code comments in `src/models/critique_rag.py`
- **Evaluation**: `src/evaluation/` scripts
- **Configuration**: `config.yaml` with inline comments

---

## 🤝 Contributing

This is a research project for CSE 244 Fall 2025. Contributions, issues, and feature requests are welcome!

---

## 📄 License

MIT License - see LICENSE file for details

---

## 👤 Author

**CSE 244 Fall 2025 Final Project**

---

## 🙏 Acknowledgments

- SQuAD dataset: Rajpurkar et al., 2016
- OpenAI API for LLM capabilities
- Course: CSE 244 - Advanced NLP

---

## 📧 Contact

For questions or issues, please open a GitHub issue or contact the author.

---

**Happy Experimenting! 🚀**
