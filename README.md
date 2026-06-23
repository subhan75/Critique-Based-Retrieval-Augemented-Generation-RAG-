# Critique-Enhanced RAG

**Improving Retrieval-Augmented Generation through LLM-based Critique and Refinement**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> CSE 244 (Advanced NLP), Fall 2025 — Subhan Shaikh, Sohail Syed
> Full report and video presentation linked at the bottom of this README.

---

## Overview

Standard RAG systems generate answers with no quality control: they retrieve context, generate, and return the result without verifying correctness, completeness, or grounding. Most research addresses this by optimizing **retrieval** (e.g., reranking), which does nothing for errors introduced at the **generation** stage.

**Critique-Enhanced RAG** keeps retrieval simple and instead adds a post-generation quality-control loop:

1. **Generate** an initial answer from the retrieved context.
2. **Critique** that answer across four dimensions (correctness, completeness, clarity, conciseness) and flag unsupported claims.
3. **Refine** the answer only if its quality score falls below a threshold — and accept the refinement only if it actually scores higher, preventing degradation.

The central finding: when retrieval is already adequate, **investing compute in generation-stage critique beats investing it in retrieval reranking** — both in quality and in cost-efficiency.

---

## Key Results

Evaluated on 150 questions from the SQuAD v1.1 validation set. All three systems use `gpt-4o-mini` for generation (temperature 0).

| Metric | Vanilla RAG | Reranker RAG | Critique RAG |
|---|---|---|---|
| Oracle Score (0–10) | 5.21 | 4.98 | **6.62** (+27.0% vs Vanilla) |
| ROUGE-L | 0.1522 | 0.1495 | **0.1722** (+13.1% vs Vanilla) |
| Cost / query | $0.0001 | $0.0006 (6×) | $0.0002 (2×) |
| Avg latency | 2.0s | 3.0s | 5.0s+ |

**Statistical significance:** paired t-tests confirm Critique RAG outperforms both baselines (vs. Vanilla: t = 4.83, p < 0.001, Cohen's d = 0.89; vs. Reranker: t = 5.91, p < 0.001, d = 1.12). The Vanilla–Reranker gap is *not* significant (p = 0.18) — i.e., reranking cost 6× more for no statistically meaningful gain.

**Refinement behavior:** refinement triggered on 84/150 answers (56.0%); of those, 95.2% improved, with an average gain of +5.80 quality points. The validation step (`accept only if s₁ > s₀`) rejected the rest, preventing degradation.

**Hallucinations:** the critic flags claims unsupported by retrieved context, and measured a 0.0% hallucination rate on this 150-question set. **Note:** this is a self-reported figure from the same model family used for generation and has not been independently verified — see [Limitations](#limitations).

> On automatic metrics alone the gains look modest (ROUGE-L +13.1%, BERTScore roughly flat). The oracle (LLM-as-judge) evaluation surfaces a larger, statistically significant quality difference, which is the paper's main point: lexical/semantic overlap metrics under-measure generation-quality improvements.

---

## How It Works

### Baselines
- **Vanilla RAG** — dense retrieval (`text-embedding-3-small`, 1536-dim, exact cosine similarity over a NumPy vector store), top-5 chunks, then generate.
- **Reranker RAG** — retrieve 20 candidates, have the LLM score each 0–10 pointwise, keep the top-5, then generate. Represents the retrieval-optimization approach.

### Critique-Enhanced RAG
Uses **identical retrieval to Vanilla** (top-5), isolating the effect of the generation-stage additions:
- **Multi-dimensional critique** — correctness (0–4), completeness (0–3), clarity (0–2), conciseness (0–1); sum = quality score `s ∈ [0,10]`, plus a hallucination flag.
- **Validated refinement** — if `s₀ < 9.0`, refine using the critique feedback, re-score to get `s₁`, and accept the refined answer only if `s₁ > s₀`. Capped at one refinement iteration.

---

## Project Structure

```
Critique-based-rag/
├── README.md
├── requirements.txt
├── config.yaml                      # model, threshold, chunking, retrieval settings
├── .env                             # OPENAI_API_KEY (create from .env.example)
│
├── src/
│   ├── data/                        # SQuAD loader, chunking, vector store, embeddings
│   ├── models/                      # vanilla_rag.py, reranker_rag.py, critique_rag.py
│   ├── evaluation/                  # per-system evaluators + compare_all.py + oracle scoring
│   └── utils/                       # logging, API cost tracking
│
├── results/                         # comparison tables + per-system JSON results
├── logs/
└── tests/
```

---

## Quick Start

**Prerequisites:** Python 3.8+, an OpenAI API key, ~500MB disk for results.

```bash
# Clone and enter
git clone https://github.com/subhan75/Critique-Based-Retrieval-Augemented-Generation-RAG-.git
cd Critique-Based-Retrieval-Augemented-Generation-RAG-

# Virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Dependencies
pip install -r requirements.txt

# API key
cp .env.example .env             # then edit .env and add OPENAI_API_KEY
```

Optionally edit `config.yaml` to change the model (default `gpt-4o-mini`), quality threshold (default 9.0), chunk size/overlap, or number of retrieved chunks.

---

## Reproducing the Results

The SQuAD v1.1 subset (150 samples) is included at `src/data/raw/squad_150.json` — no download needed.

```bash
python src/evaluation/evaluate_vanilla_rag.py     # ~15 min
python src/evaluation/evaluate_reranker_rag.py    # ~15 min
python src/evaluation/evaluate_critique_rag.py    # ~20 min
python src/evaluation/compute_oracle_scores.py    # optional, ~15 min — adds oracle scores
python src/evaluation/compare_all.py              # ~1 min — writes the comparison table
```

Results land in `results/comparison_table.txt` and `results/comparison_metrics.json`.
**Approximate total:** 30–55 minutes and ~$1.50–2.50 in API cost with `gpt-4o-mini`.

---

## Evaluation Methodology

- **Dataset:** 150 SQuAD v1.1 validation questions (extractive QA).
- **Oracle scoring:** the *same* LLM critic scores every system's output against the reference across the four dimensions above, **blind to which system produced each answer**, so the comparison is unbiased.
- **Automatic metrics:** ROUGE-L (lexical overlap) and BERTScore (semantic similarity).
- **Statistics:** paired t-tests and Cohen's d effect sizes across systems.

---

## Limitations

These are stated plainly because they bound what the numbers above can claim:
- **Oracle bias:** the oracle critic comes from the same model family (`gpt-4o-mini`) as the generator, which can introduce self-preference bias; human evaluation is needed to fully validate.
- **Scale:** 150 samples limits generalization.
- **Single domain:** only extractive QA (SQuAD) was tested; open-ended generation is untested.
- **Latency:** the 5.0s+ per-query latency makes the current setup ill-suited to real-time use.
- **Hallucination rate:** the 0.0% figure is self-reported by the critic and requires independent verification.

---

## Citation / Links

- **Report:** *Critique-Enhanced Retrieval-Augmented Generation: Improving Answer Quality through LLM-based Self-Assessment*, CSE 244, Fall 2025.
- **Code & data:** https://github.com/subhan75/Critique-Based-Retrieval-Augemented-Generation-RAG-
- **Video presentation:** https://drive.google.com/file/d/1b-vDQ69uVq55aQwdv94CLRPl7MEENiq0/view

## Authors

Subhan Shaikh · Sohail Syed — University of California, Santa Cruz

## License

MIT — see `LICENSE`.

## Acknowledgments

SQuAD (Rajpurkar et al., 2016); OpenAI API; the CSE 244 course staff.
