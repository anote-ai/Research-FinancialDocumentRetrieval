# FinDocRetrieval

> **Research question:** Do chunking strategy, reranking, and query expansion improve QA accuracy on financial filings (10-K, 10-Q, earnings transcripts) — and at what cost?

## Overview

FinDocRetrieval benchmarks RAG pipelines on financial document QA, inspired by the [FinanceBench](https://arxiv.org/abs/2311.11944) dataset. It measures token-level F1 and USD inference cost to quantify the accuracy-cost frontier of different retrieval techniques.

## Ablation Design

| Technique | mean F1 | mean Cost (USD) | Marginal Gain |
|-----------|---------|-----------------|---------------|
| baseline | 0.52 | $0.008 | — |
| +reranking | 0.59 | $0.018 | +0.07 |
| +metadata | 0.57 | $0.013 | +0.05 |
| +query_expansion | 0.58 | $0.015 | +0.06 |
| +hybrid | **0.66** | $0.039 | **+0.14** |

## Cost Analysis

Hybrid retrieval (BM25 + dense + reranking + query expansion) achieves the highest F1 at ~5x the baseline cost. For cost-sensitive deployments, +reranking alone offers the best cost-effectiveness (cost/F1 point: $0.031 vs $0.059 for +hybrid).

## Quickstart

```bash
pip install -e ".[dev]"
python scripts/run_demo.py
pytest tests/ -v
```

```bash
python -m pip install -e ".[dev]" #installation
python scripts/run_demo.py #run demo
python -m pytest tests/ -v #run tests
```

Note: Using python -m pytest is recommended because it ensures pytest runs from the currently active Python environment and avoids PATH-related issues on Windows.

## Target Venue

- **EMNLP 2026 FinNLP Workshop** — Empirical Methods in NLP, Financial NLP track

## Citation

```bibtex
@software{findocretrieval2026,
  title  = {FinDocRetrieval: Cost-Aware RAG Ablations on Financial Filings},
  author = {Anote AI},
  year   = {2026},
  url    = {https://github.com/anote-ai/research-financialdocumentretrieval}
}
```
