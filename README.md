# Financial Document Retrieval: Controlled Ablation of Chunking, Reranking & Metadata

## Context: FinanceBench

[FinanceBench](https://arxiv.org/abs/2311.11944) provides 150 expert-annotated questions over public 10-K/10-Q filings, with numerical and textual answers requiring precise passage retrieval. This project runs a controlled ablation over FinanceBench to isolate the marginal value of individual RAG components.

## Ablation Design

| Technique | Description | Expected F1 Gain |
|-----------|-------------|------------------|
| Baseline | Fixed-512 chunking, BM25 retrieval | — |
| Sentence chunking | Split on sentence boundaries | +2–5 pp |
| Cross-encoder reranking | Re-score top-50 candidates | +5–10 pp |
| LLM metadata annotation | Inject fiscal year, company, section | +3–7 pp |
| Query expansion | Synonym + HyDE expansion | +2–4 pp |
| Full pipeline | All techniques combined | +10–18 pp |

## Cost Analysis

Each technique incurs incremental inference cost. We report **cost per F1 point** (USD / F1 pp) to identify the most cost-efficient improvements:

- Sentence chunking: $0 marginal cost (pure algorithmic)
- Cross-encoder reranking: ~$0.002 / query (local model) or ~$0.01 / query (API)
- LLM metadata annotation: ~$0.005–$0.02 / document (one-time pre-processing)
- Query expansion (HyDE): ~$0.001–$0.005 / query

> **Note on the numbers above:** the "Expected F1 Gain" and cost figures in
> this README are projections, not measurements — there is currently no
> BM25/reranker/HyDE implementation wired up to real FinanceBench data in
> this repository, and `scripts/run_demo.py`'s ablation output is generated
> from a synthetic noise model (`findocretrieval.data.make_query_results`),
> not a real retrieval run. See `RESEARCH_GAP_ANALYSIS.md` for a full
> breakdown of what is implemented vs. projected, `PAPER_DRAFT.md` for the
> in-progress writeup, and `BLOG.md` for a plain-language summary.

## Quickstart

```bash
git clone https://github.com/anote-ai/research-financialdocumentretrieval.git
cd research-financialdocumentretrieval
pip install -e ".[dev]"
pytest tests/ -v
```

```python
from findocretrieval.core import Document, ChunkingConfig, fixed_size_chunker
from findocretrieval.evaluate import f1_score_tokens, marginal_gain

doc = Document(doc_id="aapl-2023", text="Apple Inc. reported revenue of $383B...", metadata={"year": 2023})
cfg = ChunkingConfig(strategy="fixed", chunk_size=512, overlap=64)
chunks = fixed_size_chunker(doc, cfg)
print(f"{len(chunks)} chunks created")

f1 = f1_score_tokens(predicted="383 billion", reference="$383B")
print(f"Token F1: {f1:.3f}")
```

## Citation

```bibtex
@misc{findocretrieval2024,
  title  = {Financial Document Retrieval: Controlled Ablation of Chunking, Reranking and Metadata},
  author = {Anote AI Research},
  year   = {2024},
  url    = {https://github.com/anote-ai/research-financialdocumentretrieval},
}

@article{financebench2023,
  title   = {FinanceBench: A New Benchmark for Financial Question Answering},
  author  = {Islam, Pranab and Kannappan, Anand and Goldie, Douwe and Patel, Rishi and Narayanan, Shreya},
  journal = {arXiv:2311.11944},
  year    = {2023},
}
```
