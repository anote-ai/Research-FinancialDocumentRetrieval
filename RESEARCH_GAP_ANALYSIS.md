# Research Gap Analysis: DESIGN_DOC.md vs. Implementation

This document compares the vision and experiments described in
`DESIGN_DOC.md` against what is actually implemented in this repository as
of this audit, so contributors can see at a glance what is real, what is a
placeholder, and what is still aspirational.

## Scope mismatch

`DESIGN_DOC.md` describes a large benchmark effort ("FinRAG-Bench"): a
2,400-question dataset built from ~6,000 SEC EDGAR filings across 6 sectors,
with human-annotated table/prose/cross-doc labels, evaluated against nine
retrieval systems including ColBERT-v2, DPR, FinBERT-retrieval, and GPT-4o.

The actual repository (per `README.md` and `src/findocretrieval/`) is a
much smaller-scope project: a chunking and evaluation-metrics library
demonstrated on four short synthetic sample filings, with a synthetic
ablation demo. There is no dataset, no corpus ingestion, no trained or
API-backed retriever of any kind.

## Item-by-item gap table

| DESIGN_DOC.md item | Implementation status |
|---|---|
| FinRAG-Bench dataset (2,400 QA pairs, SEC EDGAR) | Not started. Only 4 synthetic sample documents and 7 hand-written QA pairs exist (`data.py`). |
| XBRL / PDF table parsers (`parsers/xbrl_parser.py`, `pdf_parser.py`) | Not implemented. No table parsing of any kind. |
| Section-aware chunking (`sec_chunker.py`) | Partially implemented: `semantic_chunker` in `core.py` detects `Item` headers via string matching, but similarity scores are a length-ratio heuristic, not embeddings. |
| BM25 / DPR / ColBERT-v2 / FinBERT-retrieval / GPT-4o retrieval baselines | Not implemented. No retrieval system exists in the codebase at all -- only chunkers and scoring metrics. |
| NHR metric | Not implemented before this audit. Added in `src/findocretrieval/findoc_metrics.py`. |
| TRR metric | Not implemented before this audit. Added in `src/findocretrieval/findoc_metrics.py`. |
| Cross-document synthesis score | Not implemented before this audit. Added as `cross_document_coretrieval_rate` in `src/findocretrieval/findoc_metrics.py`. |
| Experiment 0 (baseline NDCG/MRR) | Not implemented -- no retriever, no NDCG/MRR computation in the codebase. |
| Experiments 1-5 scripts (`experiments/exp1_*.py` ... `exp5_*.py`) | Not implemented -- the `experiments/` directory does not exist. |
| Leaderboard (`leaderboard/submit.py`) | Not implemented. |
| Paper draft | Did not exist before this audit. Added as `PAPER_DRAFT.md`. |
| Blog post / accessible summary | Did not exist before this audit. Added as `BLOG.md`. |

## What the existing README claims vs. reality

README.md presents an ablation table (baseline / +reranking / +metadata /
+query_expansion / +hybrid) with specific F1 and cost numbers. Tracing this
through the code: `scripts/run_demo.py` calls
`findocretrieval.data.make_query_results`, which generates these numbers
from a fixed base F1 (0.52) plus a **hard-coded per-technique boost**
(e.g. `+hybrid: 0.14`) and Gaussian noise (`rng.gauss(0, 0.02)`). No actual
reranker, metadata filter, query expansion, or hybrid retrieval system is
implemented or run. The README does not currently disclose that these
numbers are synthetic; a reader could reasonably mistake them for measured
results. (Flagged here; recommend the README be updated to label this table
as illustrative/synthetic, matching the disclosure already added to
BLOG.md and PAPER_DRAFT.md in this update.)

## Recommended priority order for closing the gap

1. Small real corpus + hand-labeled QA set (highest leverage: unblocks
   every other experiment).
2. BM25 baseline retriever (cheapest system to implement; unblocks Exp 0-2).
3. Run TRR/NHR against (1) + (2) using the metrics added in this update.
4. Section-aware chunking recall comparison (Exp 3) -- the chunkers already
   exist, only the recall-comparison harness is missing.
5. Cross-document and cost-quality experiments (Exp 4-5) -- defer until
   1-4 are measured, since they depend on an LLM API integration.
