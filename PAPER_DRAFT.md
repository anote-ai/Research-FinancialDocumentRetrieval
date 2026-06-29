# FinDocRetrieval: Cost-Aware Retrieval Ablations and Failure-Mode Metrics for Financial Document QA

**Status: early draft / skeleton.** This document exists to give the project
a concrete writing target and to make explicit which numbers below are
measured vs. projected. Sections marked `(projected, pending full experiment
run)` are taken from DESIGN_DOC.md's "Expected results" and have NOT been
produced by running any code in this repository. No fabricated numbers are
presented as measured.

## Abstract (draft)

General-purpose retrieval benchmarks (BEIR, MS MARCO) do not capture three
failure modes specific to financial documents: table blindness (numeric
evidence trapped in tables that text-only retrievers miss), wrong-year
numeric retrieval (confidently returning the correct paragraph from the
wrong fiscal year), and cross-document synthesis (questions that require
stitching together evidence from multiple filings). We introduce three
metrics -- Table Retrieval Recall (TRR), Numeric Hallucination Rate (NHR),
and cross-document co-retrieval rate -- and an open-source implementation of
section-aware chunking for SEC filings. *(This abstract describes the
research program in DESIGN_DOC.md; the experiments below report what has
actually been run versus what remains future work.)*

## 1. Introduction

See `DESIGN_DOC.md` for full motivation. In short: production financial RAG
systems are evaluated with generic IR metrics (NDCG, MRR) that do not
distinguish table-origin from prose-origin evidence, do not penalize
retrieving a numerically wrong-but-textually-similar passage, and do not
measure whether all evidence required for a multi-document question was
actually retrieved.

## 2. Metrics

Implemented in `src/findocretrieval/findoc_metrics.py` (added in this
update) and `src/findocretrieval/evaluate.py` (pre-existing):

- **TRR** (`compute_trr`): recall over table-origin gold passages only.
- **Table-blindness gap** (`table_blindness_gap`): prose recall minus TRR.
- **NHR** (`compute_nhr`): fraction of top-1 retrieved numeric answers that
  deviate from gold by more than a relative tolerance (default 5%), plus a
  variant restricted to wrong-fiscal-year retrievals.
- **Cross-document co-retrieval rate** (`cross_document_coretrieval_rate`):
  fraction of multi-hop questions for which all required passages appear in
  the top-k retrieved set.
- **Numeric accuracy / table extraction F1** (pre-existing, `evaluate.py`):
  tolerance-based numeric comparison and cell-level precision/recall/F1 for
  table extraction.

All of the above are unit-tested (`tests/test_findoc_metrics.py`,
`tests/test_evaluate.py`) against synthetic inputs that exercise edge cases
(missing values, zero gold values, top-k cutoffs, no-table-gold questions).
None of these tests constitute a research result -- they verify the metric
functions are implemented correctly, not that any retrieval system performs
well or poorly.

## 3. Chunking strategies (implemented, `src/findocretrieval/core.py`)

Four chunkers are implemented and tested: fixed-size sliding window,
paragraph-boundary, sentence-boundary, and a section-aware "semantic"
chunker that detects SEC `Item` headers (`Item 1A`, `Item 7`, etc.) and
attaches a `section_label` to each chunk. The semantic chunker currently
uses a **length-ratio heuristic** in place of real embedding similarity
(documented in its docstring) -- it is a structural scaffold for a future
embedding-based version, not a working semantic similarity model.

## 4. Experiments

### 4.1 What has actually been run

- Unit tests over chunkers and metrics on hand-written and synthetically
  generated sample 10-K/10-Q text (`src/findocretrieval/data.py`). This
  confirms the code paths work, not that any retrieval system achieves a
  particular accuracy on real filings.
- A synthetic ablation demo (`scripts/run_demo.py`, `make_query_results` in
  `data.py`) that generates F1/cost numbers from a fixed baseline plus a
  hard-coded per-technique boost and Gaussian noise. **These are simulator
  outputs, not measurements of any real retriever.**

### 4.2 Experiments from DESIGN_DOC.md not yet implemented

| Experiment | Status |
|---|---|
| Exp 0: BM25 / DPR baseline on FinRAG-Bench | Not implemented -- no BM25 or dense retriever in the codebase, no FinRAG-Bench dataset |
| Exp 1: Table-blindness gap | Metric implemented (this update); no real retrieval run yet |
| Exp 2: Numeric Hallucination Rate | Metric implemented (this update); no real retrieval run yet |
| Exp 3: Section-aware chunking vs. sliding window | Chunkers implemented; no recall comparison run on real questions yet |
| Exp 4: Cross-document synthesis | Metric implemented (this update); no query-decomposition system implemented |
| Exp 5: Cost-quality Pareto frontier | Not implemented -- no GPT-4o or FinBERT reranker integration |

### 4.3 Expected results (projected, pending full experiment run)

The table below reproduces DESIGN_DOC.md's expected-result figures.
**These are hypotheses stated before any experiment was run, not measured
results**, and are reproduced here only so the eventual paper can show
prediction vs. measurement side by side.

| Metric | Baseline (BM25) | Best system | Source |
|---|---|---|---|
| NDCG@10 | 0.41 (projected, pending full experiment run) | 0.74, GPT-4o (projected, pending full experiment run) | DESIGN_DOC.md |
| TRR | 0.28 (projected, pending full experiment run) | 0.61, table-aware (projected, pending full experiment run) | DESIGN_DOC.md |
| NHR | 0.38 (projected, pending full experiment run) | 0.21, section-aware (projected, pending full experiment run) | DESIGN_DOC.md |
| Cross-doc co-retrieval | 0.31 (projected, pending full experiment run) | 0.67, decomposition (projected, pending full experiment run) | DESIGN_DOC.md |

## 5. Limitations (current)

- No real corpus: the only text in the repository is four short, synthetic,
  single-company sample filings used for unit tests and demos.
- No retrieval system: there is no BM25, dense, or hybrid retriever
  implementation to actually compute TRR/NHR/co-retrieval against.
- No human-annotated gold labels (table/prose origin, fiscal year, required
  passage sets for cross-doc questions).
- The semantic chunker's "similarity" scores are a length-ratio heuristic,
  not a trained embedding model.

## 6. Next steps toward a submittable paper

1. Build a minimal real corpus (10-50 filings from SEC EDGAR's bulk
   download API) and hand-label 50-100 QA pairs with origin/fiscal-year
   metadata.
2. Implement a BM25 baseline (e.g. via `rank_bm25`) and run Exp 0/1/2 against
   the real corpus, replacing every projected number above with a measured
   one.
3. Implement section-aware chunking recall comparison (Exp 3) against the
   same questions.
4. Only after 1-3 are measured, revisit Exp 4 (query decomposition) and
   Exp 5 (cost-quality, which needs an LLM API integration).

Target venue per README.md: EMNLP 2026 FinNLP Workshop.
