# Project Summary — FinancialDocumentRetrieval (findocretrieval)

## What the project appears to be
A research codebase ("FinDocRetrieval") benchmarking Retrieval-Augmented Generation (RAG) pipelines on financial document question-answering, using the FinanceBench dataset (150 questions over SEC 10-K/10-Q filings and earnings call transcripts).

## Main Purpose
Run a **controlled ablation study** measuring how chunking strategy, reranking, metadata enrichment, and query expansion affect (a) token-level F1 accuracy and (b) USD inference cost per query on financial-document QA — framed as a "cost-accuracy frontier" analysis, targeting an academic publication.

## Main Users / Stakeholders
- **Author**: Elaine Hong (per `RESEARCH_DESIGN.md`), an Anote AI Research Fellow (2026 fellowship).
- **Supervisor**: Natan Vidra, CEO, Anote AI.
- Target audience: academic — **WSDM 2027** (per `RESEARCH_DESIGN.md`) and/or **EMNLP 2026 FinNLP Workshop** (per `README.md`; these two documents list different target venues, worth clarifying with the user which is current).

## Current Project Status (inferred, as of `RESEARCH_DESIGN.md` last-updated June 23, 2026)
- Environment set up (conda env `findocretrieval`, Python 3.11, Windows/Anaconda).
- FinanceBench 150-question sample downloaded (`financebench_sample.csv`); 75 of 84 source PDFs downloaded (133/150 questions usable — 9 PDFs failed: Adobe 2015/16/17/22, J&J, MGM, excluding 17 questions).
- OpenAI API key confirmed/set; end-to-end pipeline validated on a 5-question smoke test (`test_pipeline.py`).
- **C0 (baseline) evaluation** was in progress (`run_c0.py`, 133 questions, GPT-4o) as of the last update in `RESEARCH_DESIGN.md`.
- C1–C6 ablation conditions were still pending as of that update — but `run_c1.py` and `run_c2.py` already exist in the repo (dated Jul 8, later than the doc's June 23 "last updated" stamp), suggesting more progress has happened since the design doc was last updated. **Treat `RESEARCH_DESIGN.md`'s status table as potentially stale** — confirm actual progress by checking `results/` contents and file dates before reporting status.
- The top-level `README.md` already shows a finished-looking results table (baseline/+reranking/+metadata/+query_expansion/+hybrid) — unclear if these are final results or illustrative/placeholder numbers pending the real run. Worth clarifying with the user.

## Important Files/Folders
- `README.md` — public-facing summary with a results table and quickstart.
- `DESIGN_DOC.md` — broader/original design document (vision: "FinRAG-Bench", a large from-scratch benchmark — appears more ambitious/aspirational than the current scoped-down FinanceBench ablation).
- `RESEARCH_DESIGN.md` — the actively-maintained research plan with setup status, environment details, timeline, and open questions; most current source of ground truth on progress.
- `financebench_sample.csv` — the 150-question FinanceBench sample.
- `download_data.py`, `download_pdfs.py` — data acquisition scripts.
- `run_c0.py`, `run_c1.py`, `run_c2.py` — ablation condition run scripts (cost real money via OpenAI API).
- `test_pipeline.py` — smoke test for the pipeline.
- `src/findocretrieval/` — core library (chunking, embeddings, retriever, evaluate, query_expansion, core).
- `scripts/run_ablation.py`, `scripts/run_demo.py` — orchestration scripts.
- `tests/` — pytest test suite for the library.
- `data/`, `results/` — data and results directories (not deeply inspected — respect existing contents, do not overwrite).

## Unknowns / Assumptions
- Whether `DESIGN_DOC.md` (ambitious FinRAG-Bench vision) or `RESEARCH_DESIGN.md` (scoped FinanceBench ablation) reflects the actual current direction — they describe different-scale projects. Assume `RESEARCH_DESIGN.md` is authoritative unless told otherwise, since it has explicit dates and setup status.
- Actual current status of C0–C6 runs — needs verification against `results/` contents, not just the docs.
- Whether the results table in `README.md` is final or placeholder.
- True target publication venue (WSDM 2027 vs EMNLP 2026 FinNLP) — the two docs disagree.
