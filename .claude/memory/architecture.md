# Architecture — FinancialDocumentRetrieval (findocretrieval)

## Tech Stack
- **Language**: Python >=3.10 (dev env actually uses Python 3.11 via Anaconda per `RESEARCH_DESIGN.md`).
- **Core numerics/data**: numpy, pandas, scikit-learn, rich, tqdm.
- **RAG framework**: LangChain ecosystem — `langchain`, `langchain-core`, `langchain-community`, `langchain-huggingface`, `langchain-experimental`, `langchain-text-splitters`.
- **Vector store**: FAISS (`faiss-cpu`), local, no external service.
- **Embeddings**: `sentence-transformers`, specifically `all-MiniLM-L6-v2` (HuggingFace, local, no API key required).
- **Lexical retrieval**: BM25 via `rank-bm25`.
- **Reranking / query expansion**: implemented in `src/findocretrieval/` (see below).
- **PDF parsing**: `pypdf`.
- **Evaluation metric**: ROUGE-L (`rouge-score`) plus custom token-level F1 (per README).
- **Tokenization**: `tiktoken`.
- **LLM provider**: OpenAI (GPT-4o, temperature=0) is the configured/active provider per `RESEARCH_DESIGN.md`. `requirements.txt` also lists optional `langchain-ollama` (local, no key) and comments showing `langchain-anthropic`/`langchain-openai` as swappable providers.
- **Packaging**: setuptools, `src/`-layout package (`pyproject.toml`, `[tool.setuptools.packages.find] where = ["src"]`).
- **Dev tooling**: pytest + pytest-cov, ruff (lint), mypy (non-strict).

## Main Modules (`src/findocretrieval/`)
- `core.py` (341 lines) — core data classes: `Document`, `Chunk`, `SemanticChunk` (embedding-aware chunk variant with coherence hints), plus likely the main pipeline orchestration (largest module).
- `chunking.py` (76 lines) — chunking strategies (e.g. sliding-window vs. semantic/section-aware).
- `embeddings.py` (23 lines) — embedding model wrapper (sentence-transformers).
- `retriever.py` (82 lines) — retrieval logic (BM25 / dense / hybrid).
- `query_expansion.py` (53 lines) — query expansion technique.
- `evaluate.py` (265 lines) — evaluation metrics (F1, ROUGE-L, cost tracking).
- `data.py` (229 lines) — dataset loading (FinanceBench CSV + PDFs).
- `__init__.py` (49 lines) — package exports.

## Entry Points / Scripts
- `download_data.py`, `download_pdfs.py` — fetch FinanceBench sample data and source PDFs from SEC filings.
- `test_pipeline.py` — smoke-tests the full pipeline end-to-end on a small sample (validated working per `RESEARCH_DESIGN.md`).
- `run_c0.py`, `run_c1.py`, `run_c2.py` — run individual ablation conditions (C0 = baseline; C1/C2 = specific ablation variants) against the full/filtered question set, saving progress incrementally (every 10 questions per `RESEARCH_DESIGN.md`) to `results/`.
- `scripts/run_demo.py` — quick demo run (referenced in README quickstart).
- `scripts/run_ablation.py` — likely the full multi-condition ablation runner.

## Data Flow
1. `download_data.py` / `download_pdfs.py` pull the FinanceBench question set (`financebench_sample.csv`) and associated SEC filing PDFs into `data/`.
2. Documents are parsed (`pypdf`) and chunked (`chunking.py` — sliding-window baseline at 512 tokens / 50 overlap per C0 config, or alternative strategies for ablation conditions).
3. Chunks are embedded (`embeddings.py`, `all-MiniLM-L6-v2`) and indexed into FAISS; BM25 index built in parallel for hybrid retrieval.
4. At query time, `retriever.py` retrieves top-k (k=10 for C0) chunks, optionally reranked and/or with query expansion (`query_expansion.py`) depending on the ablation condition.
5. Retrieved context + question sent to the LLM (GPT-4o via LangChain) to generate an answer.
6. `evaluate.py` scores the answer against gold (token-level F1, ROUGE-L) and tracks USD cost per query.
7. Results saved incrementally to `results/`.

## External Services / Dependencies
- **OpenAI API** — GPT-4o for answer generation (requires `OPENAI_API_KEY`; billed).
- **HuggingFace models** — downloaded locally for embeddings (`all-MiniLM-L6-v2`), no API key.
- **SEC EDGAR** — source of the underlying 10-K/10-Q/transcript PDFs (via `download_pdfs.py`).

## Testing Structure
- `tests/` — pytest suite: `test_core.py`, `test_data.py`, `test_evaluate.py` (plus `__init__.py`).
- Configured via `pyproject.toml` (`[tool.pytest.ini_options] testpaths = ["tests"]`).
- Root-level `test_pipeline.py` is a separate, broader smoke test (not under `tests/`), run directly with `pytest test_pipeline.py -v` or as a script.
