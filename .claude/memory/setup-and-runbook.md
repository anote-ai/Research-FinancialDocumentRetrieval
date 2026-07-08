# Setup & Runbook — FinancialDocumentRetrieval (findocretrieval)

## Install Dependencies
Per README quickstart:
```bash
pip install -e ".[dev]"
```
Alternative (more complete list of pinned RAG dependencies) via:
```bash
pip install -r requirements.txt
```
Note: `requirements.txt` includes the full LangChain/FAISS/sentence-transformers/BM25 stack plus commented-out lines for optional LLM providers (`langchain-anthropic`, `langchain-openai`) — uncomment the one matching your configured provider before installing if not already installed.

Per `RESEARCH_DESIGN.md`, the working environment actually used is:
- **Conda environment**: `findocretrieval`
- **Python**: 3.11 (Anaconda distribution)
- **OS**: Windows

## Run Locally
Quick demo:
```bash
python scripts/run_demo.py
```
Smoke test (validates full pipeline on a few questions):
```bash
python test_pipeline.py
```
Run a specific ablation condition (each costs real OpenAI API money — confirm with user before running):
```bash
python run_c0.py   # baseline
python run_c1.py
python run_c2.py
```
Full ablation orchestration:
```bash
python scripts/run_ablation.py
```

## Tests / Lint / Build
```bash
pytest tests/ -v            # or: python -m pytest tests/ -v
ruff check .                 # lint (configured in pyproject.toml, line-length 100)
mypy .                        # type check (non-strict, per pyproject.toml)
```
README recommends `python -m pytest` over bare `pytest` on Windows to avoid PATH-related issues with the active environment.

## Important Environment Variables (no secrets included)
- **`OPENAI_API_KEY`** — required for the active LLM provider (GPT-4o). Per `RESEARCH_DESIGN.md`, this was "confirmed and set, provided by Natan Vidra" — treat as already configured in the user's environment; never print or ask to see its value.
- Optional, only if switching providers: `ANTHROPIC_API_KEY` (if uncommenting `langchain-anthropic`).
- No API key needed for local embeddings (HuggingFace `all-MiniLM-L6-v2`) or Ollama if used instead.

## Common Troubleshooting (from `RESEARCH_DESIGN.md`)
- **LangChain import changes**: this project's LangChain version requires updated import paths — e.g. `from langchain_text_splitters import TokenTextSplitter` instead of the older `from langchain.text_splitter import TokenTextSplitter`. If you hit an `ImportError` for a LangChain symbol, check whether it has moved to a dedicated sub-package first.
- **PDF download gaps**: 9 of 84 source PDFs failed to download (Adobe 2015/16/17/22, J&J, MGM) — 17 of 150 questions are excluded from evaluation as a result (133 usable). This is a known, accepted limitation, not a bug to fix silently.
- **Windows + pytest**: prefer `python -m pytest` over bare `pytest` to avoid PATH issues (per README).
