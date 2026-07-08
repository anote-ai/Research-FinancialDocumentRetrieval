# Onboarding Report — FinancialDocumentRetrieval (Agent: John)

## What the Agent Learned
This is a Python research project (`findocretrieval`) running a cost-aware ablation study of RAG pipeline techniques (chunking, reranking, metadata enrichment, query expansion) on financial document QA, using the FinanceBench benchmark (150 questions over SEC filings). It's an Anote AI research fellowship project authored by Elaine Hong, supervised by Natan Vidra.

## Files/Folders Inspected
- `README.md`, `DESIGN_DOC.md`, `RESEARCH_DESIGN.md`
- `pyproject.toml`, `requirements.txt`
- Directory listing of `src/findocretrieval/` (with line counts per module), `scripts/`, `tests/`
- `src/findocretrieval/core.py` (partial read)
- Root scripts noted but not opened in depth: `download_data.py`, `download_pdfs.py`, `run_c0.py`, `run_c1.py`, `run_c2.py`, `test_pipeline.py`
- Not read: `data/`, `results/` contents, `financebench_sample.csv` contents, `.pytest_cache/`, `__pycache__/`

## What the Project Seems to Do
Benchmarks and ablates RAG retrieval techniques for financial-document QA, quantifying both accuracy (token-F1) and dollar cost per query, aiming at an academic publication (venue unclear — see below).

## Risks / Confusion / Missing Information
- **Two design documents disagree in scope**: `DESIGN_DOC.md` describes a much larger "FinRAG-Bench" vision; `RESEARCH_DESIGN.md` (dated, more current-looking) describes a scoped ablation on the existing FinanceBench sample. The actual code matches the smaller scope.
- **Target venue is inconsistent** between README (EMNLP 2026 FinNLP) and RESEARCH_DESIGN.md (WSDM 2027).
- **Run status may be stale in docs**: `RESEARCH_DESIGN.md` says C0 was still running and C1-C6 were pending as of its last update, but `run_c1.py`/`run_c2.py` already exist with later file dates — actual progress needs to be checked against `results/` rather than assumed from the doc.
- The README's ablation results table looks complete/final — unclear if these are real results or placeholders.
- Pipeline runs call the paid OpenAI API (GPT-4o) — running them has a real cost.

## Recommended Next Steps
1. User reviews this onboarding report and the four memory files for accuracy.
2. Clarify which design document (`DESIGN_DOC.md` vs `RESEARCH_DESIGN.md`) reflects current intent.
3. Confirm actual target venue and deadline.
4. Check `results/` directly (or ask the user) to determine true current run status before planning next steps.
5. Once confirmed, populate `.claude/workspace/current-task.md` with the first real task.
