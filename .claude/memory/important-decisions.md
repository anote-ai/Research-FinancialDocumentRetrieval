# Important Decisions — FinancialDocumentRetrieval (findocretrieval)

## Existing Decisions Visible from Docs/Code
- **Core research framing is cost-aware, not just accuracy-aware**: the central contribution (per `RESEARCH_DESIGN.md`) is analyzing marginal F1 gain *per dollar* of each RAG technique, not just which technique wins on accuracy alone. Any new ablation condition added should report both F1 and USD cost to stay consistent with this framing.
- **Baseline (C0) configuration is fixed**: `TokenTextSplitter`, chunk_size=512, chunk_overlap=50, retrieval k=10, GPT-4o at temperature=0. Changing these for the baseline would break comparability with the ablation table — treat as a locked decision unless the user says otherwise.
- **Embedding model choice**: `all-MiniLM-L6-v2` chosen specifically because it's local/free (no API key, no per-call cost) — keeps embedding cost out of the USD-cost comparison, isolating LLM generation cost as the variable of interest.
- **Evaluation dataset scope intentionally reduced**: 133 of 150 FinanceBench questions are used due to 9 PDFs failing to download — this exclusion is documented and accepted, not a bug.
- **`DESIGN_DOC.md` vs `RESEARCH_DESIGN.md` scope mismatch**: `DESIGN_DOC.md` describes a much larger, more ambitious "FinRAG-Bench" (2,400 new QA pairs, new metrics like NHR/TRR, a full leaderboard) while `RESEARCH_DESIGN.md` describes a scoped-down ablation study on the existing 150-question FinanceBench sample. The actual code in `src/findocretrieval/` and the run scripts match the smaller, scoped `RESEARCH_DESIGN.md` plan. This looks like `DESIGN_DOC.md` was an earlier/aspirational vision that was later descoped — worth confirming with the user rather than assuming.

## Open Decisions / Questions for User Review
- Which document is authoritative going forward: `RESEARCH_DESIGN.md` (scoped ablation) or `DESIGN_DOC.md` (full FinRAG-Bench vision)? This affects how much new work (e.g. building the 2,400-question dataset) is actually in scope.
- Is the target venue **WSDM 2027** (per `RESEARCH_DESIGN.md`, abstract deadline Aug 17 2026, full paper Aug 24 2026) or **EMNLP 2026 FinNLP Workshop** (per `README.md`)? These have different timelines and would affect prioritization.
- Are the results shown in `README.md`'s ablation table (baseline 0.52 F1 ... +hybrid 0.66 F1) final results, or placeholder/illustrative numbers written before the real runs completed? Given `RESEARCH_DESIGN.md` describes C0 as still "in progress" as of its last update, this needs clarification.
- What is the actual current run status of C1–C6 conditions? `run_c1.py`/`run_c2.py` exist and are newer than the design doc's last-updated date — check `results/` directly for real progress before reporting status to the user.
