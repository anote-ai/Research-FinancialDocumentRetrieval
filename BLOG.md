# Why your finance chatbot might be confidently wrong (and how we're checking)

*A plain-language look at FinDocRetrieval, an in-progress research project on
retrieval for financial documents.*

## The problem in one sentence

If you ask an AI assistant "What was Acme's R&D spend last year?", and it
quietly grabs the number from the *wrong year's* 10-K filing, you get a
confident, well-formatted, completely wrong answer -- and most retrieval
systems used today have no way of noticing.

## Why financial documents are uniquely hard for RAG

Most "retrieve relevant text, then ask an LLM to answer" (RAG) systems were
built and tuned on web search and Wikipedia-style benchmarks. Financial
filings break a lot of the assumptions those benchmarks rely on:

- **Numbers live in tables, not sentences.** A balance sheet is a grid. A
  retrieval system trained to match prose often skips right over the table
  that has the actual answer.
- **The same sentence repeats every year with a different number.** "Total
  revenue was $X billion" appears in every 10-K a company files. A system
  that doesn't track *which year* it pulled from can swap in last year's (or
  next year's) number without anyone noticing.
- **Real questions span documents.** "How did margin trend across the last
  three fiscal years?" needs three filings stitched together, not one
  paragraph.
- **Filings have mandatory structure.** SEC rules require specific sections
  (Risk Factors, MD&A, etc.). Ignoring that structure when chunking documents
  throws away free, reliable signal.

## What we're building

This project, **FinDocRetrieval**, is an early-stage testbed for studying
these failure modes directly, rather than assuming a general-purpose
retrieval benchmark score tells you anything about financial-document
performance. Concretely, the codebase currently includes:

- Multiple **chunking strategies** for SEC-style filings (fixed-size,
  paragraph, sentence, and a section-aware "semantic" chunker that detects
  `Item 1A`, `Item 7`, etc. headers).
- **Evaluation metrics** for financial QA: token-F1, exact match, a
  numeric-tolerance accuracy check, and table-cell extraction scoring.
- New in this update: first implementations of three metrics from our
  research design doc that didn't exist in code before -- **Table Retrieval
  Recall (TRR)**, **Numeric Hallucination Rate (NHR)**, and
  **cross-document co-retrieval rate** -- so we can directly measure the
  table-blindness and wrong-year-retrieval failure modes described above.
- A small, currently **synthetic** ablation demo (`scripts/run_demo.py`)
  comparing baseline retrieval against reranking, metadata filtering, query
  expansion, and hybrid search, scored on cost and answer quality.

## What's real and what's a placeholder today

We want to be upfront about where this project actually stands, because
research credibility depends on it:

- The chunkers are real, tested code that runs on real SEC-filing-style text.
- The new TRR / NHR / cross-doc metrics are real, tested implementations --
  but they have not yet been run against an actual retrieval system or a
  real annotated dataset.
- The numbers in the current README's ablation table (e.g. "+hybrid: F1
  0.66") are **illustrative numbers from a synthetic data generator**, not
  measurements from running BM25, dense retrieval, or any other system on
  real filings. No retriever (BM25, DPR, ColBERT, etc.) is implemented yet.
- There is no FinRAG-Bench dataset yet -- no SEC EDGAR corpus, no annotated
  QA pairs at the scale described in our design doc (2,400 questions across
  6 sectors).

## What's next

The roadmap (tracked in `RESEARCH_GAP_ANALYSIS.md`) is to: (1) pull a small
real corpus from SEC EDGAR's bulk API, (2) hand-label a few dozen QA pairs
with table/prose origin and fiscal year, (3) wire up an actual BM25 baseline
against that data, and (4) replace every "expected result" number in the
design doc with a measured one. We'll publish updated numbers -- clearly
marked as measured, not projected -- as soon as that pipeline runs end to
end.
