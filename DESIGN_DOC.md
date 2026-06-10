# FinancialDocumentRetrieval — Research Design Document

## Goal

Build the first retrieval and QA benchmark specifically targeting financial documents — earnings calls, 10-K filings, analyst reports, loan agreements — with focus on the failure modes that matter most in finance: numeric hallucination, table-blind retrieval, and multi-document reasoning.

## Objective

1. Construct a benchmark of 500+ financial QA pairs spanning 4 document types and 3 reasoning categories (single-passage lookup, cross-table numeric reasoning, multi-document synthesis)
2. Evaluate leading RAG pipelines on this benchmark, measuring both retrieval quality and answer accuracy
3. Characterize the hallucination rate for numeric claims — the highest-stakes failure mode in financial AI

## Background / Motivation

The global financial services industry is the largest enterprise AI deployment sector by revenue. Every major bank and asset manager is deploying RAG-based document analysis tools. Yet there is no public benchmark measuring retrieval and QA quality on the document types these tools actually process.

Existing benchmarks (FinQA, TAT-QA) focus on single-document arithmetic reasoning. They do not measure: retrieval quality across large document corpora, multi-document synthesis, or hallucination rates for numeric claims — the failure mode that could cause actual financial harm.

## Experimental Design

### Baseline Experiment

**Replicate TAT-QA evaluation on 3 leading RAG pipelines (BM25+GPT-4o, E5+GPT-4o, Cohere+Command-R)**

- Metric: exact match (EM) and F1 on numeric answers
- Purpose: confirm evaluation infrastructure produces correct numbers vs. published TAT-QA results
- Expected result: all three pipelines score 45–65% F1 on TAT-QA

### Test Experiment 1: Table-Blind Retrieval Failure Rate

Take 100 QA pairs where the answer requires information from a table. Measure retrieval recall and rank for table rows. Compare standard sentence-level chunking vs. structure-aware chunking.

**Expected result:** standard chunking fails to retrieve the relevant table passage in 35–50% of cases; structure-aware chunking reduces this to <15%

### Test Experiment 2: Numeric Hallucination Rate

For 100 QA pairs with numeric answers, measure: hallucination rate = (answers where generated number differs from true number by >1%) / (numeric answers generated). Stratify by answer complexity, document type, answer magnitude.

**Expected result:** hallucination rate of 20–35% on computed numeric answers; table-sourced answers hallucinate at 2x the rate of text-sourced answers

### Test Experiment 3: Cross-Document Synthesis

Construct 50 QA pairs requiring information from 2+ documents. Evaluate whether pipelines retrieve from both relevant documents and correctly synthesize multi-source numeric claims.

**Expected result:** all current RAG pipelines fail on >60% of cross-document synthesis tasks

## Expected Results

1. A benchmark of 500+ financial QA pairs with table-structured documents and multi-document pairs
2. Numeric hallucination rate characterization across pipeline configurations
3. **Key finding:** "Current RAG pipelines hallucinate on 1 in 4 numeric financial claims"
4. Practical recommendation: structure-aware chunking reduces numeric hallucination by X%

## Why This Matters / Why People Would Care

- **Financial institutions:** deploying RAG in production now; concrete hallucination rates will drive immediate procurement decisions
- **Regulators:** SEC, FCA actively investigating AI in financial advice; published hallucination rates are exactly the evidence they need
- **AI companies:** Anthropic, OpenAI, Google want to show strong performance on high-stakes financial tasks
- **Researchers:** numeric hallucination and table reasoning are underexplored despite practical importance

## Timeline

| Month | Milestone |
|---|---|
| 1–2 | QA pair construction (500 pairs, 4 document types, annotation + expert review) |
| 3 | Pipeline evaluation infrastructure |
| 4 | Baseline + test experiments |
| 5 | Analysis + hallucination characterization |
| 6 | Submission to ACL 2026 Industry Track |

## Related Issues

- Design doc GitHub issue: #19
- Target conferences: see issues labeled `conference-prep`
- Reproducibility package: see issues labeled `artifact-release`
