# Research Design Document

## Financial Document Retrieval: Controlled Ablation of Chunking, Reranking & Metadata

| | |
|---|---|
| **Author** | Elaine Hong |
| **Supervisor** | Natan Vidra, CEO, Anote AI |
| **Fellowship** | Anote AI Research Fellowship 2026 |
| **Target Venue** | WSDM 2027 |
| **Abstract Deadline** | August 17, 2026 |
| **Full Paper Deadline** | August 24, 2026 |
| **Last Updated** | June 23, 2026 |

---

## Table of Contents

1. [Overview](#1-overview)
2. [Setup Status](#2-setup-status)
3. [Dataset](#3-dataset)
4. [Methodology](#4-methodology)
5. [Expected Results](#5-expected-results)
6. [Related Work](#6-related-work)
7. [Paper Structure](#7-paper-structure)
8. [Timeline](#8-timeline)
9. [Risks & Mitigations](#9-risks--mitigations)
10. [Open Questions](#10-open-questions)

---

## 1. Overview

This document defines the research design for a controlled ablation study investigating how chunking strategy, reranking, metadata enrichment, and query expansion affect retrieval accuracy and inference cost in financial document question answering. The evaluation dataset is FinanceBench — 150 publicly available questions over SEC filings including 10-K, 10-Q, and earnings call transcripts from publicly traded companies.

The central contribution is a **cost-aware analysis of RAG pipeline components**. Rather than simply asking which technique improves accuracy, this paper asks at what cost per query and with what marginal return. This cost-frontier framing is absent from most prior RAG ablation work and is the primary differentiator from related papers including the closest competitor, Patel et al. (2026).

> **Core Research Question:** Which combination of chunking strategy, reranking, metadata enrichment, and query expansion maximizes token-level F1 on FinanceBench — and what is the marginal cost per F1 point for each technique individually and in combination?

---

## 2. Setup Status

> Last updated: June 23, 2026

| Status | Task | Notes |
|--------|------|-------|
| ✅ | Repository cloned and installed | `C:\Projects\Research-FinancialDocumentRetrieval` |
| ✅ | FinanceBench 150-question sample downloaded | `financebench_sample.csv` |
| ✅ | 75 of 84 source PDFs downloaded | `data/pdfs/` — 133/150 questions usable |
| ✅ | OpenAI API key confirmed and set | Provided by Natan Vidra |
| ✅ | Conda environment created | `findocretrieval`, Python 3.11, Windows (Anaconda) |
| ⚠️ | 9 PDFs failed to download | Adobe 2015/16/17/22, J&J, MGM — 17 questions excluded, manual download pending |
| ⏳ | OpenAI + LangChain libraries installing | `pip install openai langchain-openai` |
| ⏳ | Document indexing (chunking + vector store) | Next step after install completes |
| ⏳ | C0 baseline evaluation run | Next step |

### Environment Details

| Detail | Value |
|--------|-------|
| OS | Windows |
| Python version | 3.11 (Anaconda) |
| Conda environment | `findocretrieval` |
| Working directory | `C:\Projects\Research-FinancialDocumentRetrieval` |
| LLM provider | OpenAI (GPT-4o) |
| Embedding model | `text-embedding-3-large` (OpenAI) or `all-MiniLM-L6-v2` (HuggingFace fallback) |

### Failed PDF Downloads

The following 9 documents failed due to blocked company investor pages. Need manual download from [SEC EDGAR](https://www.sec.gov/cgi-bin/browse-edgar):

- `ADOBE_2015_10K`
- `ADOBE_2016_10K`
- `ADOBE_2017_10K`
- `ADOBE_2022_10K`
- `JOHNSON_JOHNSON_2022_10K`
- `JOHNSON_JOHNSON_2022Q4_EARNINGS`
- `JOHNSON_JOHNSON_2023_8K_dated-2023-08-30`
- `JOHNSON_JOHNSON_2023Q2_EARNINGS`
- `MGMRESORTS_2022Q4_EARNINGS`

---

## 3. Dataset

### 3.1 FinanceBench Overview

FinanceBench (Islam et al., 2023) is the primary evaluation benchmark. It comprises 10,231 questions about publicly traded companies with corresponding gold answers and evidence strings. Questions are designed to reflect real analyst workflows — precise, often numerical, and requiring multi-step reasoning over dense financial tables.

> **Key finding from the original paper:** GPT-4-Turbo used with a standard retrieval system incorrectly answered or refused to answer **81% of questions**. This is the baseline problem this paper is designed to systematically address.

### 3.2 Downloaded Sample

| Metric | Full FinanceBench | Public Sample | Usable for Experiments |
|--------|-------------------|---------------|------------------------|
| Total questions | 10,231 | 150 | 133 |
| Unique documents | ~800+ | 84 | 75 |
| Companies covered | ~100+ | ~40 | ~35 |
| Document types | 10-K, 10-Q, Earnings, 8-K | 10-K, 10-Q, Earnings, 8-K | 10-K, 10-Q, Earnings, 8-K |
| Failed downloads | — | 9 docs | 17 questions excluded |

The 9 failed PDFs cover 17 questions. Experiments proceed with 133 questions. The final paper will note this exclusion explicitly.

### 3.3 Question Type Breakdown

The dataset contains three main question types that stress different pipeline components:

- **Metrics-generated** — multi-step numeric calculations (e.g. fixed asset turnover ratio, 3-year average capex %). Requires finding multiple numbers across the filing and computing a formula.
- **Novel-generated** — extractive and interpretive questions requiring locating specific passages.
- **Boolean** — yes/no questions about company policies, dividends, etc.

> **Secondary Research Question:** Is the primary failure mode on numeric questions a retrieval problem (correct chunk never retrieved) or a reasoning problem (correct chunk retrieved, model miscalculates)? A 50-question pilot will determine this before the full experiment suite runs.

---

## 4. Methodology

### 4.1 Ablation Design

The study runs 7 conditions. Each varies exactly one component from the C0 baseline while holding all others fixed. C6 combines all techniques to measure the upper bound.

| Condition | Description | Chunking | Reranking | Metadata | Query Exp. | Cost Tier |
|-----------|-------------|----------|-----------|----------|------------|-----------|
| **C0** | Baseline — fixed chunking, no enhancements | Fixed 512 | ✗ | ✗ | ✗ | Low |
| **C1** | Semantic chunking only | Semantic | ✗ | ✗ | ✗ | Low |
| **C2** | Recursive / structure-aware chunking | Recursive | ✗ | ✗ | ✗ | Low |
| **C3** | Baseline + cross-encoder reranking | Fixed 512 | ✓ | ✗ | ✗ | Medium |
| **C4** | Baseline + metadata enrichment | Fixed 512 | ✗ | ✓ | ✗ | Medium |
| **C5** | Baseline + query expansion (HyDE) | Fixed 512 | ✗ | ✗ | ✓ | Medium |
| **C6** | Hybrid — all techniques combined | Recursive | ✓ | ✓ | ✓ | High |

Each condition run with **3 random seeds**. Results reported as mean ± standard deviation. Statistical significance via bootstrap resampling (n=1,000, p < 0.05).

### 4.2 Pipeline Components

#### Chunking Strategies

| Strategy | Description |
|----------|-------------|
| **Fixed (C0)** | 512-token chunks, 50-token overlap, naive token boundary split |
| **Semantic (C1)** | Split on semantic similarity boundaries using sentence embeddings; variable chunk size |
| **Recursive (C2)** | Hierarchical split: section headers → paragraphs → sentences; treats tables as single units |

#### Reranking (C3)

After retrieving the top-20 chunks, a cross-encoder re-scores and reorders them, keeping the top-5 to pass to the generator.

- Candidate models: Cohere Rerank v3, `BAAI/bge-reranker-v2-m3`
- Final model choice: to be confirmed

#### Metadata Enrichment (C4)

Each chunk tagged with: `company`, `fiscal_year`, `doc_type`, `section_header`, `page_number`. Metadata prepended to chunk text at index time and used as retrieval filters.

#### Query Expansion (C5)

- **HyDE:** Generate a hypothetical answer, embed it, retrieve against that embedding alongside the original query
- **Multi-query:** Generate 3 paraphrases, retrieve against each, union results before reranking

### 4.3 Evaluation Metrics

| Metric | Description | Why It Matters |
|--------|-------------|----------------|
| **Token-level F1** | Overlap between predicted and gold answer spans | Primary accuracy signal |
| **Exact Match (EM)** | Binary correctness — predicted matches gold exactly | Critical for numeric questions |
| **Retrieval Recall@k** | Fraction of gold evidence passages in top-k retrieved | Isolates retrieval from generation |
| **Cost per Query (USD)** | Total API spend divided by number of questions | The cost-frontier differentiator |
| **Latency (p50/p95)** | Query response time in seconds | Practical deployment constraint |

### 4.4 Implementation

| Component | Implementation |
|-----------|---------------|
| Embedding model | `text-embedding-3-large` (OpenAI) |
| Generator model | GPT-4o (OpenAI) |
| Vector store | FAISS (local dev) → Pinecone or Chroma (final runs) |
| Sparse retrieval | `rank_bm25` for BM25 in hybrid condition (C6) |
| Framework | LangChain + LangChain-community |
| Evaluation harness | Custom Python — ROUGE-L for F1, exact string match for EM |
| Source documents | 75 PDFs in `data/pdfs/` — 133 of 150 questions covered |

---

## 5. Expected Results

| Condition | Expected F1 | Expected Cost/Query | Notes |
|-----------|-------------|---------------------|-------|
| C0 Baseline | ~0.50–0.55 | ~$0.008 | Floor — naive pipeline |
| C1 Semantic | ~0.53–0.57 | ~$0.009 | Marginal improvement |
| C2 Recursive | ~0.54–0.58 | ~$0.009 | Better on table questions |
| C3 Reranking | ~0.59–0.63 | ~$0.018 | Best cost-efficiency ratio |
| C4 Metadata | ~0.55–0.59 | ~$0.013 | Helps on year/company disambiguation |
| C5 Query Exp. | ~0.56–0.60 | ~$0.015 | Limited benefit on numeric questions |
| C6 Hybrid | ~0.64–0.68 | ~$0.035–0.040 | Highest F1; ~4–5x baseline cost |

**Primary output:** A cost-frontier plot with F1 on the x-axis and cost per query on the y-axis, each condition as a labeled point.

---

## 6. Related Work

### Direct Predecessors

- **Vidra et al. (2024)** — [arXiv:2404.07221](https://arxiv.org/abs/2404.07221). Anote's prior work. Direct predecessor. This paper extends it with more rigorous ablation design, explicit cost measurement, and updated models.
- **Islam et al. (2023)** — [FinanceBench, arXiv:2311.11944](https://arxiv.org/abs/2311.11944). Primary evaluation dataset. Core motivation.

### Closest Competitor

- **Patel et al. (2026)** — [arXiv:2604.01733](https://arxiv.org/abs/2604.01733). Finds BM25 outperforms dense retrieval; hybrid + reranking wins; query expansion limited on numeric queries. **Our differentiators:** cost measurement, chunking ablation (their explicit future work), token-level F1 metric.

### Supporting References

| Paper | Relevance |
|-------|-----------|
| Banerjee et al. (2025) — [arXiv:2510.24402](https://arxiv.org/abs/2510.24402) | Metadata-driven RAG on FinanceBench. Relevant to C4. |
| Kim et al. (2025) — [arXiv:2411.16732](https://arxiv.org/abs/2411.16732) | Multi-reranker for FinanceRAG. Relevant to C3. |
| FinGEAR — EMNLP Findings 2025 | Multi-component financial retrieval ablation. |
| Greenback Bears (2024) — [arXiv:2411.07142](https://arxiv.org/abs/2411.07142) | GPT-4 gets 19% accuracy with ada-002 alone on FinanceBench. |
| FinAgentBench (ICAIF 2025) | Larger SEC corpus; tables treated as single chunk units. |

---

## 7. Paper Structure

**Target:** 8 pages + references | **Format:** ACM double-column (WSDM 2027)

| Section | Pages | Content |
|---------|-------|---------|
| 1. Introduction | ~1 | Motivate financial QA difficulty; identify cost-aware gap; state contributions |
| 2. Related Work | ~1 | Cover FinanceBench and prior ablations; address Patel et al. 2026 directly |
| 3. Methodology | ~1.5 | Dataset, 7 ablation conditions, pipeline components, evaluation metrics |
| 4. Experiments | ~0.5 | Implementation details, models used, compute setup, API cost tracking |
| 5. Results | ~2 | Ablation table, cost-frontier plot, per-metric and per-question-type breakdown |
| 6. Analysis | ~1 | Where each technique helps/hurts; table structure vs. numeric reasoning; failure modes |
| 7. Conclusion | ~0.5 | Practical recommendations, limitations, future work |
| References | ~0.5 | ACM format |

---

## 8. Timeline

| Dates | Phase | Deliverable | Deadline |
|-------|-------|-------------|----------|
| Jun 9–23 | Setup & Scoping | Repo running, data downloaded, API key confirmed, lit review started, abstract drafted | **Jun 23 — Abstract due ✓** |
| Jun 23–Jul 14 | Lit Review & Baseline | Full lit review complete; C0 baseline running with F1 + cost metrics | **Jul 14 — Methodology done** |
| Jul 14–21 | Experiments | All 7 conditions run; results table and cost-frontier plot generated | **Jul 21 — Full draft due** |
| Jul 21–28 | Writing — Draft 1 | Full paper written and shared for peer review | **Jul 28 — Peer feedback** |
| Jul 28–Aug 4 | Revision | Statistical rigor, reproducibility package, ethics statement | **Aug 4 — Final polish** |
| Aug 5–10 | Submission Prep | ACM formatting; arXiv preprint submitted; fellowship presentation | **Aug 10 — arXiv live** |
| Aug 17 | WSDM Abstract | Abstract submitted to WSDM 2027 | **Hard deadline** |
| Aug 24 | WSDM Full Paper | Full paper submitted to WSDM 2027 | **Hard deadline** |

---

## 9. Risks & Mitigations

| Risk | Level | Mitigation |
|------|-------|------------|
| Experiments take longer than expected | 🔴 High | Parallelize conditions. Start C0 immediately — it unblocks all downstream runs. |
| 9 missing PDFs remain unavailable | 🟡 Medium | Proceed with 133/150 questions. Note exclusion in paper. Try SEC EDGAR manually. |
| Overlap with Patel et al. 2026 | 🟡 Medium | Emphasize cost measurement and chunking ablation — their explicit future work. |
| Windows environment compatibility issues | 🟡 Medium | Use conda environment consistently. Fall back to HuggingFace embeddings if OpenAI issues arise. |
| Table structure bottleneck is negligible | 🟢 Low | If pilot shows it's not the issue, drop that thread early and maintain focus. |
| WSDM page limit conflict | 🟢 Low | Draft long, cut later. Keep full version on arXiv. |

---

## 10. Open Questions

> Bring to next standup.

1. **Retrieval vs. reasoning** — Is the primary failure on numeric questions a retrieval problem or a reasoning problem? Run 50-question pilot to determine.
2. **Table chunking** — Should table-containing chunks be a separate ablation condition, or folded into recursive chunking (C2)?
3. **Top-k value** — What is the right value of k for top-k retrieval? Must be fixed consistently across all 7 conditions.
4. **Evaluation scope** — Full 133 questions or a stratified sample by question type for compute efficiency?
5. **Missing PDFs** — Can the 9 failed PDFs be retrieved from SEC EDGAR before the first experiment run?
6. **Reranker model** — Cohere Rerank v3 or `BAAI/bge-reranker-v2-m3` for C3? Confirm with Natan.

---

*Anote AI Research Fellowship 2026 · Financial Document Retrieval Track*
