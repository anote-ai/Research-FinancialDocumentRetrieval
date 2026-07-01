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
5. [Preliminary Results](#5-preliminary-results)
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
| ✅ | All pipeline dependencies installed | langchain, faiss-cpu, sentence-transformers, pypdf, rouge-score, rank_bm25, openai, langchain-openai |
| ✅ | results/ directory created | Progress saves every 10 questions |
| ✅ | End-to-end pipeline validated | test_pipeline.py ran successfully on 5 questions |
| 🔄 | C0 baseline evaluation running | `run_c0.py` in progress — 133 questions, GPT-4o |
| ⚠️ | 9 PDFs failed to download | Adobe 2015/16/17/22, J&J, MGM — 17 questions excluded |
| ⏳ | C1–C6 ablation conditions | Pending C0 completion |

### Environment Details

| Detail | Value |
|--------|-------|
| OS | Windows |
| Python version | 3.11 (Anaconda) |
| Conda environment | `findocretrieval` |
| Working directory | `C:\Projects\Research-FinancialDocumentRetrieval` |
| LLM provider | OpenAI (GPT-4o, temperature=0) |
| Embedding model | `all-MiniLM-L6-v2` (HuggingFace, local, no API key) |
| Vector store | FAISS (local) |
| Chunking (C0) | TokenTextSplitter, chunk_size=512, chunk_overlap=50 |
| Retrieval k | k=10 |

### Key Import Fixes Applied

During setup the following import changes were required due to LangChain version updates:

| Old Import | New Import |
|-----------|-----------|
| `from langchain.text_splitter import TokenTextSplitter` | `from langchain_text_splitters import TokenTextSplitter` |
| `from langchain_community.embeddings import HuggingFaceEmbeddings` | `from langchain_huggingface import HuggingFaceEmbeddings` |

### Failed PDF Downloads

The following 9 documents need to be manually retrieved from [SEC EDGAR](https://www.sec.gov/cgi-bin/browse-edgar):

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

### 3.3 Question Type Breakdown

- **Metrics-generated** — multi-step numeric calculations (e.g. fixed asset turnover ratio, 3-year average capex %). Requires finding multiple numbers and computing a formula.
- **Novel-generated** — extractive and interpretive questions requiring locating specific passages.
- **Boolean** — yes/no questions about company policies, dividends, etc.

> **Secondary Research Question:** Is the primary failure mode on numeric questions a retrieval problem (correct chunk never retrieved) or a reasoning problem (correct chunk retrieved, model miscalculates)? Early C0 results suggest it is primarily a **retrieval failure** — the right passage is not making it into the top-k chunks.

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
| **Fixed (C0)** | 512-token chunks, 50-token overlap — `TokenTextSplitter` from `langchain_text_splitters` |
| **Semantic (C1)** | Split on semantic similarity boundaries using sentence embeddings; variable chunk size |
| **Recursive (C2)** | Hierarchical split: section headers → paragraphs → sentences; treats tables as single units |

#### Reranking (C3)

After retrieving the top-k chunks, a cross-encoder re-scores and reorders them, keeping the top-5 to pass to the generator.

- Candidate models: Cohere Rerank v3, `BAAI/bge-reranker-v2-m3`
- Final model choice to be confirmed

#### Metadata Enrichment (C4)

Each chunk tagged with: `company`, `fiscal_year`, `doc_type`, `section_header`, `page_number`. Metadata prepended to chunk text at index time and used as retrieval filters.

#### Query Expansion (C5)

- **HyDE** — generate a hypothetical answer, embed it, retrieve against that embedding alongside the original query
- **Multi-query** — generate 3 paraphrases, retrieve against each, union results before reranking

### 4.3 Evaluation Metrics

| Metric | Description | Why It Matters |
|--------|-------------|----------------|
| **Token-level F1 (ROUGE-L)** | Overlap between predicted and gold answer spans | Primary accuracy signal |
| **Exact Match (EM)** | Binary correctness — predicted contains gold answer | Critical for numeric questions |
| **Retrieval Recall@k** | Fraction of gold evidence in top-k retrieved | Isolates retrieval from generation |
| **Cost per Query (USD)** | Total API spend divided by number of questions | The cost-frontier differentiator |
| **Latency (p50/p95)** | Query response time in seconds | Practical deployment constraint |

### 4.4 Implementation

| Component | Implementation |
|-----------|---------------|
| Embedding model | `all-MiniLM-L6-v2` (HuggingFace, local) |
| Generator model | GPT-4o (OpenAI, temperature=0) |
| Vector store | FAISS (local) |
| Sparse retrieval | `rank_bm25` for BM25 in hybrid condition (C6) |
| Framework | LangChain + LangChain-community + LangChain-text-splitters |
| Evaluation harness | `run_c0.py` — ROUGE-L F1 + exact match, saves to `results/` |
| Source documents | 75 PDFs in `data/pdfs/` — 133 of 150 questions covered |

---

## 5. Preliminary Results

### 5.1 C0 Baseline — In Progress

> **Status:** `run_c0.py` currently running on all 133 questions. Results save to `results/c0_baseline.csv`. Update this section when complete.

**Final C0 results (fill in when run completes):**

| Metric | C0 Baseline |
|--------|-------------|
| Mean ROUGE-L F1 | `[PENDING]` |
| Mean Exact Match | `[PENDING]` |
| Mean Latency | `[PENDING]` |
| Mean Cost/Query | `[PENDING]` |
| F1 — metrics-generated | `[PENDING]` |
| F1 — novel-generated | `[PENDING]` |
| F1 — boolean | `[PENDING]` |

### 5.2 Early Signal from First 10 Questions

The first 10 questions of the C0 run reveal a clear pattern before full results are in:

| # | Company | Question Type | Gold Answer | ROUGE-L F1 | Exact Match | Observation |
|---|---------|--------------|-------------|------------|-------------|-------------|
| 1 | 3M | Numeric (capex) | $1577.00 | 0.000 | 0.0 | Retrieval failure — right chunk not in top-10 |
| 2 | 3M | Numeric (PP&E) | $8.70 | 0.105 | 0.0 | Partial hit — related text retrieved |
| 3 | 3M | Boolean | Multi-sentence | 0.101 | 0.0 | Wrong answer — model said Yes, gold says No |
| 4 | 3M | Qualitative | Multi-sentence | 0.173 | 0.0 | Partial overlap on prose question |
| 5 | 3M | Qualitative | Multi-sentence | 0.229 | 0.0 | Better on segment analysis question |
| 6 | 3M | Qualitative | Multi-sentence | 0.105 | 0.0 | Retrieval miss on liquidity question |
| 7 | 3M | Extractive | List of items | 0.200 | 0.0 | Partial — model said not found, gold is specific list |
| 8 | 3M | Boolean | Multi-sentence | 0.275 | 0.0 | Best score so far — dividend question |
| 9 | Activision | Calculated ratio | 24.26 | 0.036 | 1.0 | Model calculated correctly despite low F1 |
| 10 | Activision | Calculated % | 1.9% | 0.000 | 0.0 | Context missing — cash flow not retrieved |

### 5.3 Early Findings

**Finding 1 — Retrieval failure is the primary bottleneck on numeric questions.** Questions 1 and 10 show F1 of 0.000 because the cash flow statement chunks were not in the top-10 retrieved. The model correctly said it couldn't find the answer rather than hallucinating. This confirms the secondary research question: the problem is retrieval, not generation.

**Finding 2 — Qualitative questions perform better than numeric.** Questions 4, 5, and 8 (prose-based) score 0.173–0.275 F1, while pure numeric questions score 0.000–0.105. This split is expected and will be a key analytical finding in the paper.

**Finding 3 — Exact match works for calculated ratios.** Question 9 has EM=1.0 despite low ROUGE-L F1 — the model found and calculated the right ratio (24.26) even though the answer phrasing didn't match. This suggests EM may be a better metric than F1 for metrics-generated questions.

**Finding 4 — 512-token fixed chunking splits financial tables badly.** Retrieved chunks show partial table rows and disconnected numbers. This directly motivates C2 (recursive chunking with tables as single units).

---

## 6. Related Work

### Direct Predecessors

- **Vidra et al. (2024)** — [arXiv:2404.07221](https://arxiv.org/abs/2404.07221). Anote's prior work and direct predecessor. Tests chunking, query expansion, metadata, reranking, and embedding fine-tuning on FinanceBench. This paper extends it with more rigorous ablation design, explicit cost measurement, and updated models.
- **Islam et al. (2023)** — [FinanceBench, arXiv:2311.11944](https://arxiv.org/abs/2311.11944). Primary evaluation dataset. GPT-4-Turbo fails 81% of questions with basic retrieval. Core motivation.

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
| Jun 9–23 | Setup & Scoping | Repo running, data downloaded, API key confirmed, abstract drafted, C0 running | **Jun 23 — Abstract due ✓** |
| Jun 23–Jul 14 | Lit Review & Baseline | Full lit review; C0 results in; C1–C3 running | **Jul 14 — Methodology done** |
| Jul 14–21 | Experiments | All 7 conditions complete; results table and cost-frontier plot generated | **Jul 21 — Full draft due** |
| Jul 21–28 | Writing — Draft 1 | Full paper written and shared for peer review | **Jul 28 — Peer feedback** |
| Jul 28–Aug 4 | Revision | Statistical rigor, reproducibility package, ethics statement | **Aug 4 — Final polish** |
| Aug 5–10 | Submission Prep | ACM formatting; arXiv preprint submitted; fellowship presentation | **Aug 10 — arXiv live** |
| Aug 17 | WSDM Abstract | Abstract submitted to WSDM 2027 | **Hard deadline** |
| Aug 24 | WSDM Full Paper | Full paper submitted to WSDM 2027 | **Hard deadline** |

---

## 9. Risks & Mitigations

| Risk | Level | Mitigation |
|------|-------|------------|
| Experiments take longer than expected | 🔴 High | C0 already running. Parallelize C1–C6 once C0 finishes. |
| 9 missing PDFs remain unavailable | 🟡 Medium | Proceeding with 133/150 questions. Will note exclusion in paper. |
| Overlap with Patel et al. 2026 | 🟡 Medium | Cost measurement and chunking ablation are our explicit differentiators. |
| Windows environment compatibility | 🟡 Medium | Resolved — import fixes applied and documented in Section 2. |
| Table structure bottleneck is negligible | 🟢 Low | Early results suggest tables ARE a bottleneck (Finding 4). C2 well motivated. |
| WSDM page limit conflict | 🟢 Low | Draft long, cut later. Keep full version on arXiv. |

---

## 10. Open Questions

> Bring to next standup.

1. **Full C0 results** — what is the final mean F1 across all 133 questions? What is the breakdown by question type?
2. **Cost tracking** — need to add OpenAI API usage logging to measure cost per query. Check platform.openai.com usage dashboard after C0 run.
3. **Top-k value** — k=10 is the current setting. Should this change for different conditions?
4. **Reranker model** — Cohere Rerank v3 or `BAAI/bge-reranker-v2-m3` for C3?
5. **Missing PDFs** — can the 9 failed PDFs be retrieved from SEC EDGAR to recover the 17 excluded questions?
6. **Table chunking** — early results confirm tables are being split badly by fixed chunking. Should C2 explicitly treat every table as a single chunk unit?

---

*Anote AI Research Fellowship 2026 · Financial Document Retrieval Track*