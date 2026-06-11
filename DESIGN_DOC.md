# Research Design Document: Financial Document Retrieval

## Vision Statement

Build the definitive benchmark for evaluating retrieval systems on financial documents — establishing that general-purpose retrieval metrics fail on tables, numeric claims, and cross-document reasoning, and providing the community with **FinRAG-Bench**: a dataset, metric suite, and leaderboard that becomes the standard reference for production financial AI systems.

---

## Problem Statement & Novelty

Financial documents (10-Ks, earnings transcripts, analyst reports, regulatory filings) are structurally and semantically distinct from the web text on which general retrieval benchmarks (BEIR, MS MARCO) are built. Yet production RAG deployments for financial Q&A are routinely evaluated with NDCG and MRR computed over text-only passages. This creates a systematic blind spot:

1. **Table blindness**: Retrieval systems that cannot parse HTML/XBRL tables silently drop critical numeric evidence.
2. **Numeric hallucination**: Systems that retrieve plausible-sounding text may return the correct paragraph from the wrong fiscal year, producing confident but wrong numerical answers.
3. **Cross-document reasoning**: Analyst questions frequently require synthesizing data across multiple filings (e.g., comparing YoY revenue growth across three 10-Ks).
4. **Regulatory specificity**: SEC filings have mandatory section structures (Item 1A Risk Factors, Item 7 MD&A) that retrieval systems should exploit but typically ignore.

### Novel Contributions

| Contribution | Description |
|---|---|
| **FinRAG-Bench dataset** | 2,400 QA pairs from SEC EDGAR filings across 6 sectors, 5 document types, requiring table + text retrieval |
| **NHR metric** | Numeric Hallucination Rate: fraction of retrieved passages where numeric values differ from gold by >5% |
| **TRR metric** | Table Retrieval Recall: recall computed only over table-origin gold passages |
| **Cross-document synthesis score** | For multi-hop questions, fraction of required passages co-retrieved |
| **Section-aware chunking** | Chunking strategy exploiting SEC filing structure (Item-level segmentation) |

---

## Research Objectives

1. Quantify the **table-blindness gap**: how much does retrieval performance drop when gold evidence is in tables vs. prose?
2. Measure the **NHR** of leading retrieval systems on financial documents.
3. Demonstrate that **section-aware chunking** improves recall by ≥10 percentage points over sliding-window chunking.
4. Show that **cross-document synthesis** requires fundamentally different retrieval strategies than single-document Q&A.
5. Establish FinRAG-Bench as the standard evaluation suite, validated by financial domain experts.

---

## Dataset Construction

### Source Corpus
- **SEC EDGAR** filings: 10-K, 10-Q, 8-K, earnings call transcripts, proxy statements
- **Coverage**: 400 companies × 3 years = ~1,200 documents per type
- **Sectors**: Technology, Healthcare, Financial Services, Energy, Consumer, Industrials (6 sectors)
- **Total corpus size**: ~6,000 documents, ~180M tokens

### Question Generation Protocol

```
Phase 1: Seed question generation
  - Financial analysts (5 domain experts) generate 480 seed questions
  - Question types: point lookup (numeric), trend analysis, comparison, risk identification, synthesis
  
Phase 2: Adversarial augmentation
  - For each seed question, generate 3 distractor passages (same section, different fiscal year)
  - Human annotators verify gold passage labels
  
Phase 3: Cross-document questions
  - 400 questions requiring evidence from ≥2 documents
  - Gold evidence spans provided at passage level

Final: 2,400 questions (2,000 single-doc, 400 cross-doc)
```

### Question Type Distribution

| Type | Count | Example |
|---|---|---|
| Point lookup (numeric) | 600 | "What was AAPL's FY2023 R&D expense?" |
| Trend analysis | 400 | "How did gross margin trend from 2021–2023?" |
| YoY comparison | 400 | "Which segment grew fastest in FY2022?" |
| Risk identification | 400 | "What liquidity risks does Item 1A disclose?" |
| Cross-doc synthesis | 400 | "Compare capex intensity across FAANG in FY2023" |
| Regulatory compliance | 200 | "Does MD&A address all required disclosures?" |

---

## Systems Under Evaluation

| System | Type | Table Support | Notes |
|---|---|---|---|
| BM25 | Sparse | No | Baseline |
| DPR | Dense | No | Standard dense |
| ColBERT-v2 | Late interaction | No | Strong dense baseline |
| E5-large | Dense | Partial | SOTA general |
| FinBERT-retrieval | Dense, finance-tuned | No | Domain-adapted |
| Unstructured.io + E5 | Dense + table parser | Yes | Table-aware |
| LlamaParse + ColBERT | Dense + table parser | Yes | Commercial parser |
| GPT-4o retrieval | LLM-based | Yes | Expensive baseline |
| Section-aware BM25 | Sparse + structure | Yes (sections) | Our proposed method |

---

## Experimental Design

### Baseline Experiment (Experiment 0)
**Protocol**: Run BM25 and DPR (text-only, sliding-window chunking, 512 tokens, 50% overlap) on all 2,400 questions. Compute NDCG@10, MRR, Recall@5.

**Expected result**: NDCG@10 ≈ 0.41 (BM25), 0.47 (DPR). These numbers establish the performance floor and replicate prior financial QA retrieval baselines.

---

### Experiment 1: Table-Blindness Gap
**Hypothesis**: Retrieval performance on table-origin gold passages is ≥20 pp lower than on prose-origin passages for all text-only systems.

**Protocol**:
1. Label each gold passage as `table`, `prose`, or `mixed`.
2. Compute NDCG@10 separately for each gold-passage type.
3. Compute **TRR** (Table Retrieval Recall) across all systems.

**Expected results**:
- Text-only systems: TRR ≈ 0.28 vs. prose recall ≈ 0.61 (gap: 33 pp)
- Table-aware systems (Unstructured.io pipeline): TRR ≈ 0.59 (gap closes to 8 pp)
- Key finding: BM25 keyword matching partially works on numeric strings in tables; dense models fail more severely

```python
# TRR computation
def compute_TRR(retrieved_passages, gold_passages):
    table_golds = [p for p in gold_passages if p.origin == 'table']
    if not table_golds:
        return None
    retrieved_ids = {p.id for p in retrieved_passages}
    return len([g for g in table_golds if g.id in retrieved_ids]) / len(table_golds)
```

---

### Experiment 2: Numeric Hallucination Rate
**Hypothesis**: NHR > 0.30 for all systems not using table-aware parsing, even when the retrieved passage is ranked #1.

**Protocol**:
1. For each numeric question, extract the numeric value from the top-retrieved passage.
2. Compare to gold value; flag as hallucination if relative deviation > 5%.
3. Measure NHR = hallucinations / numeric questions.
4. Stratify by fiscal year: same-company, wrong-year retrieval is the primary error mode.

**Expected results**:

| System | NHR (overall) | NHR (wrong-year errors) |
|---|---|---|
| BM25 | 0.38 | 0.22 |
| DPR | 0.41 | 0.28 |
| ColBERT-v2 | 0.35 | 0.24 |
| FinBERT-retrieval | 0.29 | 0.18 |
| Section-aware BM25 | 0.21 | 0.11 |

- Primary finding: wrong-year retrieval accounts for ~60% of numeric errors; temporal metadata filtering reduces NHR by ~40%.

---

### Experiment 3: Section-Aware Chunking
**Hypothesis**: Chunking at SEC Item boundaries (Item 1A, Item 7, Item 8) improves Recall@10 by ≥10 pp vs. sliding-window on MD&A questions.

**Protocol**:
1. Implement three chunking strategies: (a) sliding window 512, (b) sliding window 256, (c) SEC Item-boundary chunking.
2. Run BM25, DPR, ColBERT-v2 with each strategy.
3. Compute Recall@10 stratified by question type (MD&A questions vs. others).

**Expected results**:
- Item-boundary chunking: Recall@10 ≈ 0.71 for MD&A questions vs. 0.58 for sliding-window (13 pp improvement)
- Explanation: Item-boundary chunks preserve the complete context of regulatory disclosures without splitting across section headers
- Neutral effect on numeric lookup questions (table structure more important than section boundaries)

---

### Experiment 4: Cross-Document Synthesis
**Hypothesis**: Standard retrieval systems achieve <40% co-retrieval of required passages for cross-document questions; a query decomposition approach improves this to >65%.

**Protocol**:
1. Evaluate all 400 cross-document questions with standard retrieval.
2. Implement query decomposition: LLM decomposes multi-hop question into sub-queries, retrieve independently, merge.
3. Compute co-retrieval rate = fraction of questions where all required passages are retrieved within top-10.
4. Measure answer quality with final LLM reader (GPT-4o).

**Expected results**:
- Standard retrieval: co-retrieval rate ≈ 0.31
- Query decomposition (LLM-based): co-retrieval rate ≈ 0.67 (+36 pp)
- Query decomposition (rule-based): co-retrieval rate ≈ 0.52 (+21 pp)
- Answer quality (LLM reader + decomposition): +18 pp accuracy vs. standard retrieval reader

---

### Experiment 5: Cost-Quality Analysis
**Hypothesis**: A section-aware BM25 + FinBERT reranker achieves >85% of GPT-4o retrieval quality at 1/50th the cost.

**Protocol**:
1. Compute NDCG@10 and cost-per-query for all systems.
2. Measure cost: API costs + compute time.
3. Plot cost-quality Pareto frontier.

**Expected results**:
- GPT-4o retrieval: NDCG@10 ≈ 0.74, cost ≈ $0.08/query
- Section-aware BM25 + FinBERT reranker: NDCG@10 ≈ 0.66 (89% of GPT-4o), cost ≈ $0.001/query
- Cost efficiency winner: hybrid pipeline achieves 66× cost reduction with <12% quality loss

---

## Expected Results Summary

| Metric | Baseline (BM25) | Best System | Improvement |
|---|---|---|---|
| NDCG@10 (overall) | 0.41 | 0.74 (GPT-4o) | +80% |
| TRR (table recall) | 0.28 | 0.61 (table-aware) | +118% |
| NHR (numeric hallucination) | 0.38 | 0.21 (section-aware) | −45% |
| Cross-doc co-retrieval | 0.31 | 0.67 (decomposition) | +116% |
| Cost efficiency | $0.001/q | $0.001/q (BM25) | — |

**Primary claim**: Text-only retrieval systems have NHR > 0.35 and TRR < 0.35 on financial documents, making them unsuitable for production financial AI without table-aware parsing and temporal metadata filtering.

---

## Why This Matters

**For researchers**: FinRAG-Bench fills a critical gap — financial documents are among the highest-stakes RAG deployments but have no dedicated retrieval benchmark.

**For practitioners**: NHR and TRR give actionable metrics for diagnosing production failures in financial AI products (chatbots, analyst assistants, compliance tools).

**For industry**: Wrong-year numeric retrieval is the #1 failure mode — a finding that directly informs engineering priorities for Bloomberg Terminal AI, Morgan Stanley's AI assistant, and similar products.

**Market**: $1.2T financial services sector is investing heavily in AI; a trusted benchmark could become the evaluation standard for procurement and compliance.

---

## Implementation Plan

```
research-financialdocumentretrieval/
├── data/
│   ├── corpus/          # SEC EDGAR documents (processed)
│   ├── questions/       # QA pairs with gold passage labels
│   └── annotations/     # Table/prose/mixed labels
├── parsers/
│   ├── xbrl_parser.py   # XBRL table extraction
│   ├── sec_chunker.py   # Item-boundary chunking
│   └── pdf_parser.py    # PDF table extraction
├── retrieval/
│   ├── bm25_baseline.py
│   ├── dense_retrieval.py
│   └── hybrid_pipeline.py
├── metrics/
│   ├── nhr.py           # Numeric Hallucination Rate
│   ├── trr.py           # Table Retrieval Recall
│   └── cross_doc.py     # Cross-document co-retrieval
├── experiments/
│   ├── exp1_table_blindness.py
│   ├── exp2_nhr.py
│   ├── exp3_section_chunking.py
│   ├── exp4_cross_doc.py
│   └── exp5_cost_quality.py
└── leaderboard/
    └── submit.py
```

---

## Timeline

| Phase | Duration | Deliverable | Owner |
|---|---|---|---|
| Data collection & annotation | 8 weeks | 2,400 QA pairs labeled | Data team |
| Parser development | 4 weeks | XBRL + section-aware chunker | Eng team |
| Baseline experiments | 3 weeks | Exp 0–1 results | Research team |
| Novel experiments | 5 weeks | Exp 2–5 results | Research team |
| Paper writing | 4 weeks | ACL 2026 submission | All |
| Leaderboard launch | 2 weeks | Public benchmark live | Eng team |

**Target venue**: ACL 2026 Industry Track (deadline: ~February 2026)

---

## Open Questions & Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| EDGAR scraping rate limits | Medium | Use official bulk download API |
| Domain expert annotation cost | High | Partner with financial data vendors |
| XBRL parsing quality | Medium | Validate against manual spot-checks |
| Contamination in FinBERT training data | Medium | Use post-2024 filings only |
| Benchmark gaming / leaderboard saturation | Low | Add held-out private test set |

---

## Related Issues

- Reproducibility package
- Statistical rigor & significance testing
- Ethics: financial data privacy and fair use
- Related work audit: FinQA, TAT-QA, DocNLI
- Product integration: private chatbot improvements
