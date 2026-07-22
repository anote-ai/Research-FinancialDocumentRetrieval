# We Tested 7 Ways to Make AI Better at Reading Financial Documents. Here's What We Found.

*Anote AI Research Fellowship 2026*

---

If you've ever tried to get an AI to answer a question from a 10-K filing, you've probably been disappointed. Ask it something like "What was AMD's capital expenditure in FY2022?" and it either gets it wrong, hedges endlessly, or confidently makes up a number.

This isn't a model problem. It's a retrieval problem.

This summer I ran a controlled experiment to figure out which retrieval techniques actually help on financial documents. I tested seven configurations on the FinanceBench benchmark — 133 real analyst questions over SEC filings. The results were not what I expected, and the most surprising finding came from the metric I almost didn't report.

---

## The Setup

[FinanceBench](https://arxiv.org/abs/2311.11944) is a benchmark of 150 analyst questions over real SEC filings from publicly traded companies: 10-Ks, 10-Qs, and earnings transcripts. Questions like:

- *"What is the FY2018 capital expenditure amount for 3M?"*
- *"What drove operating margin change for 3M in FY2022?"*
- *"Is 3M a capital-intensive business based on FY2022 data?"*

The original paper showed GPT-4 gets **81% of these wrong** even when the answer is in the document. I built a RAG pipeline and tested seven configurations, measuring both ROUGE-L F1 (standard overlap metric) and Exact Match (does the correct answer appear anywhere in the output) for each.

---

## The Seven Conditions

| Condition | What Changed |
|-----------|-------------|
| C0 — Baseline | Fixed 512-token chunks, basic similarity search |
| C1 — Semantic chunking | Split on meaning boundaries instead of token count |
| C2 — Recursive chunking | Split on document structure at 2,000-token max |
| C3 — Reranking | Re-score top-20 chunks with a cross-encoder |
| C4 — Metadata enrichment | Tag each chunk with company, year, filing type |
| C5 — Query expansion (HyDE) | Generate a hypothetical answer, use it to search |
| C6 — Hybrid | Everything combined |

---

## The Results

| Condition | ROUGE-L F1 | Exact Match | Latency | Numeric F1 | Numeric EM |
|-----------|-----------|------------|---------|-----------|-----------|
| C0 Baseline | 0.177 | 0.128 | 12.3s | 0.028 | 0.277 |
| **C1 Semantic** | **0.172** | **0.135** | 37.0s | 0.021 | **0.340** |
| C2 Recursive | 0.177 | 0.120 | 11.2s | 0.026 | 0.277 |
| C3 Reranking | 0.175 | 0.128 | 35.3s | 0.024 | 0.319 |
| C4 Metadata | 0.170 | 0.105 | 10.8s | 0.024 | 0.128 |
| C5 Query Exp. | 0.167 | 0.098 | 13.2s | 0.017 | 0.098 |
| **C6 Hybrid** | **0.179** | 0.098 | 49.8s | 0.018 | 0.234 |

The total ROUGE-L F1 range across all seven conditions is **0.012**. Nothing moves the needle on the standard metric.

But look at the Exact Match column for numeric questions. C1 (semantic chunking) achieves **34.0%** — the highest of any condition — while having the **lowest** overall ROUGE-L F1. That's the opposite of what you'd expect, and it changes the entire practical conclusion.

---

## Why the Standard Metric Is Misleading You

ROUGE-L measures how much the predicted text overlaps with the gold answer. For a question with gold answer "24.26", a response like:

> *"The fixed asset turnover ratio is calculated as revenue divided by average PP&E: $6,489M divided by $267.45M = 24.26"*

gets ROUGE-L approximately 0.04 — the 5-character gold answer has minimal longest-common-subsequence overlap with a 60-word explanation. But Exact Match = 1.0 because "24.26" appears in the output.

GPT-4o explains its reasoning. ROUGE-L penalizes this. Exact Match doesn't.

**ROUGE-L understates model performance on numeric questions by 10 to 16 times across all conditions.** A practitioner selecting retrieval conditions by ROUGE-L alone would reject semantic chunking (0.172 F1, lowest) and select the hybrid (0.179 F1, highest) — arriving at the exact opposite of the correct recommendation for numeric accuracy.

---

## Why Semantic Chunking Is the Best Condition for Numeric Questions

Semantic chunking splits documents on meaning boundaries instead of arbitrary token counts. When it encounters a financial table, it's more likely to keep the row label and its value together in a single chunk, because they're semantically related. Fixed 512-token splitting doesn't know or care about this relationship.

When the model receives a chunk that includes both "Purchases of property, plant and equipment" and "(1,577)", it can compute the correct answer. The row label and value are in the same context. Under fixed-token chunking, they're often in different chunks — or different chunks entirely.

This is why C1 achieves 34.0% Exact Match on numeric questions despite its low ROUGE-L score. The model is getting the right context more often. It's just wrapping its correct answer in explanation text that suppresses ROUGE-L.

---

## The Deeper Problem: Why Everything Fails on ROUGE-L

Even with semantic chunking's improved numeric Exact Match, the overall ROUGE-L picture is flat. The reason is structural.

SEC filing tables look like this when PDF text is extracted:

```
                                    2022        2021        2020
Capital expenditures             (1,577)     (1,373)     (1,229)
```

When you split this across chunk boundaries, you separate the row label from its values, and the column headers from the data rows. A similarity search for "capital expenditure 2022" finds the row label chunk. Not the value chunk. The answer doesn't exist as a retrievable unit.

No reranker can fix this — you can't rerank a value that was never in the top-20. No hybrid BM25+dense retrieval fixes it — the value is still in a different chunk. No metadata tag fixes it. No query expansion fixes it.

**The ROUGE-L ceiling of 0.017 to 0.028 on numeric questions is set by structural chunking failure, not model capability or retrieval technique.**

---

## What Each Technique Actually Did

**C1 (Semantic chunking):** Best for numeric Exact Match (34.0%). Lowest overall ROUGE-L (0.172). Highest non-reranking latency (37.0s). Use this when correct numerical answers are your priority.

**C2 (Recursive chunking):** Matches baseline exactly on ROUGE-L (0.177) at slightly lower latency (11.2s). Safe swap for baseline with no downside.

**C3 (Reranking):** Second-best numeric Exact Match (31.9%). ROUGE-L near-unchanged. Adds 23 seconds of latency per query. Use when numeric EM matters and you can tolerate the latency.

**C4 (Metadata):** Worse on both metrics than baseline. The metadata prefix shifts query embeddings away from content-specific terms. Don't use it.

**C5 (Query expansion):** Worst performance of any condition on both metrics. GPT-4o's hypothetical answer contains plausible-but-wrong numbers that retrieve the wrong passages. Avoid.

**C6 (Hybrid):** Best ROUGE-L (0.179, barely). Worst Exact Match (0.098). At 4x baseline latency, the marginal ROUGE-L gain is not worth it, and the Exact Match hit is severe.

---

## The Practical Answer

**If you care about correctly answering numeric financial questions:** Use semantic chunking (C1). You get 34.0% correct numeric answers vs. 27.7% for baseline, at 3x the latency. Worth it for high-stakes financial analysis.

**If you care about overall ROUGE-L or need low latency:** Use the baseline (C0) or recursive chunking (C2). Equal ROUGE-L, minimum cost.

**For everyone:** Avoid query expansion (C5) and metadata enrichment (C4) — they make things worse. And know that any retrieval optimization you add is working around the real problem, not solving it.

The real solution is table-aware document parsing — treating each financial table as a structured object with preserved row-column associations rather than a block of text. That's the next experiment.

---

## Key Takeaways

1. **Report Exact Match alongside ROUGE-L for financial QA.** ROUGE-L will tell you semantic chunking is your worst option when it's actually your best option for numeric accuracy. The metrics give opposite rankings.

2. **Semantic chunking is the best technique for numeric financial questions** — not because it solves the structural problem, but because it's less bad. It more often keeps row labels and values in the same chunk.

3. **Retrieval optimization is not the bottleneck.** Table-aware parsing is. The ROUGE-L ceiling on numeric questions is set by structural chunking failure; no retrieval technique in this study breaks through it.

4. **Avoid query expansion and metadata enrichment** on financial documents. Both decrease accuracy across all metrics while adding cost.

---

The full paper is being submitted to the FinNLP workshop at EMNLP 2026. Code and results are at [github.com/anote-ai/Research-FinancialDocumentRetrieval](https://github.com/anote-ai/Research-FinancialDocumentRetrieval).
