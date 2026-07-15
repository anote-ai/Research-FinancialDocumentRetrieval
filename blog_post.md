# We Tested 6 Ways to Make AI Better at Reading Financial Documents. Here's What We Found.

*Anote AI Research Fellowship 2026*

---

If you've ever tried to get an AI to answer a question from a 10-K filing, you've probably been disappointed. Ask it something like "What was AMD's capital expenditure in FY2022?" and it either gets it wrong, hedges endlessly, or confidently makes up a number.

This isn't a model problem. It's a retrieval problem.

This summer I ran a controlled experiment to figure out exactly which retrieval techniques actually help — and which ones just add cost. The results surprised me.

---

## The Setup

I used [FinanceBench](https://arxiv.org/abs/2311.11944) — a benchmark of 150 real analyst questions over SEC filings from publicly traded companies. Think 10-Ks, 10-Qs, and earnings transcripts. Questions like:

- *"What is the FY2018 capital expenditure amount for 3M?"*
- *"What drove operating margin change for 3M in FY2022?"*
- *"Is 3M a capital-intensive business based on FY2022 data?"*

The original paper showed GPT-4 gets **81% of these wrong** even when the answer is sitting in the document. I wanted to know: what actually fixes this?

I built a RAG pipeline and tested seven configurations — a baseline plus six variations — across all 133 questions for which I could get source PDFs. I measured two things for each: **accuracy** (ROUGE-L F1) and **cost per query in USD**.

---

## The Seven Conditions

| Condition | What Changed |
|-----------|-------------|
| C0 — Baseline | Fixed 512-token chunks, basic similarity search |
| C1 — Semantic chunking | Split on meaning boundaries instead of token count |
| C2 — Recursive chunking | Split on document structure (sections → paragraphs → sentences) |
| C3 — Reranking | Re-score top-20 chunks with a cross-encoder before generation |
| C4 — Metadata enrichment | Tag each chunk with company, year, and filing type |
| C5 — Query expansion (HyDE) | Generate a hypothetical answer, use it to search |
| C6 — Hybrid | Everything combined |

---

## The Results

Here's what happened:

| Condition | Overall F1 | Numeric F1 | Qualitative F1 |
|-----------|-----------|------------|----------------|
| C0 Baseline | 0.177 | 0.028 | 0.259 |
| C1 Semantic | 0.177 | 0.026 | 0.260 |
| C2 Recursive | 0.177 | 0.026 | 0.260 |
| C3 Reranking | 0.175 | 0.024 | 0.258 |
| C4 Metadata | 0.170 | 0.024 | 0.251 |
| C5 Query Expansion | 0.167 | 0.017 | 0.250 |

None of the techniques improved on the baseline in any meaningful way.

I'll be honest — this wasn't what I expected. But it turned out to be the most interesting finding of the whole study.

---

## What's Actually Going On

The headline number — overall F1 of 0.177 — hides the real story. Break it down by question type and you see something striking:

**Qualitative questions** (things like "what drove the margin change?" or "is this company capital-intensive?"): F1 around **0.255–0.263** across all conditions. The pipeline retrieves relevant prose passages reasonably well.

**Numeric questions** (things like "what was the capex in USD millions?"): F1 of **0.017–0.028** across all conditions. Near-total failure.

That's a 10x gap. And none of our techniques closed it.

---

## Why Numeric Questions Are So Hard

When I manually inspected the failure cases, the pattern was clear: the model wasn't hallucinating wrong numbers. It was saying *"I can't find this in the provided context"* — because the right table passage genuinely wasn't in the top-10 retrieved chunks.

The problem is how standard RAG pipelines handle financial tables.

A cash flow statement looks something like this in a raw PDF:

```
Purchases of property, plant and equipment
                    (1,577)        (1,373)        (1,229)
```

When you chunk this with a fixed 512-token splitter, you often get the row label in one chunk and the values in another. Or the column headers in one chunk and the data rows in three others. The semantic meaning of the table is destroyed.

Semantic chunking and recursive chunking don't fix this — they still split on text boundaries, not on table structure. And a cross-encoder reranker can only rerank what was retrieved in the first place. If the right chunk was never in the top-20, reranking can't help.

**The problem isn't which chunks get ranked first. It's that the right chunk doesn't exist.**

---

## The Cost Angle

This is where the study gets practically useful. If none of the techniques improve accuracy, they all have the same F1 — but very different costs and latencies:

| Condition | Relative Cost | Latency |
|-----------|--------------|---------|
| C0 Baseline | 1x | 12.3s |
| C1 Semantic | ~1x | 11.2s |
| C2 Recursive | ~1x | 11.2s |
| C3 Reranking | ~2.5x | 35.3s |
| C4 Metadata | ~1.2x | 10.8s |
| C5 Query Expansion | ~2x | 13.2s |

If you're deploying a financial RAG system today, **the cheapest option is also the most accurate option**. Reranking adds 3x the latency at higher cost with no F1 gain. Query expansion actually makes things slightly worse on numeric questions while adding cost.

---

## What This Means for the Field

Most RAG papers test their techniques on general-domain benchmarks where chunking doesn't matter much — the relevant passage is usually a self-contained prose paragraph, not a row in a financial table.

Financial documents are different. The failure mode here isn't semantic retrieval quality — it's structural. Standard techniques assume that if you retrieve the right *region* of the document, the answer will be there. For financial tables, that assumption breaks.

What would actually help:

- **Table-aware parsing** — treating each table as a structured object, not a text block
- **Cell-level extraction** — pulling specific rows and columns rather than chunks
- **Schema-aware retrieval** — understanding that "capital expenditure" appears under "Cash flows from investing activities" in a cash flow statement
- **Financial-domain embeddings** — models trained on SEC filings rather than general web text

These are harder to implement than swapping a chunking strategy, but they're what the problem actually requires.

---

## The Takeaway for Practitioners

If you're building a RAG system over financial documents right now:

1. **Don't expect chunking strategy to matter much** — fixed, semantic, and recursive all perform similarly on this benchmark
2. **Skip reranking and query expansion** — they add cost and latency without meaningful accuracy gains on numeric questions
3. **The real problem is table structure** — invest engineering time in table-aware parsing before optimizing retrieval
4. **Qualitative questions are tractable** — F1 around 0.26 is still not great, but the pipeline works reasonably well for prose-based questions. Numeric QA requires a fundamentally different approach.

---

## What's Next

This study establishes the baseline picture. The natural next steps are:

- Testing table-aware chunking strategies that treat each table as a single unit
- Evaluating financial-domain-specific embedding models
- Extending to the full 10,231-question FinanceBench benchmark for stronger statistical power
- Building a cell-level extraction pipeline for numeric QA

The full paper is being submitted to WSDM 2027. The code and results will be open-sourced on GitHub at [github.com/anote-ai/Research-FinancialDocumentRetrieval](https://github.com/anote-ai/Research-FinancialDocumentRetrieval).

---

*Elaine Hong is an AI Research Fellow at Anote AI (Summer 2026) and a student at Cornell University studying Operations Research and Information Engineering. This research was supervised by Natan Vidra, CEO of Anote AI.*

*Questions or feedback? Reach out on [LinkedIn](https://linkedin.com/company/anote-ai/) or open an issue on GitHub.*
