# Independent ablation: chunking and hybrid retrieval on FinanceBench

The full write-up, with related work, methodology, discussion, and
limitations, is in `Uzoama_FinancialRAG_Ablation.pdf`. This README is a
shorter summary plus instructions for reproducing the results.

This folder contains a standalone ablation study run on a subset of the
FinanceBench sample, comparing three retrieval conditions against each other:

- **C0**: fixed word-based chunking (baseline)
- **C1**: semantic-boundary chunking (paragraph/sentence-based)
- **C6**: a hybrid pipeline: structure-aware recursive chunking, a BM25 +
  TF-IDF fusion retriever, metadata prefixes, HyDE-style query expansion,
  and a lexical reranking pass

Retrieval is built on TF-IDF and BM25 rather than neural embeddings and a
cross-encoder reranker, and Claude is used as the generation model across
all three conditions, so the comparison stays internally consistent. The
sample is 14 questions, roughly stratified across FinanceBench's question
types (domain-relevant, metrics-generated, novel-generated).

To guard against a single run being over-interpreted, the whole pipeline
was run twice, independently, with a second, separately written set of
answers scored the same way (Run 1 and Run 2 below).

## Results

Mean ROUGE-L F1, paired bootstrap (10,000 resamples) against the C0 baseline:

| Condition | Run 1 mean | Run 2 mean |
|---|---|---|
| C0 (baseline) | 0.1518 | 0.1453 |
| C1 (semantic) | 0.1166 | 0.1228 |
| C6 (hybrid) | 0.1277 | 0.1244 |

- C1 vs C0: Run 1 diff = -0.0352, 95% CI [-0.0730, -0.0020] (significant).
  Run 2 diff = -0.0225, 95% CI [-0.0495, +0.0008] (not significant, but the
  same direction and close to the boundary).
- C6 vs C0: not significant in either run.

Neither enhancement beats the fixed-chunking baseline in this setup. The
semantic-chunking shortfall is the more consistent finding across the two
runs; the hybrid shortfall is smaller and noisier.

Exact Match was also computed but is not very informative at this sample
size (most answers are short free-text spans rather than exact numeric
matches), so ROUGE-L F1 is the primary metric reported here.

## Files

- `Uzoama_FinancialRAG_Ablation.pdf`: the full paper.
- `pipeline.py`: loads a stratified 14-question sample from the
  repository's `financebench_sample.csv`, extracts PDF text from the
  repository's `data/pdfs/`, builds chunks for each of the three
  conditions, retrieves the top candidates per question, and writes
  `prudence_retrieved.json`.
- `run1_answers.py`, `run2_answers.py`: two independently written sets of
  answers for each question and condition, given only the retrieved
  context (no lookup of gold answers while writing them).
- `score.py`: scores either answer set against `prudence_retrieved.json`
  using ROUGE-L F1 and runs the paired bootstrap comparison. Usage:
  `python score.py run1` or `python score.py run2`.
- `results_run1.csv`, `results_run2.csv`: the per-question, per-condition
  scored results for each run, as reported above.

## Reproducing

From this folder:

```
python pipeline.py          # regenerates prudence_retrieved.json
python score.py run1        # scores run1_answers.py, prints means + bootstrap
python score.py run2        # scores run2_answers.py, prints means + bootstrap
```

`pipeline.py` reads `financebench_sample.csv` and `data/pdfs/` from the
repository root, so it needs to be run from inside a checkout of this
repository rather than standalone. Nine documents referenced in the full
FinanceBench sample are not present in `data/pdfs/` here (four Adobe 10-Ks,
four Johnson & Johnson filings, one MGM Resorts earnings release) and are
excluded from the sample pool; `AMD_2015_10K.pdf` is present but fails to
parse with both available PDF extractors and is skipped when encountered.

## Limitations

This is a small-sample, single-model check, not a claim that semantic or
hybrid retrieval never helps on financial documents generally. The lexical
retrieval substitutes and single generation model are meaningful departures
from a full neural-embedding, cross-encoder, multi-model setup, and the
14-question sample is too small to detect anything but a fairly large
effect. The value here is narrower: under this specific setup, two
independent runs agree that the added complexity of semantic and hybrid
retrieval did not pay off relative to the simplest baseline, which is worth
noting given how often such techniques are added on the assumption that
they will help.
