# Results

Every experiment run is saved to `results/runs/<condition>/<timestamp>/`:

- `results.csv` — final per-question results (predictions, ROUGE-L F1, exact match, latency)
- `progress.csv` — incremental checkpoint written every 10 questions, useful if a run is interrupted

Nothing is overwritten between runs: rerunning `run_c1.py` .. `run_c6.py` creates a new
timestamped folder under `results/runs/<condition>/`, so every run's raw output is kept
and past runs stay available for comparison.

## Comparing runs

Use `scripts/compare_runs.py`:

```
python scripts/compare_runs.py list                                     # show every saved run
python scripts/compare_runs.py list C6_hybrid                           # show runs for one condition
python scripts/compare_runs.py show C6_hybrid                           # summarize the latest C6_hybrid run
python scripts/compare_runs.py show C6_hybrid@20260722_084907_453875    # summarize a specific run
python scripts/compare_runs.py compare C6_hybrid C1_semantic            # compare latest runs across conditions
python scripts/compare_runs.py compare C6_hybrid@TS1 C6_hybrid@TS2      # compare two runs of the same condition
```

`compare` accepts any mix of condition names (resolves to the latest run), `condition@timestamp`,
or explicit paths to a run directory.
