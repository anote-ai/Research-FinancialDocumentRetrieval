#!/usr/bin/env python3
"""Compare metrics across saved runs under results/runs/<condition>/<timestamp>/.

Every run_c*.py script writes its output to a fresh, timestamped folder
(see run_output.create_run_dir), so nothing is ever overwritten — this script
just reads that history back out for comparison.

Usage
-----
# List every saved run, newest first within each condition
python scripts/compare_runs.py list
python scripts/compare_runs.py list C6_hybrid

# Summarize a single run (defaults to the latest run of a condition)
python scripts/compare_runs.py show C6_hybrid
python scripts/compare_runs.py show C6_hybrid@20260722_084907_453875

# Compare two or more runs side by side
python scripts/compare_runs.py compare C6_hybrid C1_semantic
python scripts/compare_runs.py compare C6_hybrid@20260722_054233_592666 C6_hybrid
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / "results" / "runs"


@dataclass
class Run:
    condition: str
    timestamp: str
    path: Path

    @property
    def label(self) -> str:
        return f"{self.condition}@{self.timestamp}"

    @property
    def results_csv(self) -> Path:
        return self.path / "results.csv"


def list_runs(condition: str | None = None) -> list[Run]:
    """Every saved run with a results.csv, newest first within each condition."""
    if not RUNS_ROOT.exists():
        return []
    runs = []
    for cond_dir in sorted(RUNS_ROOT.iterdir()):
        if not cond_dir.is_dir():
            continue
        if condition and cond_dir.name != condition:
            continue
        for run_dir in sorted(cond_dir.iterdir(), reverse=True):
            if (run_dir / "results.csv").exists():
                runs.append(Run(cond_dir.name, run_dir.name, run_dir))
    return runs


def resolve_run(ref: str) -> Run:
    """Resolve a run reference.

    Accepts a condition name (-> latest run), "condition@timestamp" (-> that
    exact run), or a filesystem path to a run directory or its results.csv.
    """
    p = Path(ref)
    if p.exists():
        path = p.parent if p.name == "results.csv" else p
        return Run(path.parent.name, path.name, path)

    if "@" in ref:
        condition, timestamp = ref.split("@", 1)
        path = RUNS_ROOT / condition / timestamp
        if not (path / "results.csv").exists():
            raise SystemExit(f"No run found at {path}")
        return Run(condition, timestamp, path)

    matches = list_runs(ref)
    if not matches:
        raise SystemExit(f"No runs found for condition '{ref}' under {RUNS_ROOT}")
    return matches[0]


def summarize(run: Run) -> dict:
    df = pd.read_csv(run.results_csv)
    by_type = df.groupby("question_type")["rouge_f1"].mean().round(3).to_dict()
    return {
        "label": run.label,
        "n": len(df),
        "mean_rouge_f1": round(df["rouge_f1"].mean(), 4),
        "mean_exact_match": round(df["exact_match"].mean(), 4),
        "mean_latency_sec": round(df["latency_sec"].mean(), 2),
        "by_question_type": by_type,
    }


def cmd_list(args: argparse.Namespace) -> None:
    runs = list_runs(args.condition)
    if not runs:
        print("No runs found.")
        return
    current_condition = None
    for run in runs:
        if run.condition != current_condition:
            current_condition = run.condition
            print(f"\n{current_condition}")
        n = len(pd.read_csv(run.results_csv))
        print(f"  {run.timestamp}  ({n} rows)")


def cmd_show(args: argparse.Namespace) -> None:
    _print_summary(summarize(resolve_run(args.run)))


def cmd_compare(args: argparse.Namespace) -> None:
    summaries = [summarize(resolve_run(ref)) for ref in args.runs]
    _print_comparison(summaries)


def _print_summary(s: dict) -> None:
    print(f"\n{s['label']}")
    print(f"  N: {s['n']}")
    print(f"  Mean ROUGE-L F1:  {s['mean_rouge_f1']:.3f}")
    print(f"  Mean Exact Match: {s['mean_exact_match']:.3f}")
    print(f"  Mean Latency:     {s['mean_latency_sec']:.1f}s")
    print("  By question_type:")
    for qtype, f1 in sorted(s["by_question_type"].items()):
        print(f"    {qtype:<20} {f1:.3f}")


def _print_comparison(summaries: list[dict]) -> None:
    try:
        from rich.console import Console
        from rich.table import Table

        table = Table(title="Run Comparison")
        table.add_column("Run", style="cyan", overflow="fold")
        table.add_column("N", justify="right")
        table.add_column("ROUGE-L F1", justify="right", style="green")
        table.add_column("Exact Match", justify="right")
        table.add_column("Latency (s)", justify="right")
        for s in summaries:
            table.add_row(
                s["label"], str(s["n"]),
                f"{s['mean_rouge_f1']:.3f}", f"{s['mean_exact_match']:.3f}",
                f"{s['mean_latency_sec']:.1f}",
            )
        Console().print(table)

        qtypes = sorted({qt for s in summaries for qt in s["by_question_type"]})
        by_type_table = Table(title="Mean ROUGE-L F1 by question_type")
        by_type_table.add_column("question_type", style="cyan")
        for s in summaries:
            by_type_table.add_column(s["label"], justify="right", overflow="fold")
        for qt in qtypes:
            by_type_table.add_row(
                qt, *(f"{s['by_question_type'].get(qt, float('nan')):.3f}" for s in summaries)
            )
        Console().print(by_type_table)
    except ImportError:
        run_width = max(20, max(len(s["label"]) for s in summaries) + 2)
        header = f"{'Run':<{run_width}} {'N':>5} {'ROUGE-L F1':>11} {'EM':>7} {'Latency':>9}"
        print(f"\n{header}")
        print("-" * len(header))
        for s in summaries:
            print(
                f"{s['label']:<{run_width}} {s['n']:>5} {s['mean_rouge_f1']:>11.3f} "
                f"{s['mean_exact_match']:>7.3f} {s['mean_latency_sec']:>8.1f}s"
            )

        if len(summaries) == 2:
            a, b = summaries
            print(f"\nDelta ({b['label']} - {a['label']}):")
            print(f"  ROUGE-L F1:  {b['mean_rouge_f1'] - a['mean_rouge_f1']:+.3f}")
            print(f"  Exact Match: {b['mean_exact_match'] - a['mean_exact_match']:+.3f}")
            print(f"  Latency:     {b['mean_latency_sec'] - a['mean_latency_sec']:+.1f}s")

        qtypes = sorted({qt for s in summaries for qt in s["by_question_type"]})
        col_width = max(15, max(len(s["label"]) for s in summaries) + 2)
        print("\nMean ROUGE-L F1 by question_type:")
        print(f"{'question_type':<25}" + "".join(f"{s['label']:>{col_width}}" for s in summaries))
        for qt in qtypes:
            row = f"{qt:<25}"
            for s in summaries:
                v = s["by_question_type"].get(qt)
                row += f"{v:>{col_width}.3f}" if v is not None else f"{'-':>{col_width}}"
            print(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare FinDocRetrieval experiment runs")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="List saved runs")
    p_list.add_argument("condition", nargs="?", help="Only list runs for this condition")
    p_list.set_defaults(func=cmd_list)

    p_show = sub.add_parser("show", help="Summarize a single run")
    p_show.add_argument("run", help="Condition name, 'condition@timestamp', or a path")
    p_show.set_defaults(func=cmd_show)

    p_compare = sub.add_parser("compare", help="Compare two or more runs")
    p_compare.add_argument("runs", nargs="+", help="Condition names, 'condition@timestamp', or paths")
    p_compare.set_defaults(func=cmd_compare)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
