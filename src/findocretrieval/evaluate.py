"""Evaluation metrics for financial document retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Counter


@dataclass
class ExactMatchScore:
    """Stores per-query exact-match evaluation result."""

    query_id: str
    score: float
    cost_usd: float


def exact_match(predicted: str, reference: str) -> float:
    """Return 1.0 if the normalised strings are identical, else 0.0."""
    return 1.0 if _normalise(predicted) == _normalise(reference) else 0.0


def f1_score_tokens(predicted: str, reference: str) -> float:
    """Token-level F1 score between predicted and reference answer strings."""
    pred_tokens = _normalise(predicted).split()
    ref_tokens = _normalise(reference).split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    pred_counter = Counter(pred_tokens)
    ref_counter = Counter(ref_tokens)
    common = sum((pred_counter & ref_counter).values())
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def cost_per_f1_point(f1: float, cost_usd: float) -> float:
    """Cost in USD per F1 point gained (returns inf when f1 == 0)."""
    if f1 == 0.0:
        return float("inf")
    return cost_usd / f1


def marginal_gain(baseline_f1: float, technique_f1: float) -> float:
    """Absolute F1 improvement of a technique over the baseline."""
    return technique_f1 - baseline_f1


def ablation_summary(results: list[dict]) -> dict:
    """Summarise per-technique mean F1 and mean cost from a list of result dicts.

    Each dict should have keys: technique (str), f1 (float), cost_usd (float).

    Returns:
        Dict mapping technique name -> {mean_f1, mean_cost_usd}.
    """
    from collections import defaultdict

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in results:
        grouped[row["technique"]].append(row)

    summary = {}
    for technique, rows in grouped.items():
        mean_f1 = sum(r["f1"] for r in rows) / len(rows)
        mean_cost = sum(r["cost_usd"] for r in rows) / len(rows)
        summary[technique] = {"mean_f1": mean_f1, "mean_cost_usd": mean_cost}
    return summary


def _normalise(text: str) -> str:
    """Lowercase and strip whitespace for comparison."""
    return " ".join(text.lower().split())
