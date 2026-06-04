from __future__ import annotations
import re
from collections import Counter


def tokenize(text: str) -> list[str]:
    """Lowercase and split on whitespace + punctuation."""
    return re.findall(r"[a-z0-9]+", text.lower())


def exact_match(predicted: str, reference: str) -> float:
    """Return 1.0 if normalized strings are equal, else 0.0."""
    return 1.0 if predicted.strip().lower() == reference.strip().lower() else 0.0


def f1_score_tokens(predicted: str, reference: str) -> float:
    """Compute token-level F1 between predicted and reference strings."""
    pred_tokens = tokenize(predicted)
    ref_tokens = tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    pred_counter = Counter(pred_tokens)
    ref_counter = Counter(ref_tokens)
    common = pred_counter & ref_counter
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def cost_per_f1_point(f1: float, cost_usd: float) -> float:
    """USD cost per F1 point (avoids division by zero)."""
    return cost_usd / max(f1, 1e-9)


def marginal_gain(baseline_f1: float, technique_f1: float) -> float:
    """Absolute F1 gain of a technique over the baseline."""
    return technique_f1 - baseline_f1


def ablation_summary(results: list[dict]) -> dict:
    """Aggregate ablation results by technique.

    Args:
        results: List of dicts with keys 'technique', 'f1', 'cost_usd'.

    Returns:
        Dict mapping technique name to {mean_f1, mean_cost, marginal_gain}.
    """
    by_technique: dict[str, list[dict]] = {}
    for r in results:
        by_technique.setdefault(r["technique"], []).append(r)

    # Determine baseline F1
    baseline_rows = by_technique.get("baseline", [])
    baseline_f1 = (
        sum(r["f1"] for r in baseline_rows) / len(baseline_rows) if baseline_rows else 0.0
    )

    summary: dict[str, dict] = {}
    for technique, rows in by_technique.items():
        mean_f1 = sum(r["f1"] for r in rows) / len(rows)
        mean_cost = sum(r["cost_usd"] for r in rows) / len(rows)
        summary[technique] = {
            "mean_f1": mean_f1,
            "mean_cost": mean_cost,
            "marginal_gain": marginal_gain(baseline_f1, mean_f1),
        }
    return summary
