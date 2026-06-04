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


def answer_recall(
    predicted: str,
    reference_spans: list[str],
) -> float:
    """Fraction of reference-span tokens found anywhere in the predicted answer."""
    ref_tokens = [
        tok for span in reference_spans for tok in tokenize(span)
    ]
    if not ref_tokens:
        return 0.0
    pred_tokens = tokenize(predicted)
    if not pred_tokens:
        return 0.0
    pred_set = Counter(pred_tokens)
    ref_counter = Counter(ref_tokens)
    common = pred_set & ref_counter
    return sum(common.values()) / len(ref_tokens)


def span_precision(
    predicted: str,
    reference_spans: list[str],
) -> float:
    """Fraction of predicted tokens that appear in the union of reference spans."""
    pred_tokens = tokenize(predicted)
    if not pred_tokens:
        return 0.0
    ref_tokens = [
        tok for span in reference_spans for tok in tokenize(span)
    ]
    if not ref_tokens:
        return 0.0
    pred_counter = Counter(pred_tokens)
    ref_counter = Counter(ref_tokens)
    common = pred_counter & ref_counter
    return sum(common.values()) / len(pred_tokens)


def semantic_f1_score(
    predicted: str,
    reference_spans: list[str],
) -> float:
    """Harmonic mean of answer_recall and span_precision."""
    rec = answer_recall(predicted, reference_spans)
    prec = span_precision(predicted, reference_spans)
    if rec + prec == 0:
        return 0.0
    return 2 * rec * prec / (rec + prec)


def numeric_accuracy(predicted: str, gold: str, tolerance: float = 0.02) -> float:
    """Compare numeric values with a relative tolerance.

    Returns 1.0 if both strings parse to numbers within *tolerance* of each
    other (default 2 %), 0.0 if the values differ beyond that or cannot be
    parsed.
    """
    def _extract_number(s: str) -> float | None:
        s = s.replace(",", "").replace("%", "").replace("$", "").strip()
        try:
            return float(s)
        except ValueError:
            return None

    pred_val = _extract_number(predicted)
    gold_val = _extract_number(gold)
    if pred_val is None or gold_val is None:
        return 0.0
    if gold_val == 0:
        return 1.0 if pred_val == 0 else 0.0
    return 1.0 if abs(pred_val - gold_val) / abs(gold_val) <= tolerance else 0.0


def table_extraction_score(
    predicted_cells: list[str],
    gold_cells: list[str],
) -> dict[str, float]:
    """Evaluate tabular data extraction from financial documents.

    Returns precision, recall, and F1 over cell-level exact matches.
    """
    pred_norm = {c.strip().lower() for c in predicted_cells}
    gold_norm = {c.strip().lower() for c in gold_cells}
    tp = len(pred_norm & gold_norm)
    precision = tp / len(pred_norm) if pred_norm else 0.0
    recall = tp / len(gold_norm) if gold_norm else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def cost_per_f1_point(f1: float, cost_usd: float) -> float:
    """USD cost per F1 point (avoids division by zero)."""
    return cost_usd / max(f1, 1e-9)


def marginal_gain(baseline_f1: float, technique_f1: float) -> float:
    """Absolute F1 gain of a technique over the baseline."""
    return technique_f1 - baseline_f1


def ablation_summary(results: list[dict]) -> dict:
    """Aggregate ablation results by technique."""
    by_technique: dict[str, list[dict]] = {}
    for r in results:
        by_technique.setdefault(r["technique"], []).append(r)

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
