from __future__ import annotations
import re


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens."""
    return re.findall(r"[a-z0-9]+", text.lower())


def exact_match(predicted: str, gold: str) -> float:
    return 1.0 if predicted.strip().lower() == gold.strip().lower() else 0.0


def f1_score_tokens(predicted: str, gold: str) -> float:
    """Token-level F1 between two strings."""
    pred_toks = tokenize(predicted)
    gold_toks = tokenize(gold)
    if not pred_toks or not gold_toks:
        return 0.0
    common = set(pred_toks) & set(gold_toks)
    if not common:
        return 0.0
    precision = len(common) / len(pred_toks)
    recall = len(common) / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def answer_recall(predicted: str, reference_spans: list[str]) -> float:
    """Fraction of reference-span tokens that appear in the predicted answer."""
    if not reference_spans or not predicted:
        return 0.0
    ref_tokens: set[str] = set()
    for span in reference_spans:
        ref_tokens.update(tokenize(span))
    if not ref_tokens:
        return 0.0
    pred_tokens = set(tokenize(predicted))
    return len(pred_tokens & ref_tokens) / len(ref_tokens)


def span_precision(predicted: str, reference_spans: list[str]) -> float:
    """Fraction of predicted tokens found in at least one reference span."""
    if not predicted or not reference_spans:
        return 0.0
    pred_tokens = tokenize(predicted)
    if not pred_tokens:
        return 0.0
    ref_tokens: set[str] = set()
    for span in reference_spans:
        ref_tokens.update(tokenize(span))
    return sum(1 for t in pred_tokens if t in ref_tokens) / len(pred_tokens)


def semantic_f1_score(predicted: str, reference_spans: list[str]) -> float:
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

    Returns precision, recall, and F1 over cell-level exact matches after
    normalising whitespace.  Useful for income-statement / balance-sheet QA.
    """
    pred_norm = [c.strip().lower() for c in predicted_cells]
    gold_norm = [c.strip().lower() for c in gold_cells]
    pred_set = set(pred_norm)
    gold_set = set(gold_norm)
    tp = len(pred_set & gold_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(gold_set) if gold_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def cost_per_f1_point(cost_usd: float, f1: float) -> float:
    """USD spent per unit of F1; returns inf when F1 is zero."""
    return cost_usd / f1 if f1 > 0 else float("inf")


def marginal_gain(baseline_f1: float, improved_f1: float) -> float:
    """Absolute F1 improvement over baseline."""
    return max(0.0, improved_f1 - baseline_f1)


def ablation_summary(results: list[dict]) -> dict[str, dict]:
    """Compute per-technique stats including cumulative marginal gain."""
    summary: dict[str, dict] = {}
    baseline_f1 = results[0]["f1"] if results else 0.0
    for entry in results:
        name = entry["technique"]
        f1 = entry["f1"]
        summary[name] = {
            "f1": f1,
            "cost_usd": entry.get("cost_usd", 0.0),
            "marginal_gain": round(marginal_gain(baseline_f1, f1), 10),
            "cost_per_f1": cost_per_f1_point(entry.get("cost_usd", 0.0), f1),
        }
    return summary
