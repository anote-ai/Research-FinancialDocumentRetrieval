from __future__ import annotations
import pytest
from findocretrieval.evaluate import (
    tokenize,
    exact_match,
    f1_score_tokens,
    answer_recall,
    span_precision,
    semantic_f1_score,
    numeric_accuracy,
    table_extraction_score,
    ablation_summary,
)


def test_tokenize_basic() -> None:
    assert tokenize("Hello, World!") == ["hello", "world"]


def test_tokenize_numbers() -> None:
    tokens = tokenize("Revenue was $4.2 billion in 2025")
    assert "4" in tokens
    assert "2" in tokens
    assert "2025" in tokens


def test_exact_match_true() -> None:
    assert exact_match("Yes", "yes") == 1.0


def test_exact_match_false() -> None:
    assert exact_match("Yes", "No") == 0.0


def test_f1_perfect() -> None:
    assert f1_score_tokens("hello world", "hello world") == pytest.approx(1.0)


def test_f1_no_overlap() -> None:
    assert f1_score_tokens("foo", "bar") == pytest.approx(0.0)


def test_f1_empty() -> None:
    assert f1_score_tokens("", "hello") == pytest.approx(0.0)


def test_answer_recall_perfect() -> None:
    spans = ["revenue increased 14%", "to 4 2 billion"]
    predicted = "revenue increased 14 to 4 2 billion"
    recall = answer_recall(predicted, spans)
    assert 0.0 <= recall <= 1.0
    assert recall > 0.5


def test_answer_recall_no_overlap() -> None:
    assert answer_recall("completely different text", ["operating income margin"]) == pytest.approx(0.0)


def test_answer_recall_empty_spans() -> None:
    assert answer_recall("some answer", []) == pytest.approx(0.0)


def test_answer_recall_empty_predicted() -> None:
    assert answer_recall("", ["revenue"]) == pytest.approx(0.0)


def test_span_precision_perfect() -> None:
    spans = ["operating income"]
    predicted = "operating income"
    assert span_precision(predicted, spans) == pytest.approx(1.0)


def test_span_precision_no_overlap() -> None:
    assert span_precision("hallucinated text", ["revenue"]) == pytest.approx(0.0)


def test_span_precision_empty_predicted() -> None:
    assert span_precision("", ["revenue"]) == pytest.approx(0.0)


def test_span_precision_empty_spans() -> None:
    assert span_precision("some answer", []) == pytest.approx(0.0)


def test_semantic_f1_perfect() -> None:
    spans = ["free cash flow 540 million"]
    predicted = "free cash flow 540 million"
    assert semantic_f1_score(predicted, spans) == pytest.approx(1.0)


def test_semantic_f1_zero() -> None:
    assert semantic_f1_score("abc", ["xyz"]) == pytest.approx(0.0)


def test_semantic_f1_between_zero_and_one() -> None:
    predicted = "revenue grew 31 percent to 2 8 billion"
    spans = ["31% increase in Cloud Services revenue to $2.8 billion"]
    score = semantic_f1_score(predicted, spans)
    assert 0.0 < score < 1.0


def test_semantic_f1_harmonic_mean() -> None:
    predicted = "revenue income margin"
    spans = ["revenue operating income"]
    rec = answer_recall(predicted, spans)
    prec = span_precision(predicted, spans)
    expected = 2 * rec * prec / (rec + prec) if (rec + prec) > 0 else 0.0
    assert semantic_f1_score(predicted, spans) == pytest.approx(expected)


def test_numeric_accuracy_exact() -> None:
    assert numeric_accuracy("14.2", "14.2") == 1.0


def test_numeric_accuracy_within_tolerance() -> None:
    assert numeric_accuracy("14.1", "14.2", tolerance=0.05) == 1.0


def test_numeric_accuracy_outside_tolerance() -> None:
    assert numeric_accuracy("10.0", "14.2") == 0.0


def test_numeric_accuracy_non_numeric() -> None:
    assert numeric_accuracy("N/A", "14.2") == 0.0


def test_numeric_accuracy_currency_symbols() -> None:
    assert numeric_accuracy("$4.2", "4.2") == 1.0


def test_table_extraction_score_perfect() -> None:
    cells = ["Revenue", "$4.2B", "Net Income", "$1.1B"]
    result = table_extraction_score(cells, cells)
    assert result["f1"] == pytest.approx(1.0)


def test_table_extraction_score_partial() -> None:
    pred = ["Revenue", "$4.2B"]
    gold = ["Revenue", "$4.2B", "Net Income"]
    result = table_extraction_score(pred, gold)
    assert 0.0 < result["f1"] < 1.0
    assert result["recall"] < 1.0


def test_table_extraction_score_empty_pred() -> None:
    result = table_extraction_score([], ["Revenue"])
    assert result["f1"] == pytest.approx(0.0)


def test_ablation_summary_marginal_gain() -> None:
    results = [
        {"technique": "baseline", "f1": 0.5, "cost_usd": 0.01},
        {"technique": "+reranking", "f1": 0.6, "cost_usd": 0.02},
    ]
    summary = ablation_summary(results)
    assert summary["baseline"]["marginal_gain"] == pytest.approx(0.0)
    assert summary["+reranking"]["marginal_gain"] == pytest.approx(0.1)
