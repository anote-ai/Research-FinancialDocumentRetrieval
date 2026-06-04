from __future__ import annotations
import pytest
from findocretrieval.evaluate import (
    exact_match,
    f1_score_tokens,
    cost_per_f1_point,
    ablation_summary,
)


def test_exact_match_identical() -> None:
    assert exact_match("hello world", "hello world") == pytest.approx(1.0)


def test_exact_match_case_insensitive() -> None:
    assert exact_match("Hello World", "hello world") == pytest.approx(1.0)


def test_exact_match_different() -> None:
    assert exact_match("foo", "bar") == pytest.approx(0.0)


def test_f1_score_tokens_exact() -> None:
    assert f1_score_tokens("the cat sat", "the cat sat") == pytest.approx(1.0)


def test_f1_score_tokens_no_overlap() -> None:
    assert f1_score_tokens("apple orange", "banana grape") == pytest.approx(0.0)


def test_f1_score_tokens_partial() -> None:
    score = f1_score_tokens("the cat sat on the mat", "the cat ran on the floor")
    assert 0.0 < score < 1.0


def test_cost_per_f1_point_normal() -> None:
    result = cost_per_f1_point(f1=0.5, cost_usd=0.10)
    assert result == pytest.approx(0.20)


def test_ablation_summary_structure() -> None:
    results = [
        {"technique": "baseline", "f1": 0.5, "cost_usd": 0.01},
        {"technique": "baseline", "f1": 0.6, "cost_usd": 0.01},
        {"technique": "+reranking", "f1": 0.7, "cost_usd": 0.02},
    ]
    summary = ablation_summary(results)
    assert "baseline" in summary
    assert "+reranking" in summary
    assert "mean_f1" in summary["baseline"]
    assert "marginal_gain" in summary["+reranking"]
    assert summary["baseline"]["marginal_gain"] == pytest.approx(0.0)
