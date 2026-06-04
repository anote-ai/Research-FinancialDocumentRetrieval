"""Tests for findocretrieval.evaluate."""

import pytest

from findocretrieval.evaluate import (
    ExactMatchScore,
    ablation_summary,
    cost_per_f1_point,
    exact_match,
    f1_score_tokens,
    marginal_gain,
)


# --- exact_match ---

def test_exact_match_identical():
    assert exact_match("Apple Inc.", "Apple Inc.") == 1.0


def test_exact_match_normalised_whitespace():
    assert exact_match("  Apple  Inc. ", "Apple Inc.") == 1.0


def test_exact_match_different():
    assert exact_match("Apple", "Google") == 0.0


def test_exact_match_case_insensitive():
    assert exact_match("APPLE INC", "apple inc") == 1.0


# --- f1_score_tokens ---

def test_f1_score_perfect():
    assert f1_score_tokens("net income was 100", "net income was 100") == pytest.approx(1.0)


def test_f1_score_partial():
    score = f1_score_tokens("net income", "net income was 100")
    assert 0.0 < score < 1.0


def test_f1_score_no_overlap():
    assert f1_score_tokens("apple orange", "banana grape") == pytest.approx(0.0)


def test_f1_score_both_empty():
    assert f1_score_tokens("", "") == pytest.approx(1.0)


# --- cost_per_f1_point ---

def test_cost_per_f1_point_basic():
    assert cost_per_f1_point(0.5, 1.0) == pytest.approx(2.0)


def test_cost_per_f1_point_zero_f1():
    assert cost_per_f1_point(0.0, 0.5) == float("inf")


# --- marginal_gain ---

def test_marginal_gain_positive():
    assert marginal_gain(0.6, 0.75) == pytest.approx(0.15)


def test_marginal_gain_negative():
    assert marginal_gain(0.8, 0.7) == pytest.approx(-0.1)


# --- ablation_summary ---

def test_ablation_summary_groups_correctly():
    results = [
        {"technique": "reranking", "f1": 0.8, "cost_usd": 0.01},
        {"technique": "reranking", "f1": 0.6, "cost_usd": 0.01},
        {"technique": "metadata", "f1": 0.7, "cost_usd": 0.005},
    ]
    summary = ablation_summary(results)
    assert "reranking" in summary
    assert "metadata" in summary
    assert summary["reranking"]["mean_f1"] == pytest.approx(0.7)
    assert summary["metadata"]["mean_f1"] == pytest.approx(0.7)
