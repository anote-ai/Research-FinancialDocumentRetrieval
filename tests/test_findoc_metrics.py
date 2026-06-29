from __future__ import annotations
from findocretrieval.findoc_metrics import (
    GoldPassage,
    RetrievedPassage,
    compute_trr,
    compute_prose_recall,
    table_blindness_gap,
    is_numeric_hallucination,
    compute_nhr,
    cross_document_coretrieval_rate,
)


# ---------------------------------------------------------------------------
# TRR / prose recall / table-blindness gap
# ---------------------------------------------------------------------------

def test_compute_trr_no_table_golds_returns_none() -> None:
    gold = [GoldPassage(id="p1", origin="prose")]
    retrieved = [RetrievedPassage(id="p1", rank=1)]
    assert compute_trr(retrieved, gold) is None


def test_compute_trr_full_recall() -> None:
    gold = [GoldPassage(id="t1", origin="table"), GoldPassage(id="t2", origin="table")]
    retrieved = [RetrievedPassage(id="t1", rank=1), RetrievedPassage(id="t2", rank=2)]
    assert compute_trr(retrieved, gold) == 1.0


def test_compute_trr_partial_recall() -> None:
    gold = [GoldPassage(id="t1", origin="table"), GoldPassage(id="t2", origin="table")]
    retrieved = [RetrievedPassage(id="t1", rank=1)]
    assert compute_trr(retrieved, gold) == 0.5


def test_compute_prose_recall_basic() -> None:
    gold = [GoldPassage(id="x1", origin="prose")]
    retrieved = [RetrievedPassage(id="x1", rank=1)]
    assert compute_prose_recall(retrieved, gold) == 1.0


def test_table_blindness_gap_positive_when_table_worse() -> None:
    gold = [
        GoldPassage(id="t1", origin="table"),
        GoldPassage(id="p1", origin="prose"),
    ]
    retrieved = [RetrievedPassage(id="p1", rank=1)]  # only prose retrieved
    gap = table_blindness_gap(retrieved, gold)
    assert gap == 1.0  # prose recall 1.0 - table recall 0.0


def test_table_blindness_gap_none_when_no_table_golds() -> None:
    gold = [GoldPassage(id="p1", origin="prose")]
    retrieved = [RetrievedPassage(id="p1", rank=1)]
    assert table_blindness_gap(retrieved, gold) is None


# ---------------------------------------------------------------------------
# NHR
# ---------------------------------------------------------------------------

def test_is_numeric_hallucination_within_tolerance() -> None:
    assert is_numeric_hallucination(100.0, 103.0) is False


def test_is_numeric_hallucination_outside_tolerance() -> None:
    assert is_numeric_hallucination(100.0, 150.0) is True


def test_is_numeric_hallucination_missing_value_returns_none() -> None:
    assert is_numeric_hallucination(None, 100.0) is None
    assert is_numeric_hallucination(100.0, None) is None


def test_is_numeric_hallucination_gold_zero() -> None:
    assert is_numeric_hallucination(0.0, 0.0) is False
    assert is_numeric_hallucination(5.0, 0.0) is True


def test_compute_nhr_basic() -> None:
    top_retrieved = [
        RetrievedPassage(id="a", rank=1, numeric_value=100.0, fiscal_year=2024),
        RetrievedPassage(id="b", rank=1, numeric_value=50.0, fiscal_year=2023),
    ]
    gold = [
        GoldPassage(id="a", origin="prose", numeric_value=103.0, fiscal_year=2024),
        GoldPassage(id="b", origin="prose", numeric_value=10.0, fiscal_year=2022),
    ]
    result = compute_nhr(top_retrieved, gold)
    assert result["nhr"] == 0.5  # one hallucination out of two
    assert result["nhr_wrong_year"] == 1.0  # the wrong-year case is also the hallucination


def test_compute_nhr_no_numeric_questions_returns_none() -> None:
    top_retrieved = [RetrievedPassage(id="a", rank=1)]
    gold = [GoldPassage(id="a", origin="prose")]
    result = compute_nhr(top_retrieved, gold)
    assert result["nhr"] is None
    assert result["nhr_wrong_year"] is None


# ---------------------------------------------------------------------------
# Cross-document co-retrieval
# ---------------------------------------------------------------------------

def test_cross_document_coretrieval_rate_all_hit() -> None:
    retrieved_by_question = [
        [RetrievedPassage(id="a", rank=1), RetrievedPassage(id="b", rank=2)],
    ]
    required_ids_by_question = [["a", "b"]]
    assert cross_document_coretrieval_rate(retrieved_by_question, required_ids_by_question) == 1.0


def test_cross_document_coretrieval_rate_partial_miss() -> None:
    retrieved_by_question = [
        [RetrievedPassage(id="a", rank=1)],  # missing 'b'
        [RetrievedPassage(id="c", rank=1), RetrievedPassage(id="d", rank=2)],
    ]
    required_ids_by_question = [["a", "b"], ["c", "d"]]
    assert cross_document_coretrieval_rate(retrieved_by_question, required_ids_by_question) == 0.5


def test_cross_document_coretrieval_rate_respects_top_k() -> None:
    retrieved_by_question = [
        [RetrievedPassage(id="a", rank=1), RetrievedPassage(id="b", rank=11)],
    ]
    required_ids_by_question = [["a", "b"]]
    assert cross_document_coretrieval_rate(
        retrieved_by_question, required_ids_by_question, top_k=10
    ) == 0.0


def test_cross_document_coretrieval_rate_empty_returns_zero() -> None:
    assert cross_document_coretrieval_rate([], []) == 0.0
