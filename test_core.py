from __future__ import annotations

from findocretrieval.core import Document
from findocretrieval.data import (
    ABLATION_TECHNIQUES,
    SAMPLE_10K_TEXT,
    make_document,
    make_query_results,
)


def test_sample_10k_text_non_empty() -> None:
    assert len(SAMPLE_10K_TEXT) > 100


def test_make_document_type() -> None:
    doc = make_document()
    assert isinstance(doc, Document)
    assert doc.doc_id == "doc_001"


def test_ablation_techniques_contains_baseline() -> None:
    assert "baseline" in ABLATION_TECHNIQUES
    assert len(ABLATION_TECHNIQUES) >= 2


def test_make_query_results_structure() -> None:
    results = make_query_results(n=3, seed=0)
    assert len(results) == len(ABLATION_TECHNIQUES) * 3
    for r in results:
        assert "technique" in r
        assert "f1" in r
        assert "cost_usd" in r
        assert 0.0 <= r["f1"] <= 1.0
