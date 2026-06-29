"""Metrics from DESIGN_DOC.md Experiments 1, 2, and 4: NHR, TRR, and
cross-document co-retrieval.

These were specified in DESIGN_DOC.md but had no corresponding
implementation anywhere in src/findocretrieval. This module provides a
first, dependency-free implementation operating on plain dataclasses so
they can be wired up to whatever retriever is evaluated next (BM25,
dense, hybrid, ...). It does NOT include a FinRAG-Bench dataset, an
EDGAR scraper, or trained retrieval models -- those remain future work
(see RESEARCH_GAP_ANALYSIS.md).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

Origin = Literal["table", "prose", "mixed"]


@dataclass
class GoldPassage:
    """A gold-labelled supporting passage for a question."""

    id: str
    origin: Origin
    numeric_value: float | None = None
    fiscal_year: int | None = None


@dataclass
class RetrievedPassage:
    """A passage returned by a retrieval system, ranked by the system."""

    id: str
    rank: int
    numeric_value: float | None = None
    fiscal_year: int | None = None


def compute_trr(
    retrieved: list[RetrievedPassage],
    gold: list[GoldPassage],
) -> float | None:
    """Table Retrieval Recall (DESIGN_DOC.md, Experiment 1).

    Recall computed only over gold passages whose origin is 'table'.
    Returns None when the question has no table-origin gold passages
    (the metric is undefined / not applicable for that question).
    """
    table_golds = [g for g in gold if g.origin == "table"]
    if not table_golds:
        return None
    retrieved_ids = {r.id for r in retrieved}
    hits = sum(1 for g in table_golds if g.id in retrieved_ids)
    return hits / len(table_golds)


def compute_prose_recall(
    retrieved: list[RetrievedPassage],
    gold: list[GoldPassage],
) -> float | None:
    """Recall restricted to prose-origin gold passages, for comparison
    against compute_trr (the "table-blindness gap" in Experiment 1).
    """
    prose_golds = [g for g in gold if g.origin == "prose"]
    if not prose_golds:
        return None
    retrieved_ids = {r.id for r in retrieved}
    hits = sum(1 for g in prose_golds if g.id in retrieved_ids)
    return hits / len(prose_golds)


def table_blindness_gap(
    retrieved: list[RetrievedPassage],
    gold: list[GoldPassage],
) -> float | None:
    """Prose recall minus table recall (TRR), in percentage points.

    Positive values indicate the system is comparatively worse at
    retrieving table evidence than prose evidence ("table blindness").
    Returns None if either recall is undefined for this question.
    """
    trr = compute_trr(retrieved, gold)
    prose = compute_prose_recall(retrieved, gold)
    if trr is None or prose is None:
        return None
    return prose - trr


def is_numeric_hallucination(
    retrieved_value: float | None,
    gold_value: float | None,
    relative_tolerance: float = 0.05,
) -> bool | None:
    """True if the top-retrieved numeric value deviates from gold by more
    than *relative_tolerance* (DESIGN_DOC.md Experiment 2, default 5%).

    Returns None if either value is missing (non-numeric question).
    """
    if retrieved_value is None or gold_value is None:
        return None
    if gold_value == 0:
        return retrieved_value != 0
    return abs(retrieved_value - gold_value) / abs(gold_value) > relative_tolerance


def compute_nhr(
    top_retrieved: list[RetrievedPassage],
    gold: list[GoldPassage],
) -> dict[str, float | None]:
    """Numeric Hallucination Rate over a batch of (top-1 retrieved, gold)
    pairs, one pair per numeric question.

    Args:
        top_retrieved: The single top-ranked retrieved passage for each
            numeric question (parallel to *gold*).
        gold: The gold passage for each numeric question (parallel to
            *top_retrieved*).

    Returns:
        Dict with 'nhr' (overall hallucination rate) and
        'nhr_wrong_year' (hallucination rate restricted to cases where
        the retrieved passage's fiscal_year differs from gold's), or
        None for either key if there are no scorable numeric questions.
    """
    flags: list[bool] = []
    wrong_year_flags: list[bool] = []
    for top, g in zip(top_retrieved, gold):
        flag = is_numeric_hallucination(top.numeric_value, g.numeric_value)
        if flag is None:
            continue
        flags.append(flag)
        if g.fiscal_year is not None and top.fiscal_year is not None:
            if top.fiscal_year != g.fiscal_year:
                wrong_year_flags.append(flag)

    nhr = sum(flags) / len(flags) if flags else None
    nhr_wrong_year = (
        sum(wrong_year_flags) / len(wrong_year_flags) if wrong_year_flags else None
    )
    return {"nhr": nhr, "nhr_wrong_year": nhr_wrong_year}


def cross_document_coretrieval_rate(
    retrieved_by_question: list[list[RetrievedPassage]],
    required_ids_by_question: list[list[str]],
    top_k: int = 10,
) -> float:
    """Fraction of cross-document questions for which ALL required gold
    passages are retrieved within the top *top_k* (DESIGN_DOC.md
    Experiment 4: "co-retrieval rate").

    Args:
        retrieved_by_question: For each cross-doc question, the ranked
            list of retrieved passages.
        required_ids_by_question: For each cross-doc question, the list
            of gold passage ids that must ALL be present in the top_k
            retrieved set for the question to count as a "hit".
        top_k: Cutoff rank applied to each retrieved list before
            checking coverage.

    Returns:
        co-retrieval rate in [0, 1]. Returns 0.0 if there are no
        questions (rather than raising), matching the "no signal yet"
        semantics used elsewhere in this module.
    """
    if not required_ids_by_question:
        return 0.0
    hits = 0
    for retrieved, required_ids in zip(retrieved_by_question, required_ids_by_question):
        top_ids = {r.id for r in retrieved if r.rank <= top_k}
        if all(rid in top_ids for rid in required_ids):
            hits += 1
    return hits / len(required_ids_by_question)
