from __future__ import annotations
import random
from .core import Document

SAMPLE_10K_TEXT: str = (
    "Item 1. Business. "
    "Acme Corporation (the 'Company') is a leading provider of enterprise software solutions "
    "serving Fortune 500 clients across North America, Europe, and Asia-Pacific. "
    "Founded in 1998, the Company operates through three segments: Cloud Services, "
    "Professional Services, and Licensing. "
    "In fiscal year 2025, total revenue increased 14% year-over-year to $4.2 billion, "
    "driven primarily by a 31% increase in Cloud Services revenue to $2.8 billion. "
    "Item 1A. Risk Factors. "
    "The Company faces significant competition from established vendors including large "
    "technology conglomerates with substantially greater financial resources. "
    "Macroeconomic conditions, including inflationary pressures and rising interest rates, "
    "may adversely affect customer IT budgets and lengthen sales cycles. "
    "Cybersecurity incidents, including ransomware attacks, represent a material risk "
    "to operations and could result in regulatory fines and reputational harm. "
    "Item 7. Management Discussion and Analysis. "
    "Operating income for fiscal 2025 was $620 million, a margin of 14.8%, compared to "
    "$510 million and 13.9% in fiscal 2024. "
    "The improvement reflects operating leverage in the Cloud Services segment as "
    "infrastructure costs grew more slowly than revenue. "
    "Free cash flow was $540 million, up from $430 million in the prior year, "
    "supporting continued investment in R&D and strategic acquisitions."
)

ABLATION_TECHNIQUES: list[str] = [
    "baseline",
    "+reranking",
    "+metadata",
    "+query_expansion",
    "+hybrid",
]


def make_document(doc_id: str = "doc_001") -> Document:
    """Return a Document backed by the sample 10-K text."""
    return Document(
        doc_id=doc_id,
        text=SAMPLE_10K_TEXT,
        metadata={"source": "10-K", "fiscal_year": 2025, "company": "Acme Corporation"},
        source="10-K",
    )


def make_query_results(n: int = 5, seed: int = 42) -> list[dict]:
    """Generate synthetic ablation result dicts.

    Returns:
        List of dicts with keys: technique, f1, cost_usd.
    """
    rng = random.Random(seed)
    base_f1 = 0.52
    base_cost = 0.008
    boosts = {
        "baseline": 0.0,
        "+reranking": 0.07,
        "+metadata": 0.05,
        "+query_expansion": 0.06,
        "+hybrid": 0.14,
    }
    rows: list[dict] = []
    for technique in ABLATION_TECHNIQUES:
        for _ in range(n):
            noise = rng.gauss(0, 0.02)
            f1 = min(1.0, max(0.0, base_f1 + boosts[technique] + noise))
            cost_noise = rng.gauss(0, 0.001)
            cost_multiplier = 1.0 + boosts[technique] * 4
            cost = max(0.001, base_cost * cost_multiplier + cost_noise)
            rows.append({"technique": technique, "f1": f1, "cost_usd": cost})
    return rows
