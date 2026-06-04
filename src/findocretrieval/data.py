from __future__ import annotations
import random
from .core import Document

# ---------------------------------------------------------------------------
# Realistic multi-section 10-K / 10-Q text samples
# ---------------------------------------------------------------------------

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
    "Concentration of revenue among a small number of customers increases vulnerability "
    "to churn; the top ten customers accounted for 38% of revenue in fiscal 2025. "
    "Item 7. Management Discussion and Analysis. "
    "Operating income for fiscal 2025 was $620 million, a margin of 14.8%, compared to "
    "$510 million and 13.9% in fiscal 2024. "
    "The improvement reflects operating leverage in the Cloud Services segment as "
    "infrastructure costs grew more slowly than revenue. "
    "Free cash flow was $540 million, up from $430 million in the prior year, "
    "supporting continued investment in R&D and strategic acquisitions. "
    "Item 8. Financial Statements. "
    "Total assets as of December 31, 2025 were $9.1 billion, including $1.8 billion of "
    "goodwill and $640 million of other intangible assets. "
    "Total liabilities were $4.3 billion, of which long-term debt accounted for $2.1 billion. "
    "Shareholders' equity was $4.8 billion."
)

SAMPLE_10Q_TEXT: str = (
    "Item 1. Financial Statements (Unaudited). "
    "For the three months ended September 30, 2025, Acme Corporation reported net revenue "
    "of $1.12 billion, a 16% increase over the prior-year quarter. "
    "Gross profit was $720 million, representing a gross margin of 64.3%, up 120 basis points "
    "year-over-year, driven by higher Cloud Services mix and improved contract economics. "
    "Operating expenses totaled $580 million including $180 million of R&D expense. "
    "Net income was $105 million, or $0.38 per diluted share, compared with $84 million, "
    "or $0.30 per diluted share, in the same period last year. "
    "Item 2. Management Discussion and Analysis of Financial Condition. "
    "Liquidity and Capital Resources. Cash and cash equivalents were $920 million at "
    "quarter end, compared to $810 million at December 31, 2024. "
    "The Company generated $165 million in operating cash flow during the quarter. "
    "Capital expenditures were $42 million, reflecting continued investment in data-center "
    "infrastructure to support Cloud Services growth. "
    "Item 1A. Risk Factors. "
    "There have been no material changes to the risk factors disclosed in the Company's "
    "Annual Report on Form 10-K for fiscal 2024, other than the following update: "
    "Geopolitical tensions and potential trade restrictions in certain international "
    "markets could adversely affect the Company's ability to serve clients in those regions."
)

SAMPLE_MANDA_TEXT: str = (
    "Item 7. Management's Discussion and Analysis of Financial Condition "
    "and Results of Operations. "
    "Revenue. Total revenue for fiscal 2025 was $4.2 billion compared with $3.7 billion "
    "in fiscal 2024, an increase of $500 million or 14%. Cloud Services revenue grew "
    "$667 million or 31% to $2.8 billion, reflecting increased adoption of our "
    "subscription-based SaaS offerings and expansion within the existing customer base. "
    "Professional Services revenue declined 4% to $780 million due to project completions "
    "not fully offset by new engagements. "
    "Licensing revenue was $620 million, essentially flat year-over-year as customers "
    "continue to migrate from perpetual licenses to subscription arrangements. "
    "Cost of Revenue. Total cost of revenue was $1.68 billion, or 40% of revenue, "
    "compared with $1.55 billion, or 42% of revenue, in the prior year. "
    "Gross profit increased $370 million to $2.52 billion; gross margin expanded 200 basis points. "
    "Operating Expenses. Research and development expense was $630 million (15% of revenue), "
    "sales and marketing was $840 million (20%), and general and administrative was $430 million (10%). "
    "Operating Income. Operating income was $620 million (14.8% margin) versus "
    "$510 million (13.9%) in fiscal 2024."
)

SAMPLE_RISK_FACTORS_TEXT: str = (
    "Item 1A. Risk Factors. "
    "We operate in a highly competitive market; failure to innovate or respond to "
    "competitor offerings could materially harm our business. "
    "Key risks include: "
    "(1) Cybersecurity and data privacy. A breach of our systems or a customer's systems "
    "could expose sensitive data, resulting in regulatory penalties under GDPR, CCPA, and "
    "other data protection regimes, and significant reputational harm. "
    "(2) Macroeconomic conditions. A global recession, rising interest rates, or currency "
    "fluctuations could reduce customer IT spending and lengthen our sales cycles. "
    "(3) Dependence on cloud infrastructure providers. We rely on Amazon Web Services and "
    "Microsoft Azure for a substantial portion of our Cloud Services delivery; any service "
    "disruption or pricing change by these providers could affect our margins. "
    "(4) Regulatory and legal risks. Changes in tax laws, export control regulations, or "
    "antitrust enforcement could impose additional compliance costs or restrict our operations. "
    "(5) Key personnel. The loss of key management or engineering talent could delay product "
    "development and harm our competitive position."
)

ABLATION_TECHNIQUES: list[str] = [
    "baseline",
    "+reranking",
    "+metadata",
    "+query_expansion",
    "+hybrid",
    "+semantic_chunking",
]


def make_document(
    doc_id: str = "doc_001",
    section: str = "10-K",
) -> Document:
    """Return a Document backed by the appropriate sample text.

    Args:
        doc_id: Unique document identifier.
        section: One of '10-K', '10-Q', 'MDA', 'risk_factors'.
            Defaults to '10-K'.
    """
    text_map = {
        "10-K": SAMPLE_10K_TEXT,
        "10-Q": SAMPLE_10Q_TEXT,
        "MDA": SAMPLE_MANDA_TEXT,
        "risk_factors": SAMPLE_RISK_FACTORS_TEXT,
    }
    text = text_map.get(section, SAMPLE_10K_TEXT)
    return Document(
        doc_id=doc_id,
        text=text,
        metadata={
            "source": section,
            "fiscal_year": 2025,
            "company": "Acme Corporation",
            "filing_date": "2026-02-15",
            "section": section,
        },
        source=section,
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
        "+semantic_chunking": 0.10,
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


def make_qa_pairs(seed: int = 42) -> list[dict]:
    """Return realistic 10-K / 10-Q QA pairs with gold answer spans.

    Each element is a dict with:
    - 'question': question text
    - 'answer': expected short answer
    - 'spans': list of supporting text spans from the sample documents
    - 'section': source section label
    """
    return [
        {
            "question": "What was Acme's total revenue in fiscal year 2025?",
            "answer": "$4.2 billion",
            "spans": ["total revenue increased 14% year-over-year to $4.2 billion"],
            "section": "10-K",
        },
        {
            "question": "By how much did Cloud Services revenue grow in fiscal 2025?",
            "answer": "31% to $2.8 billion",
            "spans": ["31% increase in Cloud Services revenue to $2.8 billion"],
            "section": "10-K",
        },
        {
            "question": "What was the operating income margin in fiscal 2025?",
            "answer": "14.8%",
            "spans": ["Operating income for fiscal 2025 was $620 million, a margin of 14.8%"],
            "section": "MDA",
        },
        {
            "question": "What were the main cybersecurity risks disclosed?",
            "answer": "ransomware attacks, regulatory fines, reputational harm",
            "spans": [
                "Cybersecurity incidents, including ransomware attacks, represent a material risk",
                "could result in regulatory fines and reputational harm",
            ],
            "section": "risk_factors",
        },
        {
            "question": "What was free cash flow in fiscal 2025?",
            "answer": "$540 million",
            "spans": ["Free cash flow was $540 million, up from $430 million in the prior year"],
            "section": "10-K",
        },
        {
            "question": "What were net revenues in Q3 2025?",
            "answer": "$1.12 billion",
            "spans": ["net revenue of $1.12 billion, a 16% increase over the prior-year quarter"],
            "section": "10-Q",
        },
        {
            "question": "What cloud providers does Acme depend on?",
            "answer": "Amazon Web Services and Microsoft Azure",
            "spans": [
                "We rely on Amazon Web Services and Microsoft Azure for a substantial portion"
            ],
            "section": "risk_factors",
        },
    ]
