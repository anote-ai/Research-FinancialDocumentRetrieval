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


# ---------------------------------------------------------------------------
# LangChain end-to-end evaluation (LangChain imports are lazy so the
# pure-Python metrics above remain importable without LangChain installed)
# ---------------------------------------------------------------------------

def evaluate_condition(
    condition_name: str,
    retriever,
    llm,
    df,
) -> "pd.DataFrame":
    """Run every row in *df* through retriever + LLM and return per-row metrics.

    Args:
        condition_name: Label for the ablation condition (e.g. "C0_fixed_base").
        retriever: A LangChain BaseRetriever, **or** a callable(pd.Series) ->
                   BaseRetriever for per-row retriever creation (needed for the
                   metadata-filtered C2 condition).
        llm: Any LangChain BaseLLM / BaseChatModel.
        df: DataFrame with at least columns ``question`` and ``answer``.

    Returns:
        DataFrame with columns: condition, question, gold_answer,
        predicted_answer, rouge_f1, exact_match, cost_usd, latency_s.
    """
    import time

    import pandas as pd
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import RunnableLambda, RunnablePassthrough

    try:
        from rouge_score import rouge_scorer as _rouge_scorer_mod
        _scorer = _rouge_scorer_mod.RougeScorer(["rougeL"], use_stemmer=True)
    except ImportError:
        _scorer = None

    _QA_PROMPT = PromptTemplate.from_template(
        "You are a financial analyst. Answer the question using only the provided context.\n"
        "Be concise. If the answer is a number or dollar amount, state it directly.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )

    def _format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    def _rouge_f1(pred: str, gold: str) -> float:
        if _scorer is not None:
            return _scorer.score(gold, pred)["rougeL"].fmeasure
        return f1_score_tokens(pred, gold)

    # Build a reusable chain when the retriever is fixed (not per-row).
    _static_chain = None
    if not callable(retriever):
        _static_chain = (
            {
                "context": retriever | RunnableLambda(_format_docs),
                "question": RunnablePassthrough(),
            }
            | _QA_PROMPT
            | llm
            | StrOutputParser()
        )

    rows = []
    for _, row in df.iterrows():
        question = str(row.get("question", ""))
        gold = str(row.get("answer", ""))

        t0 = time.perf_counter()
        try:
            if callable(retriever):
                cur = retriever(row)
                chain = (
                    {
                        "context": cur | RunnableLambda(_format_docs),
                        "question": RunnablePassthrough(),
                    }
                    | _QA_PROMPT
                    | llm
                    | StrOutputParser()
                )
                predicted = chain.invoke(question)
            else:
                predicted = _static_chain.invoke(question)
        except Exception as exc:
            print(f"[warn] {condition_name} | failed: {exc}")
            predicted = ""

        rows.append({
            "condition": condition_name,
            "question": question,
            "gold_answer": gold,
            "predicted_answer": predicted,
            "rouge_f1": _rouge_f1(predicted, gold),
            "exact_match": exact_match(predicted, gold),
            "cost_usd": 0.0,
            "latency_s": round(time.perf_counter() - t0, 2),
        })

    return pd.DataFrame(rows)
