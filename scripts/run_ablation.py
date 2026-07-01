#!/usr/bin/env python3
"""Run the 7-condition ablation study (C0-C6) on FinanceBench.

Usage examples
--------------
# Local Ollama (no API key required):
python scripts/run_ablation.py --llm-provider ollama --llm-model llama3

# Anthropic (requires ANTHROPIC_API_KEY):
python scripts/run_ablation.py --llm-provider anthropic --llm-model claude-haiku-4-5

# OpenAI (requires OPENAI_API_KEY):
python scripts/run_ablation.py --llm-provider openai --llm-model gpt-4o-mini

# Dry-run on 5 rows:
python scripts/run_ablation.py --sample 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from findocretrieval.chunking import (  # noqa: E402
    build_index,
    fixed_chunking,
    recursive_chunking,
    semantic_chunking,
)
from findocretrieval.evaluate import evaluate_condition  # noqa: E402
from findocretrieval.query_expansion import HyDERetriever  # noqa: E402
from findocretrieval.retriever import (  # noqa: E402
    get_base_retriever,
    get_hybrid_retriever,
    get_metadata_retriever,
    get_reranking_retriever,
)

_INDEX_BASE = PROJECT_ROOT / "data" / "index"

# Human-readable descriptions for the summary table
_CONDITIONS = {
    "C0_fixed_base": "Fixed chunking + base retriever (baseline)",
    "C1_fixed_rerank": "Fixed chunking + CrossEncoder reranking",
    "C2_fixed_metadata": "Fixed chunking + metadata filtering",
    "C3_fixed_hyde": "Fixed chunking + HyDE query expansion",
    "C4_fixed_hybrid": "Fixed chunking + hybrid BM25+dense+rerank",
    "C5_semantic_base": "Semantic chunking + base retriever",
    "C6_recursive_base": "Recursive chunking + base retriever",
}


# ---------------------------------------------------------------------------
# PDF loading
# ---------------------------------------------------------------------------

def load_pdfs(pdf_dir: Path) -> list:
    """Load all *.pdf files from *pdf_dir* as LangChain Documents."""
    from langchain_community.document_loaders import PyPDFLoader

    all_pages = []
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        try:
            pages = PyPDFLoader(str(pdf_path)).load()
            meta = _parse_pdf_meta(pdf_path)
            for p in pages:
                p.metadata.update(meta)
            all_pages.extend(pages)
            print(f"[load] {pdf_path.name} → {len(pages)} pages")
        except Exception as exc:
            print(f"[warn] skipping {pdf_path.name}: {exc}")
    return all_pages


def _parse_pdf_meta(pdf_path: Path) -> dict:
    """Extract company, doc_period, doc_type from filenames like '3M_2018_10K.pdf'."""
    parts = pdf_path.stem.split("_")
    return {
        "company": parts[0] if parts else "unknown",
        "doc_period": next((p for p in parts if p.isdigit() and len(p) == 4), "unknown"),
        "doc_type": parts[-1].lower() if len(parts) >= 3 else "unknown",
        "source": str(pdf_path),
    }


# ---------------------------------------------------------------------------
# Index building
# ---------------------------------------------------------------------------

def build_all_indices(pages: list) -> dict[str, Path]:
    """Chunk *pages* with all three strategies and persist FAISS indices."""
    indices: dict[str, Path] = {}
    for name, chunk_fn in [
        ("fixed", fixed_chunking),
        ("semantic", semantic_chunking),
        ("recursive", recursive_chunking),
    ]:
        chunks = chunk_fn(pages)
        print(f"[chunk] {name}: {len(chunks)} chunks")
        build_index(chunks, name)
        indices[name] = _INDEX_BASE / name
    return indices


# ---------------------------------------------------------------------------
# Per-row metadata-retriever factory (C2)
# ---------------------------------------------------------------------------

def _metadata_retriever_fn(index_path: Path):
    """Return a callable(row) -> retriever that filters by company + doc_period."""
    def fn(row):
        return get_metadata_retriever(
            index_path,
            company=str(row.get("company", "")),
            doc_period=str(row.get("doc_period", "")),
        )
    return fn


# ---------------------------------------------------------------------------
# Main ablation runner
# ---------------------------------------------------------------------------

def run_ablation(
    df: pd.DataFrame,
    llm,
    pdf_dir: Path,
    results_dir: Path,
) -> pd.DataFrame:
    """Build indices, run all 7 conditions, save per-condition CSVs."""
    results_dir.mkdir(parents=True, exist_ok=True)

    pages = load_pdfs(pdf_dir)
    if not pages:
        raise RuntimeError(f"No PDFs found in {pdf_dir}")
    print(f"\n[ablation] {len(pages)} pages loaded\n")

    indices = build_all_indices(pages)
    fixed_chunks = fixed_chunking(pages)  # kept in memory for hybrid BM25

    all_results: list[pd.DataFrame] = []

    def _run(name: str, retriever) -> None:
        print(f"\n=== {name}: {_CONDITIONS[name]} ===")
        res = evaluate_condition(name, retriever, llm, df)
        res.to_csv(results_dir / f"{name}.csv", index=False)
        all_results.append(res)
        mean_f1 = res["rouge_f1"].mean()
        print(f"    mean ROUGE-L F1 = {mean_f1:.3f}  (n={len(res)})")

    # C0 — baseline
    _run("C0_fixed_base", get_base_retriever(indices["fixed"]))

    # C1 — reranking
    _run("C1_fixed_rerank", get_reranking_retriever(indices["fixed"]))

    # C2 — metadata filter (per-row retriever factory)
    _run("C2_fixed_metadata", _metadata_retriever_fn(indices["fixed"]))

    # C3 — HyDE
    from langchain_community.vectorstores import FAISS
    from findocretrieval.embeddings import get_embedder
    vs = FAISS.load_local(
        str(indices["fixed"]), get_embedder(), allow_dangerous_deserialization=True
    )
    _run("C3_fixed_hyde", HyDERetriever(vectorstore=vs, llm=llm, k=5))

    # C4 — hybrid BM25 + dense + rerank
    _run("C4_fixed_hybrid", get_hybrid_retriever(fixed_chunks, indices["fixed"]))

    # C5 — semantic chunking
    _run("C5_semantic_base", get_base_retriever(indices["semantic"]))

    # C6 — recursive chunking
    _run("C6_recursive_base", get_base_retriever(indices["recursive"]))

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(results_dir / "all_conditions.csv", index=False)
    print(f"\n[ablation] Results saved to {results_dir}/")
    return combined


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(combined: pd.DataFrame) -> None:
    try:
        from rich.console import Console
        from rich.table import Table

        tbl = Table(title="Ablation Summary")
        tbl.add_column("Condition", style="cyan")
        tbl.add_column("Description")
        tbl.add_column("Mean ROUGE-L F1", justify="right", style="green")
        tbl.add_column("Mean Cost (USD)", justify="right")
        tbl.add_column("N", justify="right")

        for cname, desc in _CONDITIONS.items():
            sub = combined[combined["condition"] == cname]
            if sub.empty:
                continue
            tbl.add_row(
                cname,
                desc,
                f"{sub['rouge_f1'].mean():.3f}",
                f"${sub['cost_usd'].mean():.4f}",
                str(len(sub)),
            )
        Console().print(tbl)

    except ImportError:
        header = f"{'Condition':<25} {'ROUGE-L F1':>12} {'Cost':>10} {'N':>5}"
        print(f"\n{header}")
        print("-" * len(header))
        for cname in _CONDITIONS:
            sub = combined[combined["condition"] == cname]
            if sub.empty:
                continue
            print(
                f"{cname:<25} {sub['rouge_f1'].mean():>12.3f} "
                f"${sub['cost_usd'].mean():>8.4f} {len(sub):>5}"
            )


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def build_llm(provider: str, model: str = ""):
    """Instantiate a LangChain LLM for the given provider.

    Providers
    ---------
    ollama    — local, no API key required (default)
    anthropic — requires ANTHROPIC_API_KEY env var
    openai    — requires OPENAI_API_KEY env var
    """
    if provider == "ollama":
        from langchain_ollama import OllamaLLM
        return OllamaLLM(model=model or "llama3")

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model or "claude-haiku-4-5")

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model or "gpt-4o-mini")

    raise ValueError(f"Unknown provider '{provider}'. Choose: ollama, anthropic, openai")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FinanceBench RAG ablation study (C0-C6)")
    parser.add_argument(
        "--csv",
        default=str(PROJECT_ROOT / "financebench_sample.csv"),
        help="Path to FinanceBench CSV (must have 'question' and 'answer' columns)",
    )
    parser.add_argument(
        "--pdf-dir",
        default=str(PROJECT_ROOT / "data" / "pdfs"),
        help="Directory of financial PDF filings",
    )
    parser.add_argument(
        "--results-dir",
        default=str(PROJECT_ROOT / "results"),
        help="Output directory for CSV results (created if absent)",
    )
    parser.add_argument(
        "--llm-provider",
        default="ollama",
        choices=["ollama", "anthropic", "openai"],
        help="LLM backend (default: ollama — local, no API key required)",
    )
    parser.add_argument("--llm-model", default="", help="Model name override")
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Limit to first N questions (0 = use all)",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if args.sample > 0:
        df = df.head(args.sample)
    print(f"[main] {len(df)} questions from {args.csv}")

    llm = build_llm(args.llm_provider, args.llm_model)
    print(f"[main] LLM provider: {args.llm_provider}  model: {args.llm_model or '(default)'}")

    combined = run_ablation(
        df=df,
        llm=llm,
        pdf_dir=Path(args.pdf_dir),
        results_dir=Path(args.results_dir),
    )
    print_summary(combined)
