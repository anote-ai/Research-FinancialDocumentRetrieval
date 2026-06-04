#!/usr/bin/env python3
"""Demo script: financial document chunking and ablation evaluation."""
from __future__ import annotations
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from findocretrieval.core import ChunkingConfig, fixed_size_chunker, sentence_chunker
from findocretrieval.data import make_document, make_query_results, ABLATION_TECHNIQUES
from findocretrieval.evaluate import ablation_summary, f1_score_tokens


def main() -> None:
    print("=== FinDocRetrieval Demo ===")

    doc = make_document()
    print(f"\nDocument: {doc.doc_id} ({len(doc.text)} chars, source={doc.source})")

    # Fixed-size chunking
    cfg = ChunkingConfig(strategy="fixed", chunk_size=200, overlap=40)
    fixed_chunks = fixed_size_chunker(doc, cfg)
    print(f"Fixed-size chunks (size=200, overlap=40): {len(fixed_chunks)}")
    for c in fixed_chunks[:2]:
        print(f"  [{c.start_char}:{c.end_char}] {c.text[:60]}...")

    # Sentence chunking
    sent_chunks = sentence_chunker(doc, max_chars=300)
    print(f"\nSentence chunks (max_chars=300): {len(sent_chunks)}")
    for c in sent_chunks[:2]:
        print(f"  [{c.start_char}:{c.end_char}] {c.text[:60]}...")

    # Ablation summary
    print("\n--- Ablation Summary ---")
    results = make_query_results(n=10, seed=42)
    summary = ablation_summary(results)
    for tech, stats in sorted(summary.items(), key=lambda x: -x[1]["mean_f1"]):
        print(
            f"  {tech:<20} mean_f1={stats['mean_f1']:.4f}  "
            f"mean_cost=${stats['mean_cost']:.4f}  "
            f"marginal_gain={stats['marginal_gain']:+.4f}"
        )


if __name__ == "__main__":
    main()
