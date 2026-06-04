"""FinDocRetrieval: Financial document retrieval and QA evaluation toolkit."""
from .core import (
    Document,
    Chunk,
    QueryResult,
    ChunkingConfig,
    fixed_size_chunker,
    sentence_chunker,
)
from .evaluate import (
    tokenize,
    exact_match,
    f1_score_tokens,
    cost_per_f1_point,
    marginal_gain,
    ablation_summary,
)

__all__ = [
    "Document",
    "Chunk",
    "QueryResult",
    "ChunkingConfig",
    "fixed_size_chunker",
    "sentence_chunker",
    "tokenize",
    "exact_match",
    "f1_score_tokens",
    "cost_per_f1_point",
    "marginal_gain",
    "ablation_summary",
]
__version__ = "0.1.0"
