"""FinDocRetrieval: Financial document retrieval and QA evaluation toolkit."""
from .core import (
    Document,
    Chunk,
    SemanticChunk,
    QueryResult,
    ChunkingConfig,
    fixed_size_chunker,
    paragraph_chunker,
    sentence_chunker,
    semantic_chunker,
)
from .evaluate import (
    tokenize,
    exact_match,
    f1_score_tokens,
    answer_recall,
    span_precision,
    semantic_f1_score,
    numeric_accuracy,
    table_extraction_score,
    cost_per_f1_point,
    marginal_gain,
    ablation_summary,
)

__all__ = [
    "Document",
    "Chunk",
    "SemanticChunk",
    "QueryResult",
    "ChunkingConfig",
    "fixed_size_chunker",
    "paragraph_chunker",
    "sentence_chunker",
    "semantic_chunker",
    "tokenize",
    "exact_match",
    "f1_score_tokens",
    "answer_recall",
    "span_precision",
    "semantic_f1_score",
    "numeric_accuracy",
    "table_extraction_score",
    "cost_per_f1_point",
    "marginal_gain",
    "ablation_summary",
]
__version__ = "0.2.0"
