"""FinDocRetrieval: Financial document retrieval and QA evaluation toolkit."""
from .core import (
    Chunk,
    ChunkingConfig,
    Document,
    QueryResult,
    SemanticChunk,
    fixed_size_chunker,
    paragraph_chunker,
    semantic_chunker,
    sentence_chunker,
)
from .evaluate import (
    ablation_summary,
    answer_recall,
    cost_per_f1_point,
    exact_match,
    f1_score_tokens,
    marginal_gain,
    numeric_accuracy,
    semantic_f1_score,
    span_precision,
    table_extraction_score,
    tokenize,
)

__all__ = [
    "Chunk",
    "ChunkingConfig",
    "Document",
    "QueryResult",
    "SemanticChunk",
    "ablation_summary",
    "answer_recall",
    "cost_per_f1_point",
    "exact_match",
    "f1_score_tokens",
    "fixed_size_chunker",
    "marginal_gain",
    "numeric_accuracy",
    "paragraph_chunker",
    "semantic_chunker",
    "semantic_f1_score",
    "sentence_chunker",
    "span_precision",
    "table_extraction_score",
    "tokenize",
]
__version__ = "0.2.0"
