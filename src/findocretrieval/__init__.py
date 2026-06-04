"""findocretrieval: Financial Document Retrieval ablation over FinanceBench 10-K filings."""

__version__ = "0.1.0"

from findocretrieval.core import (
    Chunk,
    ChunkingConfig,
    Document,
    QueryResult,
    fixed_size_chunker,
    sentence_chunker,
)

__all__ = [
    "Document",
    "Chunk",
    "QueryResult",
    "ChunkingConfig",
    "fixed_size_chunker",
    "sentence_chunker",
]
