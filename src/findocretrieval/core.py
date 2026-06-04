"""Core data structures and chunking functions for financial document retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


@dataclass
class Document:
    """A single financial document (e.g. a 10-K filing)."""

    doc_id: str
    text: str
    metadata: dict
    source: str = "10-K"


@dataclass
class Chunk:
    """A contiguous text chunk derived from a Document."""

    chunk_id: str
    doc_id: str
    text: str
    start_char: int
    end_char: int
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.start_char < 0:
            raise ValueError("start_char must be >= 0")
        if self.end_char < self.start_char:
            raise ValueError("end_char must be >= start_char")


@dataclass
class QueryResult:
    """Result of a single retrieval query."""

    query_id: str
    query: str
    retrieved_chunks: list[Chunk]
    answer: str
    cost_usd: float


@dataclass
class ChunkingConfig:
    """Parameters controlling fixed-size chunking."""

    strategy: str
    chunk_size: int
    overlap: int

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.overlap < 0:
            raise ValueError("overlap must be non-negative")
        if self.overlap >= self.chunk_size:
            raise ValueError("overlap must be less than chunk_size")


def fixed_size_chunker(doc: Document, config: ChunkingConfig) -> list[Chunk]:
    """Split a document into fixed-size character chunks with configurable overlap.

    Args:
        doc: Source document.
        config: Chunking parameters (chunk_size, overlap).

    Returns:
        Ordered list of Chunk objects.
    """
    text = doc.text
    size = config.chunk_size
    step = size - config.overlap
    chunks: list[Chunk] = []
    start = 0
    idx = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunk = Chunk(
            chunk_id=f"{doc.doc_id}_chunk_{idx}",
            doc_id=doc.doc_id,
            text=text[start:end],
            start_char=start,
            end_char=end,
            metadata={**doc.metadata},
        )
        chunks.append(chunk)
        if end == len(text):
            break
        start += step
        idx += 1
    return chunks


def sentence_chunker(doc: Document) -> list[Chunk]:
    """Split a document into sentence-level chunks.

    Stub implementation — splits on '. ' as a simple heuristic.
    Replace with a proper sentence tokeniser for production use.
    """
    sentences = doc.text.split(". ")
    chunks: list[Chunk] = []
    cursor = 0
    for idx, sentence in enumerate(sentences):
        end = cursor + len(sentence)
        chunk = Chunk(
            chunk_id=f"{doc.doc_id}_sent_{idx}",
            doc_id=doc.doc_id,
            text=sentence,
            start_char=cursor,
            end_char=end,
            metadata={**doc.metadata},
        )
        chunks.append(chunk)
        cursor = end + 2  # account for ". "
    return chunks
