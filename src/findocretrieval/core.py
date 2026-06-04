from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import uuid


@dataclass
class Document:
    doc_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    source: str = "10-K"


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    start_char: int
    end_char: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        """Number of characters in this chunk."""
        return self.end_char - self.start_char


@dataclass
class QueryResult:
    query_id: str
    query: str
    retrieved_chunks: list[Chunk]
    answer: str
    cost_usd: float = 0.0


@dataclass
class ChunkingConfig:
    strategy: str
    chunk_size: int = 512
    overlap: int = 64

    def __post_init__(self) -> None:
        if self.overlap >= self.chunk_size:
            raise ValueError(
                f"overlap ({self.overlap}) must be less than chunk_size ({self.chunk_size})"
            )


def fixed_size_chunker(doc: Document, config: ChunkingConfig) -> list[Chunk]:
    """Split *doc.text* into fixed-size character chunks with overlap.

    The step between chunk starts is ``chunk_size - overlap``.  Each chunk
    carries start/end character offsets into the original text.
    """
    text = doc.text
    step = config.chunk_size - config.overlap
    chunks: list[Chunk] = []
    start = 0
    idx = 0
    while start < len(text):
        end = min(start + config.chunk_size, len(text))
        chunk_text = text[start:end]
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}_chunk_{idx:04d}",
                doc_id=doc.doc_id,
                text=chunk_text,
                start_char=start,
                end_char=end,
                metadata=dict(doc.metadata),
            )
        )
        if end == len(text):
            break
        start += step
        idx += 1
    return chunks


def sentence_chunker(doc: Document, max_chars: int = 512) -> list[Chunk]:
    """Split *doc.text* on sentence boundaries ('. '), respecting *max_chars*.

    Sentences are greedily packed into chunks.  A sentence that exceeds
    *max_chars* on its own is placed in a chunk by itself.
    """
    text = doc.text
    # Split on ". " keeping the period attached to the preceding sentence.
    raw_sentences = text.split(". ")
    sentences: list[str] = []
    for i, s in enumerate(raw_sentences):
        # Re-attach the period unless it is the last fragment.
        sentences.append(s + "." if i < len(raw_sentences) - 1 else s)

    chunks: list[Chunk] = []
    current_text = ""
    current_start = 0
    idx = 0
    char_pos = 0  # tracks absolute position in the original text

    for sentence in sentences:
        # +1 for the space that was consumed by split
        sentence_len = len(sentence)
        if current_text and len(current_text) + 1 + sentence_len > max_chars:
            # Flush current chunk
            end_char = current_start + len(current_text)
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}_chunk_{idx:04d}",
                    doc_id=doc.doc_id,
                    text=current_text,
                    start_char=current_start,
                    end_char=end_char,
                    metadata=dict(doc.metadata),
                )
            )
            idx += 1
            current_start = end_char + 1  # skip the space
            current_text = sentence
        else:
            current_text = current_text + " " + sentence if current_text else sentence

    # Flush remaining text
    if current_text:
        end_char = current_start + len(current_text)
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}_chunk_{idx:04d}",
                doc_id=doc.doc_id,
                text=current_text,
                start_char=current_start,
                end_char=end_char,
                metadata=dict(doc.metadata),
            )
        )

    return chunks
