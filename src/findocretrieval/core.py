from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


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
        return self.end_char - self.start_char


@dataclass
class SemanticChunk(Chunk):
    """A chunk produced by embedding-aware splitting.

    Extends :class:`Chunk` with similarity scores that capture how
    cohesive the chunk is relative to its neighbours.  These hints can
    be used downstream to filter or merge low-coherence chunks before
    indexing.

    Attributes:
        embedding_hint: Optional pre-computed centroid embedding vector
            (e.g. from a sentence-transformer model).  ``None`` when the
            chunk was produced without an encoder.
        similarity_to_prev: Cosine similarity to the preceding chunk's
            embedding, or ``None`` if this is the first chunk.
        similarity_to_next: Cosine similarity to the next chunk's
            embedding, or ``None`` if this is the last chunk.
        section_label: Detected section header (e.g. 'Item 7', 'Risk Factors')
            if available, otherwise ``None``.
    """

    embedding_hint: list[float] | None = None
    similarity_to_prev: float | None = None
    similarity_to_next: float | None = None
    section_label: str | None = None

    @property
    def coherence_score(self) -> float:
        """Mean similarity to neighbours (ignores None neighbours)."""
        values = [
            v
            for v in (self.similarity_to_prev, self.similarity_to_next)
            if v is not None
        ]
        return sum(values) / len(values) if values else 0.0

    def is_boundary(self, threshold: float = 0.5) -> bool:
        """Return True if this chunk marks a topic boundary.

        A chunk is considered a boundary when at least one of its
        neighbour similarities is below *threshold*.
        """
        for sim in (self.similarity_to_prev, self.similarity_to_next):
            if sim is not None and sim < threshold:
                return True
        return False


# Section header patterns found in SEC filings
_SEC_SECTION_HEADERS: list[str] = [
    "Item 1.",
    "Item 1A.",
    "Item 1B.",
    "Item 2.",
    "Item 3.",
    "Item 4.",
    "Item 5.",
    "Item 6.",
    "Item 7.",
    "Item 7A.",
    "Item 8.",
    "Item 9.",
    "Item 9A.",
]


def _detect_section(text: str) -> str | None:
    """Return the first matching SEC section header found in *text*, or None."""
    for header in _SEC_SECTION_HEADERS:
        if header in text:
            return header
    return None


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
    """Split *doc.text* into fixed-size character chunks with overlap."""
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
    """Split *doc.text* on sentence boundaries ('. '), respecting *max_chars*."""
    text = doc.text
    raw_sentences = text.split(". ")
    sentences: list[str] = []
    for i, s in enumerate(raw_sentences):
        sentences.append(s + "." if i < len(raw_sentences) - 1 else s)

    chunks: list[Chunk] = []
    current_text = ""
    current_start = 0
    idx = 0

    for sentence in sentences:
        sentence_len = len(sentence)
        if current_text and len(current_text) + 1 + sentence_len > max_chars:
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
            current_start = end_char + 1
            current_text = sentence
        else:
            current_text = current_text + " " + sentence if current_text else sentence

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


def semantic_chunker(
    doc: Document,
    similarity_threshold: float = 0.5,
    max_chars: int = 1024,
) -> list[SemanticChunk]:
    """Split *doc.text* on sentence boundaries with SemanticChunk metadata.

    Since embedding models are not available at import time this function
    uses character-length heuristics to assign synthetic similarity scores.
    When integrated with a real encoder, callers can replace the heuristic
    values with actual cosine similarities.

    Args:
        doc: Source document to split.
        similarity_threshold: Boundary detection threshold passed through to
            :meth:`SemanticChunk.is_boundary`.
        max_chars: Maximum characters per chunk before forcing a split.

    Returns:
        List of :class:`SemanticChunk` with section labels and heuristic
        similarity scores filled in.
    """
    raw_sentences = doc.text.split(". ")
    sentences: list[str] = [
        s + "." if i < len(raw_sentences) - 1 else s
        for i, s in enumerate(raw_sentences)
    ]

    raw_chunks: list[str] = []
    current = ""
    for sent in sentences:
        if current and len(current) + 1 + len(sent) > max_chars:
            raw_chunks.append(current)
            current = sent
        else:
            current = current + " " + sent if current else sent
    if current:
        raw_chunks.append(current)

    sem_chunks: list[SemanticChunk] = []
    char_pos = 0
    for idx, text in enumerate(raw_chunks):
        start = char_pos
        end = start + len(text)
        # Heuristic: similarity inversely proportional to length difference
        sim_prev: float | None = None
        sim_next: float | None = None
        if idx > 0:
            prev_len = len(raw_chunks[idx - 1])
            ratio = min(len(text), prev_len) / max(len(text), prev_len)
            sim_prev = round(0.4 + 0.5 * ratio, 4)
        if idx < len(raw_chunks) - 1:
            next_len = len(raw_chunks[idx + 1])
            ratio = min(len(text), next_len) / max(len(text), next_len)
            sim_next = round(0.4 + 0.5 * ratio, 4)

        meta = dict(doc.metadata)
        section = _detect_section(text)
        if section:
            meta["section"] = section

        sem_chunks.append(
            SemanticChunk(
                chunk_id=f"{doc.doc_id}_sem_{idx:04d}",
                doc_id=doc.doc_id,
                text=text,
                start_char=start,
                end_char=end,
                metadata=meta,
                similarity_to_prev=sim_prev,
                similarity_to_next=sim_next,
                section_label=section,
            )
        )
        char_pos = end + 1  # account for separator

    return sem_chunks
