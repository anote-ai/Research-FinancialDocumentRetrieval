from __future__ import annotations
import pytest
from findocretrieval.core import (
    Document,
    Chunk,
    QueryResult,
    ChunkingConfig,
    fixed_size_chunker,
    sentence_chunker,
)


# --- Document ---

def test_document_construction() -> None:
    doc = Document(doc_id="d1", text="Hello world", source="10-K")
    assert doc.doc_id == "d1"
    assert doc.source == "10-K"


# --- Chunk ---

def test_chunk_length() -> None:
    chunk = Chunk(chunk_id="c1", doc_id="d1", text="Hello", start_char=0, end_char=5)
    assert chunk.length == 5


# --- ChunkingConfig ---

def test_chunking_config_defaults() -> None:
    cfg = ChunkingConfig(strategy="fixed")
    assert cfg.chunk_size == 512
    assert cfg.overlap == 64


def test_chunking_config_overlap_ge_chunk_size_raises() -> None:
    with pytest.raises(ValueError):
        ChunkingConfig(strategy="fixed", chunk_size=100, overlap=100)


def test_chunking_config_overlap_exceeds_chunk_size_raises() -> None:
    with pytest.raises(ValueError):
        ChunkingConfig(strategy="fixed", chunk_size=100, overlap=150)


# --- fixed_size_chunker ---

def _make_doc(text: str = "A" * 1000) -> Document:
    return Document(doc_id="doc_test", text=text)


def test_fixed_size_chunker_returns_chunks() -> None:
    doc = _make_doc("X" * 300)
    cfg = ChunkingConfig(strategy="fixed", chunk_size=100, overlap=10)
    chunks = fixed_size_chunker(doc, cfg)
    assert len(chunks) > 0


def test_fixed_size_chunker_chunk_count() -> None:
    text = "A" * 500
    doc = Document(doc_id="d", text=text)
    cfg = ChunkingConfig(strategy="fixed", chunk_size=200, overlap=0)
    chunks = fixed_size_chunker(doc, cfg)
    # 500 / 200 = 2.5 → 3 chunks
    assert len(chunks) == 3


def test_fixed_size_chunker_covers_original() -> None:
    text = "Hello world this is a test sentence for chunking purposes."
    doc = Document(doc_id="d", text=text)
    cfg = ChunkingConfig(strategy="fixed", chunk_size=20, overlap=5)
    chunks = fixed_size_chunker(doc, cfg)
    # First chunk should start at 0
    assert chunks[0].start_char == 0
    # Last chunk should end at len(text)
    assert chunks[-1].end_char == len(text)


def test_fixed_size_chunker_bounds() -> None:
    text = "A" * 250
    doc = Document(doc_id="d", text=text)
    cfg = ChunkingConfig(strategy="fixed", chunk_size=100, overlap=20)
    for chunk in fixed_size_chunker(doc, cfg):
        assert chunk.start_char >= 0
        assert chunk.end_char <= len(text)
        assert chunk.start_char < chunk.end_char


def test_fixed_size_chunker_doc_ids() -> None:
    doc = Document(doc_id="mydoc", text="B" * 300)
    cfg = ChunkingConfig(strategy="fixed", chunk_size=100, overlap=10)
    for chunk in fixed_size_chunker(doc, cfg):
        assert chunk.doc_id == "mydoc"


# --- sentence_chunker ---

def test_sentence_chunker_returns_chunks() -> None:
    text = "The revenue was high. The costs were low. The profit increased."
    doc = Document(doc_id="d", text=text)
    chunks = sentence_chunker(doc, max_chars=50)
    assert len(chunks) >= 1


def test_sentence_chunker_doc_ids() -> None:
    text = "First sentence. Second sentence. Third sentence."
    doc = Document(doc_id="sdoc", text=text)
    for chunk in sentence_chunker(doc, max_chars=30):
        assert chunk.doc_id == "sdoc"


# --- QueryResult ---

def test_query_result_construction() -> None:
    qr = QueryResult(
        query_id="q1",
        query="What was the revenue?",
        retrieved_chunks=[],
        answer="$4.2 billion",
        cost_usd=0.005,
    )
    assert qr.query_id == "q1"
    assert qr.cost_usd == pytest.approx(0.005)
