"""Tests for findocretrieval.core."""

import pytest

from findocretrieval.core import (
    Chunk,
    ChunkingConfig,
    Document,
    QueryResult,
    fixed_size_chunker,
)


# --- Document ---

def test_document_construction():
    doc = Document(doc_id="doc1", text="Some 10-K text.", metadata={"year": 2023})
    assert doc.doc_id == "doc1"
    assert doc.source == "10-K"
    assert doc.metadata["year"] == 2023


def test_document_custom_source():
    doc = Document(doc_id="d2", text="text", metadata={}, source="10-Q")
    assert doc.source == "10-Q"


# --- Chunk ---

def test_chunk_construction():
    chunk = Chunk(chunk_id="c0", doc_id="doc1", text="hello", start_char=0, end_char=5)
    assert chunk.chunk_id == "c0"
    assert chunk.metadata == {}


def test_chunk_invalid_start_char():
    with pytest.raises(ValueError, match="start_char"):
        Chunk(chunk_id="c0", doc_id="d", text="x", start_char=-1, end_char=1)


def test_chunk_invalid_end_char():
    with pytest.raises(ValueError, match="end_char"):
        Chunk(chunk_id="c0", doc_id="d", text="x", start_char=5, end_char=3)


# --- ChunkingConfig ---

def test_chunking_config_valid():
    cfg = ChunkingConfig(strategy="fixed", chunk_size=512, overlap=64)
    assert cfg.chunk_size == 512


def test_chunking_config_overlap_gte_size():
    with pytest.raises(ValueError, match="overlap"):
        ChunkingConfig(strategy="fixed", chunk_size=100, overlap=100)


# --- fixed_size_chunker ---

def test_fixed_size_chunker_basic():
    text = "a" * 1000
    doc = Document(doc_id="d", text=text, metadata={})
    cfg = ChunkingConfig(strategy="fixed", chunk_size=200, overlap=0)
    chunks = fixed_size_chunker(doc, cfg)
    assert len(chunks) == 5
    for chunk in chunks:
        assert chunk.doc_id == "d"


def test_fixed_size_chunker_overlap():
    text = "x" * 300
    doc = Document(doc_id="d", text=text, metadata={})
    cfg = ChunkingConfig(strategy="fixed", chunk_size=100, overlap=50)
    chunks = fixed_size_chunker(doc, cfg)
    # step = 50, so chunks start at 0, 50, 100, 150, 200, 250
    assert len(chunks) == 6
    assert chunks[1].start_char == 50


def test_fixed_size_chunker_short_text():
    doc = Document(doc_id="d", text="short", metadata={})
    cfg = ChunkingConfig(strategy="fixed", chunk_size=512, overlap=0)
    chunks = fixed_size_chunker(doc, cfg)
    assert len(chunks) == 1
    assert chunks[0].text == "short"


def test_fixed_size_chunker_chunk_ids_unique():
    doc = Document(doc_id="d", text="z" * 500, metadata={})
    cfg = ChunkingConfig(strategy="fixed", chunk_size=100, overlap=0)
    chunks = fixed_size_chunker(doc, cfg)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


# --- QueryResult ---

def test_query_result_construction():
    chunk = Chunk(chunk_id="c0", doc_id="d", text="text", start_char=0, end_char=4)
    qr = QueryResult(
        query_id="q1",
        query="What is revenue?",
        retrieved_chunks=[chunk],
        answer="$1B",
        cost_usd=0.002,
    )
    assert qr.query_id == "q1"
    assert len(qr.retrieved_chunks) == 1
    assert qr.cost_usd == pytest.approx(0.002)
