from __future__ import annotations
import pytest
from findocretrieval.core import (
    Document,
    Chunk,
    SemanticChunk,
    ChunkingConfig,
    fixed_size_chunker,
    paragraph_chunker,
    sentence_chunker,
    semantic_chunker,
)


# ---------------------------------------------------------------------------
# SemanticChunk
# ---------------------------------------------------------------------------

def test_semantic_chunk_coherence_score_both_neighbours() -> None:
    chunk = SemanticChunk(
        chunk_id="c0",
        doc_id="d0",
        text="hello world",
        start_char=0,
        end_char=11,
        similarity_to_prev=0.8,
        similarity_to_next=0.6,
    )
    assert chunk.coherence_score == pytest.approx(0.7)


def test_semantic_chunk_coherence_score_no_neighbours() -> None:
    chunk = SemanticChunk(
        chunk_id="c0",
        doc_id="d0",
        text="solo",
        start_char=0,
        end_char=4,
    )
    assert chunk.coherence_score == pytest.approx(0.0)


def test_semantic_chunk_is_boundary_true() -> None:
    chunk = SemanticChunk(
        chunk_id="c0",
        doc_id="d0",
        text="text",
        start_char=0,
        end_char=4,
        similarity_to_prev=0.3,
        similarity_to_next=0.8,
    )
    assert chunk.is_boundary(threshold=0.5) is True


def test_semantic_chunk_is_boundary_false() -> None:
    chunk = SemanticChunk(
        chunk_id="c0",
        doc_id="d0",
        text="text",
        start_char=0,
        end_char=4,
        similarity_to_prev=0.7,
        similarity_to_next=0.9,
    )
    assert chunk.is_boundary(threshold=0.5) is False


# ---------------------------------------------------------------------------
# semantic_chunker
# ---------------------------------------------------------------------------

def _make_doc(text: str = "Hello world. Foo bar. Baz qux.") -> Document:
    return Document(doc_id="test_doc", text=text, source="10-K")


def test_semantic_chunker_returns_semantic_chunks() -> None:
    doc = _make_doc()
    chunks = semantic_chunker(doc)
    assert all(isinstance(c, SemanticChunk) for c in chunks)


def test_semantic_chunker_non_empty() -> None:
    doc = _make_doc()
    chunks = semantic_chunker(doc)
    assert len(chunks) >= 1


def test_semantic_chunker_section_label() -> None:
    from findocretrieval.data import SAMPLE_10K_TEXT

    doc = Document(doc_id="doc_10k", text=SAMPLE_10K_TEXT, source="10-K")
    chunks = semantic_chunker(doc)
    labelled = [c for c in chunks if c.section_label is not None]
    assert len(labelled) >= 1


def test_semantic_chunker_first_chunk_no_prev() -> None:
    doc = _make_doc()
    chunks = semantic_chunker(doc)
    assert chunks[0].similarity_to_prev is None


def test_semantic_chunker_last_chunk_no_next() -> None:
    doc = _make_doc()
    chunks = semantic_chunker(doc)
    if len(chunks) > 1:
        assert chunks[-1].similarity_to_next is None


# ---------------------------------------------------------------------------
# Existing chunker tests
# ---------------------------------------------------------------------------

def test_fixed_size_chunker_lengths() -> None:
    doc = Document(doc_id="d", text="a" * 1000, source="10-K")
    config = ChunkingConfig(strategy="fixed", chunk_size=200, overlap=50)
    chunks = fixed_size_chunker(doc, config)
    assert all(c.length <= 200 for c in chunks)
    assert len(chunks) > 1


def test_sentence_chunker_basic() -> None:
    doc = Document(
        doc_id="d",
        text="First sentence. Second sentence. Third sentence.",
        source="10-K",
    )
    chunks = sentence_chunker(doc, max_chars=30)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert isinstance(chunk, Chunk)


def test_chunking_config_overlap_validation() -> None:
    with pytest.raises(ValueError, match="overlap"):
        ChunkingConfig(strategy="fixed", chunk_size=100, overlap=100)


# ---------------------------------------------------------------------------
# paragraph_chunker
# ---------------------------------------------------------------------------

def _make_para_doc(n_paragraphs: int = 3, words_per_para: int = 20) -> Document:
    paras = [" ".join([f"word{i}" for i in range(words_per_para)]) for _ in range(n_paragraphs)]
    return Document(doc_id="para_doc", text="\n\n".join(paras), source="10-K")


def test_paragraph_chunker_returns_chunks() -> None:
    doc = _make_para_doc(3)
    chunks = paragraph_chunker(doc)
    assert all(isinstance(c, Chunk) for c in chunks)


def test_paragraph_chunker_non_empty() -> None:
    doc = _make_para_doc(3)
    chunks = paragraph_chunker(doc)
    assert len(chunks) >= 1


def test_paragraph_chunker_single_paragraph() -> None:
    doc = Document(doc_id="d", text="No paragraph breaks here at all.", source="10-K")
    chunks = paragraph_chunker(doc)
    assert len(chunks) == 1
    assert chunks[0].text == "No paragraph breaks here at all."


def test_paragraph_chunker_respects_max_chars() -> None:
    long_para = "x " * 600  # ~1200 chars each
    doc = Document(doc_id="d", text=f"{long_para}\n\n{long_para}\n\n{long_para}", source="10-K")
    chunks = paragraph_chunker(doc, max_chars=1500)
    assert len(chunks) >= 2
    for c in chunks:
        assert len(c.text) <= 1500 + 10  # small tolerance for separator


def test_paragraph_chunker_preserves_text() -> None:
    doc = _make_para_doc(2)
    chunks = paragraph_chunker(doc)
    recovered = "\n\n".join(c.text for c in chunks)
    assert recovered == doc.text


def test_paragraph_chunker_chunk_ids_unique() -> None:
    doc = _make_para_doc(4)
    chunks = paragraph_chunker(doc)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_paragraph_chunker_empty_text() -> None:
    doc = Document(doc_id="d", text="", source="10-K")
    chunks = paragraph_chunker(doc)
    assert chunks == []


def test_paragraph_chunker_metadata_copied() -> None:
    doc = Document(doc_id="d", text="Para one.\n\nPara two.", source="10-K", metadata={"fiscal_year": 2025})
    chunks = paragraph_chunker(doc)
    for c in chunks:
        assert c.metadata.get("fiscal_year") == 2025
