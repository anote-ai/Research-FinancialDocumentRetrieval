"""Three LangChain-based chunking strategies plus FAISS index builder."""
from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter

from .embeddings import get_embedder

_INDEX_BASE = Path(__file__).resolve().parents[3] / "data" / "index"


# ---------------------------------------------------------------------------
# Chunking strategies
# ---------------------------------------------------------------------------

def fixed_chunking(pages: list[Document]) -> list[Document]:
    """Split pages into 512-token chunks with 50-token overlap."""
    splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=50)
    return splitter.split_documents(pages)


def semantic_chunking(pages: list[Document]) -> list[Document]:
    """Split pages at embedding-similarity breakpoints (SemanticChunker)."""
    splitter = SemanticChunker(
        get_embedder(), breakpoint_threshold_type="percentile"
    )
    return splitter.split_documents(pages)


def recursive_chunking(pages: list[Document]) -> list[Document]:
    """Split pages recursively on section → paragraph → sentence → word boundaries."""
    separators = ["\n\n## ", "\n\n", "\n", ". ", "! ", "? ", " ", ""]
    splitter = RecursiveCharacterTextSplitter(
        separators=separators,
        chunk_size=1000,
        chunk_overlap=100,
    )
    return splitter.split_documents(pages)


# ---------------------------------------------------------------------------
# FAISS index builder
# ---------------------------------------------------------------------------

def build_index(chunks: list[Document], condition_name: str):
    """Embed *chunks* and persist a FAISS index to data/index/{condition_name}/.

    Returns the FAISS vectorstore so callers can use it immediately.
    """
    from langchain_community.vectorstores import FAISS

    if not chunks:
        raise ValueError(f"No chunks provided for condition '{condition_name}'")

    index_path = _INDEX_BASE / condition_name
    index_path.mkdir(parents=True, exist_ok=True)

    vectorstore = FAISS.from_documents(chunks, get_embedder())
    vectorstore.save_local(str(index_path))
    print(f"[index] Saved {len(chunks)} chunks → {index_path}")
    return vectorstore


def load_index(condition_name: str):
    """Load an existing FAISS index from data/index/{condition_name}/."""
    from langchain_community.vectorstores import FAISS

    index_path = _INDEX_BASE / condition_name
    if not index_path.exists():
        raise FileNotFoundError(f"No index at {index_path}")
    return FAISS.load_local(
        str(index_path), get_embedder(), allow_dangerous_deserialization=True
    )
