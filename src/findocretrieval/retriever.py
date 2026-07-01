"""Four retriever factory functions for the ablation study."""
from __future__ import annotations

from pathlib import Path
from typing import Union

from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker

from .embeddings import get_embedder

_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
_INDEX_BASE = Path(__file__).resolve().parents[3] / "data" / "index"


def _load_vectorstore(index_path: Union[str, Path]) -> FAISS:
    return FAISS.load_local(
        str(index_path), get_embedder(), allow_dangerous_deserialization=True
    )


def get_base_retriever(index_path: Union[str, Path], k: int = 20):
    """FAISS cosine-similarity retriever."""
    vs = _load_vectorstore(index_path)
    return vs.as_retriever(search_type="similarity", search_kwargs={"k": k})


def get_reranking_retriever(
    index_path: Union[str, Path], k: int = 20, top_n: int = 5
):
    """FAISS retriever with BAAI/bge-reranker-v2-m3 CrossEncoder reranking."""
    base = get_base_retriever(index_path, k=k)
    model = HuggingFaceCrossEncoder(model_name=_RERANKER_MODEL)
    compressor = CrossEncoderReranker(model=model, top_n=top_n)
    return ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base
    )


def get_metadata_retriever(
    index_path: Union[str, Path],
    company: str,
    doc_period: str,
    k: int = 20,
):
    """FAISS retriever with post-retrieval metadata filter on company + doc_period.

    Used for C2 (per-question retriever created inside run_ablation).
    """
    vs = _load_vectorstore(index_path)
    return vs.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": k,
            "filter": {"company": company, "doc_period": str(doc_period)},
        },
    )


def get_hybrid_retriever(
    chunks: list[Document],
    index_path: Union[str, Path],
    k: int = 20,
    top_n: int = 5,
):
    """EnsembleRetriever (BM25 + FAISS dense) then CrossEncoder reranking."""
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = k

    vs = _load_vectorstore(index_path)
    dense = vs.as_retriever(search_kwargs={"k": k})

    ensemble = EnsembleRetriever(retrievers=[bm25, dense], weights=[0.5, 0.5])
    model = HuggingFaceCrossEncoder(model_name=_RERANKER_MODEL)
    compressor = CrossEncoderReranker(model=model, top_n=top_n)
    return ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=ensemble
    )
