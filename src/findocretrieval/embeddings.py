"""Shared HuggingFace embedding utility — no API key required."""
from __future__ import annotations

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

# all-MiniLM-L6-v2 is small (80 MB), fast, and strong for English financial text
_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_embedder() -> HuggingFaceEmbeddings:
    """Return a cached HuggingFaceEmbeddings instance.

    The model is downloaded on first call and then kept in memory.
    Vectors are L2-normalised so cosine similarity == dot product.
    """
    return HuggingFaceEmbeddings(
        model_name=_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
