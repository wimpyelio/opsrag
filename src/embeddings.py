"""Sentence Transformers embedding helpers."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from .config import settings


@lru_cache(maxsize=2)
def get_model(model_name: str | None = None):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name or settings.embedding_model)


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed document/passages with the model's document encoder when available."""
    if not texts:
        return []
    model = get_model()
    encoder = getattr(model, "encode_document", None) or model.encode
    vectors = encoder(
        texts,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return np.asarray(vectors, dtype="float32").tolist()


def embed_query(query: str) -> list[float]:
    model = get_model()
    encoder = getattr(model, "encode_query", None) or model.encode
    vector = encoder(
        query,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return np.asarray(vector, dtype="float32").tolist()


def embedding_dimension() -> int:
    model = get_model()
    dimension = model.get_sentence_embedding_dimension()
    if dimension is None:
        raise RuntimeError("Embedding model did not report an output dimension")
    return int(dimension)
