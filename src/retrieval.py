"""BM25, vector, and Elasticsearch-native RRF retrieval."""

from __future__ import annotations

from typing import Any

from .config import settings
from .elasticsearch_client import get_client
from .embeddings import embed_query


RETURN_FIELDS = [
    "document_id",
    "title",
    "source",
    "topic",
    "section",
    "chunk_id",
    "text",
]


def _format_hits(response: Any) -> list[dict[str, Any]]:
    hits = response.get("hits", {}).get("hits", [])
    results: list[dict[str, Any]] = []
    for hit in hits:
        source = dict(hit.get("_source", {}))
        results.append({
            **source,
            "score": hit.get("_score"),
        })
    return results


def search_bm25(query: str, top_k: int = 5, client: Any | None = None) -> list[dict[str, Any]]:
    owns_client = client is None
    client = client or get_client()
    try:
        response = client.search(
            index=settings.elasticsearch_index,
            size=top_k,
            query={
                "multi_match": {
                    "query": query,
                    "fields": ["text^3", "title^2", "topic", "section"],
                    "operator": "or",
                }
            },
        )
        return _format_hits(response)
    finally:
        if owns_client and hasattr(client, "close"):
            client.close()


def search_vector(query: str, top_k: int = 5, client: Any | None = None) -> list[dict[str, Any]]:
    owns_client = client is None
    client = client or get_client()
    try:
        query_vector = embed_query(query)
        response = client.search(
            index=settings.elasticsearch_index,
            size=top_k,
            knn={
                "field": "embedding",
                "query_vector": query_vector,
                "k": top_k,
                "num_candidates": max(settings.vector_num_candidates, top_k),
            },
        )
        return _format_hits(response)
    finally:
        if owns_client and hasattr(client, "close"):
            client.close()


def search_hybrid(query: str, top_k: int = 5, client: Any | None = None) -> list[dict[str, Any]]:
    owns_client = client is None
    client = client or get_client()
    try:
        query_vector = embed_query(query)
        response = client.search(
            index=settings.elasticsearch_index,
            size=top_k,
            retriever={
                "rrf": {
                    "retrievers": [
                        {
                            "standard": {
                                "query": {
                                    "multi_match": {
                                        "query": query,
                                        "fields": ["text^3", "title^2", "topic", "section"],
                                        "operator": "or",
                                    }
                                }
                            }
                        },
                        {
                            "knn": {
                                "field": "embedding",
                                "query_vector": query_vector,
                                "k": top_k,
                                "num_candidates": max(settings.vector_num_candidates, top_k),
                            }
                        },
                    ],
                    "rank_constant": 60,
                    "rank_window_size": max(top_k * 3, 10),
                }
            },
        )
        return _format_hits(response)
    finally:
        if owns_client and hasattr(client, "close"):
            client.close()


_METHOD_MAP: dict[str, Any] = {}  # populated below after function definitions


def retrieve(query: str, method: str, top_k: int = 5, client: Any | None = None) -> list[dict[str, Any]]:
    """Dispatch to the appropriate search function.

    ``method`` is matched case-insensitively so callers may pass
    ``"bm25"``, ``"BM25"``, or ``"Bm25"`` interchangeably.
    Valid values: BM25 · Vector · Hybrid.
    """
    fn = _METHOD_MAP.get(method.strip().lower())
    if fn is None:
        raise ValueError(
            f"Unknown retrieval method: {method!r}. Valid values: BM25, Vector, Hybrid."
        )
    return fn(query, top_k, client=client)


_METHOD_MAP.update(
    {
        "bm25": search_bm25,
        "vector": search_vector,
        "hybrid": search_hybrid,
    }
)
