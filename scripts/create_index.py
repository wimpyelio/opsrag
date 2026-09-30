"""Create or recreate the OpsRAG Elasticsearch index."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import settings
from src.embeddings import embedding_dimension
from src.elasticsearch_client import get_client


def main() -> None:
    client = get_client()
    try:
        if client.indices.exists(index=settings.elasticsearch_index):
            client.indices.delete(index=settings.elasticsearch_index)

        dimensions = embedding_dimension()
        mapping = {
            "settings": {"number_of_shards": 1},
            "mappings": {
                "properties": {
                    "document_id": {"type": "keyword"},
                    "title": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                    "source": {"type": "keyword", "ignore_above": 2048},
                    "topic": {"type": "keyword"},
                    "section": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                    "chunk_id": {"type": "keyword"},
                    "text": {"type": "text"},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": dimensions,
                        "index": True,
                        "similarity": "cosine",
                    },
                }
            },
        }
        client.indices.create(index=settings.elasticsearch_index, **mapping)
        print(f"Created index {settings.elasticsearch_index} with embedding dims={dimensions}.")
    finally:
        if hasattr(client, "close"):
            client.close()


if __name__ == "__main__":
    main()
