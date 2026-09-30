# Architecture

OpsRAG keeps each RAG stage explicit so that the retrieval path can be inspected without a framework abstraction layer.

```text
Document files
    │
    ├── PDF → pypdf text extraction
    └── TXT/MD → metadata header + text
    │
    ▼
Document
    │
    ▼
Chunking
    │  default: 500 characters / 80 overlap
    ▼
Sentence Transformers
    │  normalized dense vectors
    ▼
Elasticsearch
    │  text + metadata + dense_vector
    ├───────────────┐
    ▼               ▼
  BM25            kNN vector
    │               │
    └───────┬───────┘
            ▼
       native RRF
            │
            ▼
      top-k chunks
            │
            ▼
  bounded prompt context
            │
            ▼
       Ollama chat
            │
            ▼
 answer + exact labels
            │
            ▼
  deterministic metrics
```

## Retrieval contract

`src/retrieval.py` exposes three methods:

- `search_bm25()` uses an Elasticsearch `multi_match` query over `text^3`, `title^2`, `topic`, and `section`.
- `search_vector()` embeds the query and sends a kNN request against the `embedding` field.
- `search_hybrid()` combines the same lexical query and kNN retriever through Elasticsearch's native RRF retriever.

All three paths return the same source metadata fields, which allows the generation and evaluation layers to remain retrieval-method agnostic.

## Traceability contract

Every indexed chunk carries:

```text
document_id
title
source
topic
section
chunk_id
text
embedding
```

The prompt builder turns `title + section` into an exact citation label. The generation layer then checks that source labels are exposed in the final response.

## Runtime boundaries

Elasticsearch is the only persistent retrieval store. Sentence Transformers is used for local embedding generation. Ollama is used for local answer generation. No hosted LLM API key is required by the application path.
