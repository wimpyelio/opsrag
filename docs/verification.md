# Verification record

This document separates development evidence from repeatable repository checks.

## Observed during project development

| Check | Observed result | Interpretation |
|---|---|---|
| Elasticsearch server | 9.5.4 | Native local Elasticsearch node was reachable during development. |
| Elasticsearch Python client | 9.5.1 | The project client version matched the pinned requirement. |
| Sample ingestion | 10 documents / 10 chunks / 10 indexed / 0 failures | The sample indexing path completed successfully. |
| Elasticsearch count | 10 | The index contained the expected number of stored chunks immediately after the ingestion run. |
| BM25 probe: `emergency communications` | `emergency_communications-c0001` returned | Lexical retrieval path was exercised against the indexed corpus. |
| Vector probe: `emergency communications` | `plain_language_comms` ranked first; NECP chunk also returned | Dense retrieval produced a different ordering from BM25 on this small corpus. |
| Hybrid probe | NECP chunk returned in a later successful probe | Native-RRF path was exercised, but an earlier hybrid call failed during debugging. |
| Ollama generation | HTTP 200 with `qwen3:4b` configured | Local chat generation path returned content. |

## Repeatable repository checks

From a clean checkout with Python dependencies installed:

```bash
python -m compileall -q src scripts app
pytest -q
```

These checks do not prove live Elasticsearch or Ollama compatibility. They verify syntax and the deterministic/unit-test surface.

## Deliberately excluded evidence

The public repository does not include:

- Windows event logs
- Elasticsearch logs
- Java process dumps
- memory or crash diagnostics
- local installation paths
- local certificate files
- local model blobs
- generated evaluation CSVs
- `.env` contents

Those artifacts were useful during troubleshooting but do not belong in a reproducible source repository.
