# Experiment log

## Baseline configuration

- Retrieval: Elasticsearch BM25.
- Prompt: `prompts/baseline.txt`.
- Evaluation set: `data/evaluation/questions.json`.

## Improved configuration

- Retrieval: Elasticsearch hybrid retrieval using the same lexical query and a kNN retriever fused with native RRF.
- Prompt: `prompts/improved.txt`.
- Evaluation set: the same JSON file and question order as baseline.

## Development notes

The project was developed against native Elasticsearch on Windows rather than a containerized search service. Connection debugging included TLS/transport failures; the final repository keeps certificate handling in environment configuration instead of committing a machine-specific certificate path.

The development corpus was ingested successfully once the indexing path was working: 10 documents produced 10 chunks, all 10 chunks were indexed, and Elasticsearch count returned 10.

Retrieval probes were run during debugging. BM25 returned the NECP chunk for `emergency communications`. Vector retrieval returned the same chunk within its top results but ranked a plain-language communications chunk first. A later hybrid probe also returned the NECP chunk. An earlier hybrid request errored during debugging, so this log records the successful probe without turning it into a blanket claim of production readiness.

A local Ollama generation check returned HTTP 200 with `qwen3:4b` configured. The machine also had `qwen3:8b` available during later model inspection. The repository therefore treats the Ollama model as a local configuration choice.

No evaluation CSV values from those historical runs were retained in the source archive. This repository does not invent replacements.
