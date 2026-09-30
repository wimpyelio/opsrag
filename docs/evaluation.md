# Evaluation design

The evaluation set contains **35 questions**. There are **34 answerable questions** and **1 deliberately unanswerable question** (`q035`), with **37 total relevant-chunk references** across the answerable items.

## Experimental comparison

The evaluation script runs two definitions over the same question order:

| Experiment | Retrieval | Prompt |
|---|---|---|
| `baseline` | BM25 | `prompts/baseline.txt` |
| `improved` | Hybrid RRF | `prompts/improved.txt` |

This is a controlled retrieval/prompt comparison inside the same corpus and generation pipeline. It is not a claim that the improved configuration is universally better.

## Retrieval metrics

**Recall@5** is the fraction of unique expected relevant chunk IDs found in the first five retrieved chunks.

**MRR** is the reciprocal rank of the first relevant chunk.

For the unanswerable item, Recall@5 and MRR are omitted because there is no relevant chunk to recall.

## Answer metrics

**Answer Correctness** is token-level F1 between the generated answer and the stored reference answer. It is deterministic and does not use a second model as an evaluator.

**Citation Coverage** checks whether each source label attached to a retrieved relevant chunk appears verbatim in the answer. This measures citation exposure for evidence that was actually retrieved; it does not prove that the citation semantically supports the claim. For the unanswerable question, the metric is satisfied only when the response contains a clear insufficient-evidence refusal marker.

## Failure categories

The evaluator records diagnostic labels from the measured row:

- `irrelevant retrieval` when an answerable question has Recall@5 = 0
- `missing evidence` when an answerable question has Recall@5 < 1
- `unsupported answer` when token F1 < 0.5
- `incomplete answer` when an answerable question has token F1 < 0.75
- `wrong source` when an answerable question has zero citation coverage

A row can have multiple labels. These thresholds are project diagnostics, not externally validated quality thresholds.

## Generated outputs

A live run writes:

```text
results/baseline.csv
results/improved.csv
results/summary.csv
```

Those files are ignored by Git because they contain run-specific generated answers and measured values. The repository intentionally ships the evaluation definition, not a fabricated benchmark snapshot.
