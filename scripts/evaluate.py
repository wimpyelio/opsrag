"""Run baseline and improved RAG evaluations and write CSV results."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import project_path
from src.elasticsearch_client import get_client
from src.evaluation import evaluate_item, summarize
from src.generation import answer_question

RESULT_FIELDS = [
    "id",
    "question",
    "answer",
    "recall_at_5",
    "mrr",
    "answer_correctness",
    "citation_coverage",
    "relevant_chunks",
    "retrieved_chunks",
    "failure_category",
]


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N questions.")
    args = parser.parse_args()

    questions_path = project_path("data", "evaluation", "questions.json")
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    if args.limit is not None:
        questions = questions[: args.limit]

    experiments = {
        "baseline": ("BM25", "baseline"),
        "improved": ("Hybrid", "improved"),
    }
    summaries: list[dict] = []
    for label, (method, prompt_name) in experiments.items():
        print(f"[{label}] method={method}  prompt={prompt_name}  questions={len(questions)}")
        client = get_client()
        try:
            rows: list[dict] = []
            for idx, item in enumerate(questions, start=1):
                result = answer_question(
                    item["question"],
                    method,
                    prompt_name=prompt_name,
                    es_client=client,
                )
                row = evaluate_item(item, result["retrieved_chunks"], result["answer"])
                rows.append(row)
                print(
                    f"  {idx:>3}/{len(questions)}  {item['id']}  "
                    f"recall={row['recall_at_5'] or 'n/a':>6}  "
                    f"f1={row['answer_correctness']:.3f}  "
                    f"cite={row['citation_coverage']:.2f}  "
                    f"[{row['failure_category']}]"
                )
        finally:
            if hasattr(client, "close"):
                client.close()
        write_rows(project_path("results", f"{label}.csv"), rows)
        metric_summary = summarize(rows)
        metric_summary["experiment"] = label
        summaries.append(metric_summary)

    summary_path = project_path("results", "summary.csv")
    summary_fields = ["experiment", "questions", "answerable_questions", "Recall@5", "MRR", "Answer Correctness", "Citation Coverage"]
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summaries)

    # Print summary table to stdout so results are visible without opening CSV files.
    col_w = 22
    metric_keys = ["Recall@5", "MRR", "Answer Correctness", "Citation Coverage"]
    header = f"{'Experiment':<14}" + "".join(f"{k:>{col_w}}" for k in metric_keys)
    print()
    print(header)
    print("-" * len(header))
    for row in summaries:
        line = f"{row['experiment']:<14}" + "".join(
            f"{row[k]:>{col_w}.4f}" for k in metric_keys
        )
        print(line)
    print()
    print(f"Full per-question results: results/baseline.csv · results/improved.csv")


if __name__ == "__main__":
    main()
