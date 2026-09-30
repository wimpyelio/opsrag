"""Deterministic evaluation metrics for retrieval and generated answers."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def token_f1(prediction: str, reference: str) -> float:
    pred = Counter(_tokens(prediction))
    ref = Counter(_tokens(reference))
    if not pred or not ref:
        return float(not pred and not ref)
    overlap = sum((pred & ref).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(pred.values())
    recall = overlap / sum(ref.values())
    return 2 * precision * recall / (precision + recall)


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int = 5) -> float:
    if not relevant_ids:
        return 1.0
    retrieved = set(retrieved_ids[:k])
    return len(retrieved & set(relevant_ids)) / len(set(relevant_ids))


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    relevant = set(relevant_ids)
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def citation_coverage(answer: str, expected_labels: list[str], *, answerable: bool) -> float:
    if not answerable:
        lowered = answer.lower()
        refusal_markers = ("not in the provided", "not contain enough evidence", "insufficient evidence", "cannot answer")
        return 1.0 if any(marker in lowered for marker in refusal_markers) else 0.0
    if not expected_labels:
        return 0.0
    matched = sum(1 for label in expected_labels if label.lower() in answer.lower())
    return matched / len(expected_labels)


def failure_category(
    *,
    recall5: float | None,
    correctness: float,
    citation_cov: float,
    answerable: bool,
) -> str:
    failures: list[str] = []
    if answerable and recall5 is not None and recall5 == 0:
        failures.append("irrelevant retrieval")
    if answerable and recall5 is not None and recall5 < 1:
        failures.append("missing evidence")
    if correctness < 0.5:
        failures.append("unsupported answer")
    if answerable and correctness < 0.75:
        failures.append("incomplete answer")
    if answerable and citation_cov == 0:
        failures.append("wrong source")
    return "; ".join(dict.fromkeys(failures)) or "none"


def evaluate_item(item: dict[str, Any], results: list[dict[str, Any]], answer: str) -> dict[str, Any]:
    relevant_ids = list(item.get("relevant_chunks", []))
    retrieved_ids = [result.get("chunk_id", "") for result in results]
    answerable = bool(relevant_ids)
    recall5 = recall_at_k(retrieved_ids, relevant_ids, 5) if answerable else None
    mrr = reciprocal_rank(retrieved_ids, relevant_ids) if answerable else None
    correctness = token_f1(answer, item["reference_answer"])
    expected_labels = [
        f"[{result.get('title', '')} — {result.get('section', 'General')}]"
        for result in results
        if result.get("chunk_id") in relevant_ids
    ]
    citation_cov = citation_coverage(
        answer,
        expected_labels,
        answerable=answerable,
    )
    return {
        "id": item["id"],
        "question": item["question"],
        "answer": answer,
        "recall_at_5": "" if recall5 is None else round(recall5, 4),
        "mrr": "" if mrr is None else round(mrr, 4),
        "answer_correctness": round(correctness, 4),
        "citation_coverage": round(citation_cov, 4),
        "relevant_chunks": ",".join(relevant_ids),
        "retrieved_chunks": ",".join(retrieved_ids),
        "failure_category": failure_category(
            recall5=recall5,
            correctness=correctness,
            citation_cov=citation_cov,
            answerable=answerable,
        ),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    answerable = [row for row in rows if row["recall_at_5"] != ""]
    def mean_float(key: str, subset: list[dict[str, Any]]) -> float:
        values = [float(row[key]) for row in subset]
        return round(sum(values) / len(values), 4) if values else 0.0
    return {
        "questions": len(rows),
        "answerable_questions": len(answerable),
        "Recall@5": mean_float("recall_at_5", answerable),
        "MRR": mean_float("mrr", answerable),
        "Answer Correctness": mean_float("answer_correctness", rows),
        "Citation Coverage": mean_float("citation_coverage", rows),
    }
