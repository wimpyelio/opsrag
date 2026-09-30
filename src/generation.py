"""Ollama generation wrapper and end-to-end RAG function."""

from __future__ import annotations

from typing import Any

from .config import settings
from .prompting import render_prompt
from .retrieval import retrieve


def get_ollama_client():
    from ollama import Client

    return Client(host=settings.ollama_host)


def generate_answer(prompt: str, *, client: Any | None = None) -> str:
    client = client or get_ollama_client()
    try:
        response = client.chat(
            model=settings.ollama_model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0},
        )
    except Exception as exc:
        raise RuntimeError(
            "Ollama generation failed. Confirm Ollama is running and the configured "
            f"model exists ({settings.ollama_model}). Error type: {type(exc).__name__}."
        ) from exc
    message = getattr(response, "message", None)
    content = getattr(message, "content", None) if message else None
    if content is None and isinstance(response, dict):
        content = response.get("message", {}).get("content")
    if not content:
        raise RuntimeError("Ollama returned an empty answer")
    return str(content).strip()


def _source_labels(results: list[dict[str, Any]]) -> list[str]:
    return [
        f"[{item['title']} — {item.get('section', 'General')}]"
        for item in results
    ]


def _ensure_source_citations(answer: str, results: list[dict[str, Any]]) -> str:
    """Ensure exact source labels are exposed in the final answer."""
    if not results:
        return answer

    labels = _source_labels(results)
    missing = [label for label in labels if label not in answer]
    if not missing and "Sources:" in answer:
        return answer

    body = answer.rstrip()
    if "Sources:" in body:
        return body + " " + " ".join(missing)
    return body + "\n\nSources: " + " ".join(labels)


def answer_question(
    question: str,
    method: str,
    *,
    prompt_name: str = "improved",
    top_k: int | None = None,
    es_client: Any | None = None,
    ollama_client: Any | None = None,
) -> dict[str, Any]:
    owns_es_client = es_client is None
    if es_client is None:
        from .elasticsearch_client import get_client

        es_client = get_client()
    try:
        results = retrieve(question, method, top_k or settings.top_k, client=es_client)
        prompt = render_prompt(prompt_name, question, results)
        answer = _ensure_source_citations(generate_answer(prompt, client=ollama_client), results)
        return {
            "question": question,
            "method": method,
            "answer": answer,
            "sources": [
                {
                    "title": item["title"],
                    "section": item.get("section", "General"),
                    "chunk_id": item.get("chunk_id", ""),
                    "source": item.get("source", ""),
                }
                for item in results
            ],
            "retrieved_chunks": results,
        }
    finally:
        if owns_es_client and hasattr(es_client, "close"):
            es_client.close()
