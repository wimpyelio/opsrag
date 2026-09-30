"""Prompt loading and context formatting."""

from __future__ import annotations

from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts"


def load_prompt(name: str) -> str:
    path = PROMPT_DIR / f"{name}.txt"
    return path.read_text(encoding="utf-8")


def build_context(results: list[dict]) -> str:
    blocks = []
    for index, result in enumerate(results, start=1):
        citation = f"[{result['title']} — {result.get('section', 'General')}]"
        blocks.append(
            f"Chunk {index}\n"
            f"Citation: {citation}\n"
            f"Chunk ID: {result.get('chunk_id', '')}\n"
            f"Text: {result['text']}"
        )
    return "\n\n".join(blocks)


def render_prompt(prompt_name: str, question: str, results: list[dict]) -> str:
    template = load_prompt(prompt_name)
    context = build_context(results)
    return template.replace("{{QUESTION}}", question).replace("{{CONTEXT}}", context)
