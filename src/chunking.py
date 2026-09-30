"""Small deterministic character-based chunker."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    document_id: str
    title: str
    source: str
    topic: str
    section: str
    chunk_id: str
    text: str


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(
    text: str,
    *,
    chunk_size: int = 500,
    chunk_overlap: int = 80,
) -> list[str]:
    """Split text into overlapping character windows, preferring word boundaries."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")

    text = clean_text(text)
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)
        if end < length:
            boundary = text.rfind(" ", start + max(1, chunk_size // 2), end)
            if boundary > start:
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= length:
            break
        next_start = max(end - chunk_overlap, start + 1)
        while next_start < length and text[next_start].isspace():
            next_start += 1
        start = next_start

    return chunks


def build_chunks(
    *,
    document_id: str,
    title: str,
    source: str,
    topic: str,
    section: str,
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 80,
) -> list[Chunk]:
    pieces = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return [
        Chunk(
            document_id=document_id,
            title=title,
            source=source,
            topic=topic,
            section=section,
            chunk_id=f"{document_id}-c{idx:04d}",
            text=piece,
        )
        for idx, piece in enumerate(pieces, start=1)
    ]
