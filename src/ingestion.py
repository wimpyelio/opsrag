"""Read PDF/text sources, extract metadata, embed, and bulk-index chunks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from pypdf import PdfReader

from .chunking import Chunk, build_chunks
from .config import settings
from .elasticsearch_client import get_client
from .embeddings import embed_documents


@dataclass(frozen=True)
class Document:
    document_id: str
    title: str
    source: str
    topic: str
    section: str
    text: str


def _parse_text_metadata(path: Path, raw_text: str) -> Document:
    """Read optional TITLE/SOURCE/TOPIC/SECTION headers from a text source."""
    metadata: dict[str, str] = {}
    remaining: list[str] = []
    header_done = False
    for line in raw_text.splitlines():
        match = re.match(r"^(TITLE|SOURCE|TOPIC|SECTION):\s*(.+)$", line.strip())
        if not header_done and match:
            metadata[match.group(1).lower()] = match.group(2).strip()
            continue
        header_done = True
        remaining.append(line)

    return Document(
        document_id=path.stem.lower().replace(" ", "_"),
        title=metadata.get("title", path.stem.replace("_", " ").title()),
        source=metadata.get("source", path.name),
        topic=metadata.get("topic", "operations"),
        section=metadata.get("section", "General"),
        text="\n".join(remaining).strip(),
    )


def extract_document(path: Path) -> Document:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        text = "\n\n".join((page.extract_text() or "") for page in reader.pages).strip()
        return Document(
            document_id=path.stem.lower().replace(" ", "_"),
            title=path.stem.replace("_", " ").title(),
            source=path.name,
            topic="operations",
            section="Extracted text",
            text=text,
        )
    if suffix in {".txt", ".md"}:
        return _parse_text_metadata(path, path.read_text(encoding="utf-8"))
    raise ValueError(f"Unsupported document type: {path.suffix}")


def build_document_chunks(doc: Document) -> list[Chunk]:
    return build_chunks(
        document_id=doc.document_id,
        title=doc.title,
        source=doc.source,
        topic=doc.topic,
        section=doc.section,
        text=doc.text,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )


def ingest_directory(data_dir: str | Path) -> dict[str, int]:
    """Extract, embed, and index every PDF/text file in a directory tree."""
    data_path = Path(data_dir)
    paths = sorted(
        path
        for path in data_path.rglob("*")
        if path.is_file() and path.suffix.lower() in {".pdf", ".txt", ".md"}
    )
    if not paths:
        raise FileNotFoundError(f"No .pdf, .txt, or .md documents found under {data_path}")

    client = get_client()
    try:
        all_chunks: list[Chunk] = []
        docs_indexed = 0
        for path in paths:
            doc = extract_document(path)
            chunks = build_document_chunks(doc)
            if not chunks:
                continue
            all_chunks.extend(chunks)
            docs_indexed += 1

        if not all_chunks:
            raise ValueError("No non-empty chunks were produced from the input documents")

        vectors = embed_documents([chunk.text for chunk in all_chunks])
        actions = [
            {
                "_index": settings.elasticsearch_index,
                "_id": chunk.chunk_id,
                "_source": {
                    "document_id": chunk.document_id,
                    "title": chunk.title,
                    "source": chunk.source,
                    "topic": chunk.topic,
                    "section": chunk.section,
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "embedding": vector,
                },
            }
            for chunk, vector in zip(all_chunks, vectors, strict=True)
        ]

        from elasticsearch.helpers import bulk

        success, failures = bulk(client, actions, refresh="wait_for")
        return {
            "documents": docs_indexed,
            "chunks": len(all_chunks),
            "indexed": int(success),
            "failures": len(failures),
        }
    finally:
        client.close()
