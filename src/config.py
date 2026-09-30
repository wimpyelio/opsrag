"""Environment-backed project configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


@dataclass(frozen=True)
class Settings:
    elasticsearch_url: str = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
    elasticsearch_api_key: str | None = os.getenv("ELASTICSEARCH_API_KEY") or None
    elasticsearch_username: str | None = os.getenv("ELASTICSEARCH_USERNAME") or None
    elasticsearch_password: str | None = os.getenv("ELASTICSEARCH_PASSWORD") or None
    elasticsearch_ca_cert: str | None = os.getenv("ELASTICSEARCH_CA_CERT") or None
    elasticsearch_verify_certs: bool = _env_bool("ELASTICSEARCH_VERIFY_CERTS", True)
    elasticsearch_index: str = os.getenv("ELASTICSEARCH_INDEX", "opsrag_chunks")
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen3:8b")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "80"))
    top_k: int = int(os.getenv("TOP_K", "5"))
    vector_num_candidates: int = int(os.getenv("VECTOR_NUM_CANDIDATES", "25"))


settings = Settings()


def project_path(*parts: str) -> Path:
    """Return a path inside the repository root."""
    return ROOT_DIR.joinpath(*parts)
