"""Ingest the sample document collection."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import project_path
from src.ingestion import ingest_directory


if __name__ == "__main__":
    summary = ingest_directory(project_path("data", "sample"))
    print(summary)
