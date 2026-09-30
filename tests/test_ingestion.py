from pathlib import Path

from src.ingestion import extract_document


def test_text_metadata_uses_declared_source(tmp_path: Path):
    path = tmp_path / "sample_note.txt"
    path.write_text(
        "TITLE: Sample\nSOURCE: https://example.org/source\nTOPIC: ops\nSECTION: One\nBody text",
        encoding="utf-8",
    )

    document = extract_document(path)

    assert document.title == "Sample"
    assert document.source == "https://example.org/source"
    assert document.topic == "ops"
    assert document.section == "One"
    assert document.text == "Body text"


def test_text_metadata_never_uses_absolute_local_path_as_source(tmp_path: Path):
    path = tmp_path / "local_only.txt"
    path.write_text("Body text", encoding="utf-8")

    document = extract_document(path)

    assert document.source == path.name
    assert str(tmp_path) not in document.source
