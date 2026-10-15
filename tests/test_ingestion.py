from pathlib import Path

import pytest

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


# ── New tests added after stress-test findings ────────────────────────────────


def test_blank_line_between_metadata_headers_is_tolerated(tmp_path: Path):
    """BUG-3 fix: blank lines inside the header block must not terminate parsing early."""
    path = tmp_path / "spaced_headers.txt"
    path.write_text(
        "TITLE: My Doc\n\nSOURCE: https://example.org/doc\nBody content.",
        encoding="utf-8",
    )

    document = extract_document(path)

    assert document.title == "My Doc"
    assert document.source == "https://example.org/doc", (
        "SOURCE was not parsed; blank line between headers incorrectly ended header section"
    )
    assert "Body content" in document.text


def test_multiple_blank_lines_between_all_header_fields(tmp_path: Path):
    """All four fields survive arbitrary blank lines between them."""
    path = tmp_path / "gapped.txt"
    path.write_text(
        "TITLE: Gapped\n\n\nSOURCE: https://src.org\n\nTOPIC: planning\n\nSECTION: Alpha\nBody.",
        encoding="utf-8",
    )

    doc = extract_document(path)

    assert doc.title == "Gapped"
    assert doc.source == "https://src.org"
    assert doc.topic == "planning"
    assert doc.section == "Alpha"
    assert doc.text == "Body."


def test_partial_metadata_falls_back_for_missing_fields(tmp_path: Path):
    """Only TITLE declared; SOURCE, TOPIC, SECTION use defaults."""
    path = tmp_path / "partial.txt"
    path.write_text("TITLE: Partial Doc\nSome body text.", encoding="utf-8")

    doc = extract_document(path)

    assert doc.title == "Partial Doc"
    assert doc.source == "partial.txt"       # filename fallback
    assert doc.topic == "operations"          # module default
    assert doc.section == "General"           # module default
    assert "body text" in doc.text.lower()


def test_empty_file_produces_document_with_empty_text(tmp_path: Path):
    path = tmp_path / "empty.txt"
    path.write_text("", encoding="utf-8")

    doc = extract_document(path)

    assert doc.text == ""
    assert doc.document_id == "empty"


def test_unsupported_file_extension_raises_value_error(tmp_path: Path):
    path = tmp_path / "data.csv"
    path.write_text("col1,col2\na,b", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported document type"):
        extract_document(path)


def test_markdown_file_uses_same_metadata_parsing_as_txt(tmp_path: Path):
    path = tmp_path / "doc.md"
    path.write_text(
        "TITLE: MD Doc\nSOURCE: https://md.org\nMarkdown body.", encoding="utf-8"
    )

    doc = extract_document(path)

    assert doc.title == "MD Doc"
    assert doc.source == "https://md.org"
    assert "Markdown body" in doc.text


def test_document_id_normalised_from_filename(tmp_path: Path):
    path = tmp_path / "My Document File.txt"
    path.write_text("Body.", encoding="utf-8")

    doc = extract_document(path)

    assert doc.document_id == "my_document_file"


def test_non_header_line_terminates_header_section(tmp_path: Path):
    """A genuine body line before SECTION must push SECTION into body text."""
    path = tmp_path / "body_first.txt"
    path.write_text(
        "TITLE: T\nThis is body text.\nSECTION: Should Be Body\nMore body.",
        encoding="utf-8",
    )

    doc = extract_document(path)

    assert doc.title == "T"
    assert doc.section == "General"   # SECTION came after body line → not parsed
    assert "SECTION: Should Be Body" in doc.text
