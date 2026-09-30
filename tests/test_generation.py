from src.generation import _ensure_source_citations


def test_source_labels_are_added_when_missing():
    results = [{"title": "Doc", "section": "One"}]
    answer = _ensure_source_citations("An answer.", results)
    assert "Sources:" in answer
    assert "[Doc — One]" in answer


def test_missing_label_is_added_to_existing_sources_line():
    results = [
        {"title": "Doc A", "section": "One"},
        {"title": "Doc B", "section": "Two"},
    ]
    answer = _ensure_source_citations("Answer.\n\nSources: [Doc A — One]", results)
    assert "[Doc A — One]" in answer
    assert "[Doc B — Two]" in answer
