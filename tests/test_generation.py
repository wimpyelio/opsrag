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


# ── New tests added after stress-test findings ────────────────────────────────


def test_label_already_in_prose_body_is_not_duplicated():
    """BUG-1 fix: if a label appears in prose, no redundant Sources block is added."""
    results = [{"title": "Doc A", "section": "Sec 1"}]
    answer = "The answer references [Doc A — Sec 1] inline."

    result = _ensure_source_citations(answer, results)

    assert result.count("[Doc A — Sec 1]") == 1, (
        "Label already in prose was duplicated in a Sources block"
    )


def test_label_check_is_case_insensitive():
    """Citation check uses same lower() comparison as citation_coverage metric."""
    results = [{"title": "Doc A", "section": "Sec 1"}]
    answer = "Answer body [DOC A — SEC 1] already present."

    result = _ensure_source_citations(answer, results)

    # No Sources: block should be appended — label is already present.
    assert result == answer


def test_all_labels_present_returns_answer_unchanged():
    results = [
        {"title": "Alpha", "section": "One"},
        {"title": "Beta", "section": "Two"},
    ]
    answer = "Text.\n\nSources: [Alpha — One] [Beta — Two]"

    result = _ensure_source_citations(answer, results)

    assert result == answer


def test_empty_results_returns_answer_unchanged():
    answer = "Some answer with no retrievals."
    result = _ensure_source_citations(answer, [])
    assert result == answer


def test_empty_answer_string_gets_sources_appended():
    results = [{"title": "Doc A", "section": "Sec 1"}]
    result = _ensure_source_citations("", results)
    assert "[Doc A — Sec 1]" in result
    assert "Sources:" in result


def test_section_key_missing_defaults_to_general():
    """Result dict without 'section' key should not raise KeyError."""
    results = [{"title": "Doc A"}]
    answer = _ensure_source_citations("An answer.", results)
    assert "[Doc A — General]" in answer
