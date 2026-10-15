from src.chunking import build_chunks, chunk_text


def test_chunk_text_is_overlapping_and_nonempty():
    text = " ".join(f"word{i}" for i in range(220))
    chunks = chunk_text(text, chunk_size=120, chunk_overlap=20)
    assert len(chunks) > 1
    assert all(chunk for chunk in chunks)
    assert chunks[0][-20:] in chunks[1]


def test_chunk_validation():
    try:
        chunk_text("abc", chunk_size=20, chunk_overlap=20)
    except ValueError as exc:
        assert "smaller" in str(exc)
    else:
        raise AssertionError("Expected chunk_overlap validation error")


def test_build_chunks_has_deterministic_ids():
    chunks = build_chunks(
        document_id="doc1",
        title="Doc",
        source="https://example.org",
        topic="ops",
        section="One",
        text="alpha beta gamma delta",
        chunk_size=50,
        chunk_overlap=5,
    )
    assert [chunk.chunk_id for chunk in chunks] == ["doc1-c0001"]
    assert chunks[0].document_id == "doc1"


# ── New tests added after stress-test findings ────────────────────────────────


def test_empty_text_returns_empty_list():
    assert chunk_text("") == []


def test_text_shorter_than_chunk_size_returns_single_chunk():
    text = "short text"
    chunks = chunk_text(text, chunk_size=500, chunk_overlap=80)
    assert chunks == [text]


def test_chunk_size_zero_raises_value_error():
    import pytest
    with pytest.raises(ValueError):
        chunk_text("hello", chunk_size=0)


def test_nul_bytes_are_replaced_before_chunking():
    # clean_text replaces NUL with a space, not an empty string.
    result = chunk_text("hello\x00world", chunk_size=500)
    assert "\x00" not in result[0]
    assert "hello" in result[0]
    assert "world" in result[0]


def test_excess_newlines_are_collapsed():
    result = chunk_text("a\n\n\n\n\nb", chunk_size=500)
    assert "\n\n\n" not in result[0]


def test_zero_overlap_produces_multiple_nonempty_chunks():
    # Use varied text so chunks are verifiably different.
    words = [f"word{i}" for i in range(100)]
    text = " ".join(words)
    chunks = chunk_text(text, chunk_size=50, chunk_overlap=0)
    assert len(chunks) > 1
    assert all(c.strip() for c in chunks), "All chunks must be non-empty"
    # The first chunk must not contain words from the last chunk (zero-overlap guarantee).
    first_words = set(chunks[0].split())
    last_words = set(chunks[-1].split())
    assert first_words.isdisjoint(last_words), (
        "Zero overlap: first and last chunks share words, suggesting overlap is non-zero"
    )


def test_chunk_ids_are_sequential_across_multiple_chunks():
    long_text = "word " * 200
    chunks = build_chunks(
        document_id="longdoc",
        title="Long",
        source="s",
        topic="t",
        section="s",
        text=long_text,
        chunk_size=100,
        chunk_overlap=10,
    )
    ids = [c.chunk_id for c in chunks]
    expected = [f"longdoc-c{i:04d}" for i in range(1, len(chunks) + 1)]
    assert ids == expected


def test_unicode_text_chunks_without_error():
    text = "héllo wörld tëst ünïcode chäracters " * 15
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 0
    assert all(isinstance(c, str) for c in chunks)
