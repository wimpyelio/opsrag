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
