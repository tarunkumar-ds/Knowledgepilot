from src.rag import split_text


def test_split_text_empty():
    assert split_text("") == []


def test_split_text_short_text():
    assert split_text("hello world") == ["hello world"]


def test_split_text_chunks_long_text():
    text = "word " * 600
    chunks = split_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    assert all(chunks)
    assert all(len(chunk) <= 100 for chunk in chunks)
