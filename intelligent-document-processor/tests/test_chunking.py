import pytest
from langchain_core.documents import Document

from idp.chunking import chunk_documents


def test_recursive_respects_size():
    doc = Document(page_content="word " * 1000, metadata={"source": "x.md"})
    chunks = chunk_documents([doc], "recursive", chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 1
    assert all(len(c.page_content) <= 200 for c in chunks)
    assert chunks[0].metadata["source"] == "x.md"


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        chunk_documents([Document(page_content="a")], "recursive", chunk_size=100, chunk_overlap=100)


def test_unknown_strategy():
    with pytest.raises(ValueError):
        chunk_documents([Document(page_content="a")], "nope")
