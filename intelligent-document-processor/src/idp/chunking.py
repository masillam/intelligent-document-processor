"""Configurable chunking: recursive (size/overlap) and semantic strategies."""
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import settings

STRATEGIES = ("recursive", "semantic")


def chunk_documents(
    docs: list[Document],
    strategy: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    embeddings=None,
) -> list[Document]:
    strategy = strategy or settings.chunk_strategy
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = settings.chunk_overlap if chunk_overlap is None else chunk_overlap

    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from {STRATEGIES}")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    recursive = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
    )
    if strategy == "recursive":
        return recursive.split_documents(docs)

    # Semantic: split where embedding similarity between sentences drops.
    from langchain_experimental.text_splitter import SemanticChunker

    if embeddings is None:
        from .vectorstore import get_embeddings

        embeddings = get_embeddings()
    semantic = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")
    chunks = semantic.split_documents(docs)
    # Cap very large semantic chunks so they stay within a sensible context size.
    return recursive.split_documents(chunks)
