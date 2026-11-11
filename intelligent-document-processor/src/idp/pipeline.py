"""Ingestion: load -> extract insights -> chunk -> embed -> store."""
import hashlib
import json
import time
import uuid
from pathlib import Path

from .chunking import chunk_documents
from .config import PROCESSED_DIR, settings
from .extraction import extract_insights
from .loaders import load_file
from .vectorstore import get_store


def _file_id(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()[:16]


def ingest_file(
    path: str | Path,
    strategy: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    collection: str | None = None,
    with_insights: bool = True,
) -> dict:
    path = Path(path)
    strategy = strategy or settings.chunk_strategy
    started = time.time()

    docs = load_file(path)
    doc_id = _file_id(path)

    insights = None
    if with_insights:
        full_text = "\n\n".join(d.page_content for d in docs)
        insights = extract_insights(full_text).model_dump()
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        record = {"doc_id": doc_id, "filename": path.name, "pages": len(docs), **insights}
        (PROCESSED_DIR / f"{doc_id}.json").write_text(json.dumps(record, indent=2))

    chunks = chunk_documents(docs, strategy, chunk_size, chunk_overlap)
    for i, c in enumerate(chunks):
        c.metadata.update({"doc_id": doc_id, "chunk_index": i, "strategy": strategy})

    # Deterministic IDs make re-ingesting the same file idempotent.
    ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}:{strategy}:{i}")) for i in range(len(chunks))]
    get_store(strategy, collection).add_documents(chunks, ids=ids)

    lengths = [len(c.page_content) for c in chunks]
    return {
        "doc_id": doc_id,
        "filename": path.name,
        "strategy": strategy,
        "chunks": len(chunks),
        "avg_chunk_chars": round(sum(lengths) / max(len(lengths), 1), 1),
        "seconds": round(time.time() - started, 2),
        "insights": insights,
    }
