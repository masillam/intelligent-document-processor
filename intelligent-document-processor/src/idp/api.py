"""FastAPI service: upload documents and ask questions.

Run: uvicorn idp.api:app --reload
"""
import json
import shutil

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .chunking import STRATEGIES
from .config import PROCESSED_DIR, UPLOAD_DIR, settings
from .loaders import SUPPORTED_EXTENSIONS
from .pipeline import ingest_file
from .qa import answer_question

app = FastAPI(title="Intelligent Document Processor", version="0.1.0")


class AskRequest(BaseModel):
    question: str = Field(min_length=3)
    strategy: str = settings.chunk_strategy
    top_k: int = Field(default=settings.top_k, ge=1, le=20)


@app.get("/health")
def health():
    return {"status": "ok", "backend": settings.vector_backend, "embedding_model": settings.embedding_model}


@app.post("/ingest")
def ingest(
    file: UploadFile = File(...),
    strategy: str = Form(settings.chunk_strategy),
    chunk_size: int = Form(settings.chunk_size),
    chunk_overlap: int = Form(settings.chunk_overlap),
):
    from pathlib import Path

    name = Path(file.filename or "upload").name
    ext = Path(name).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type '{ext}'")
    if strategy not in STRATEGIES:
        raise HTTPException(400, f"strategy must be one of {STRATEGIES}")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = UPLOAD_DIR / name
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        return ingest_file(dest, strategy, chunk_size, chunk_overlap)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/ask")
def ask(req: AskRequest):
    if req.strategy not in STRATEGIES:
        raise HTTPException(400, f"strategy must be one of {STRATEGIES}")
    result = answer_question(req.question, req.strategy, req.top_k)
    result.pop("contexts")  # keep API response light; sources are returned
    return result


@app.get("/documents")
def documents():
    """Structured insights extracted for every ingested document."""
    if not PROCESSED_DIR.exists():
        return []
    return [json.loads(p.read_text()) for p in sorted(PROCESSED_DIR.glob("*.json"))]
