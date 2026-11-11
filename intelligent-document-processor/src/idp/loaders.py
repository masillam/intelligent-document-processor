"""Load PDF, DOCX, HTML and Markdown files into LangChain Documents."""
from pathlib import Path

from langchain_core.documents import Document

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".html", ".htm", ".md", ".markdown"}


def load_file(path: str | Path) -> list[Document]:
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}")

    if ext == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader

        docs = PyPDFLoader(str(path)).load()
    elif ext == ".docx":
        from langchain_community.document_loaders import Docx2txtLoader

        docs = Docx2txtLoader(str(path)).load()
    elif ext in {".html", ".htm"}:
        from langchain_community.document_loaders import BSHTMLLoader

        docs = BSHTMLLoader(str(path), open_encoding="utf-8").load()
    else:
        from langchain_community.document_loaders import TextLoader

        docs = TextLoader(str(path), encoding="utf-8").load()

    for d in docs:
        d.metadata["source"] = path.name
        d.metadata["file_type"] = ext.lstrip(".")
    return [d for d in docs if d.page_content.strip()]
