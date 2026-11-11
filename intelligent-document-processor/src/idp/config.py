import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
UPLOAD_DIR = ROOT / "data" / "uploads"

EMBEDDING_DIMS = {
    "text-embedding-3-large": 3072,
    "text-embedding-3-small": 1536,
}


@dataclass(frozen=True)
class Settings:
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")

    vector_backend: str = os.getenv("VECTOR_BACKEND", "qdrant").lower()
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "")
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
    pinecone_cloud: str = os.getenv("PINECONE_CLOUD", "aws")
    pinecone_region: str = os.getenv("PINECONE_REGION", "us-east-1")
    collection_prefix: str = os.getenv("COLLECTION_PREFIX", "idp")

    chunk_strategy: str = os.getenv("CHUNK_STRATEGY", "recursive")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    top_k: int = int(os.getenv("TOP_K", "4"))

    @property
    def embedding_dim(self) -> int:
        return EMBEDDING_DIMS.get(self.embedding_model, 3072)


settings = Settings()
