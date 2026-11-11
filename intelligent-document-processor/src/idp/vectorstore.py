"""Embeddings + vector DB (Qdrant or Pinecone), one collection per chunking strategy."""
from langchain_openai import OpenAIEmbeddings

from .config import settings


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=settings.embedding_model, api_key=settings.openai_api_key)


def collection_name(strategy: str) -> str:
    return f"{settings.collection_prefix}-{strategy}"


def get_store(strategy: str, collection: str | None = None):
    name = collection or collection_name(strategy)
    embeddings = get_embeddings()

    if settings.vector_backend == "qdrant":
        from langchain_qdrant import QdrantVectorStore
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
        if not client.collection_exists(name):
            client.create_collection(
                name, vectors_config=VectorParams(size=settings.embedding_dim, distance=Distance.COSINE)
            )
        return QdrantVectorStore(client=client, collection_name=name, embedding=embeddings)

    if settings.vector_backend == "pinecone":
        from langchain_pinecone import PineconeVectorStore
        from pinecone import Pinecone, ServerlessSpec

        pc = Pinecone(api_key=settings.pinecone_api_key)
        if name not in pc.list_indexes().names():
            pc.create_index(
                name=name,
                dimension=settings.embedding_dim,
                metric="cosine",
                spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
            )
        return PineconeVectorStore(index=pc.Index(name), embedding=embeddings)

    raise ValueError("VECTOR_BACKEND must be 'qdrant' or 'pinecone'")
