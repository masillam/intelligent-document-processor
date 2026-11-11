"""Retrieval-augmented question answering."""
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .config import settings
from .vectorstore import get_store

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the question using ONLY the context. If the context does not contain the "
            "answer, say you don't know. Be concise and cite sources like [file, p.N].",
        ),
        ("human", "Question: {question}\n\nContext:\n{context}"),
    ]
)


def _format(docs) -> str:
    parts = []
    for d in docs:
        page = d.metadata.get("page")
        tag = f"{d.metadata.get('source', '?')}" + (f", p.{page + 1}" if isinstance(page, int) else "")
        parts.append(f"[{tag}]\n{d.page_content}")
    return "\n\n".join(parts)


def answer_question(
    question: str,
    strategy: str | None = None,
    top_k: int | None = None,
    collection: str | None = None,
) -> dict:
    strategy = strategy or settings.chunk_strategy
    store = get_store(strategy, collection)
    docs = store.similarity_search(question, k=top_k or settings.top_k)
    llm = ChatOpenAI(model=settings.llm_model, api_key=settings.openai_api_key, temperature=0)
    msg = (PROMPT | llm).invoke({"question": question, "context": _format(docs)})
    return {
        "answer": msg.content,
        "contexts": [d.page_content for d in docs],
        "sources": [
            {"source": d.metadata.get("source"), "page": d.metadata.get("page"), "chunk": d.metadata.get("chunk_index")}
            for d in docs
        ],
    }
