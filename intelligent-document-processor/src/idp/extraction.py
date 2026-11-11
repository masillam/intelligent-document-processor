"""Turn raw document text into structured, searchable information."""
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from .config import settings

MAX_CHARS = 60_000  # long documents are truncated for the one-shot extraction call


class DocumentInsights(BaseModel):
    title: str = Field(description="Best title for the document")
    document_type: str = Field(description="e.g. report, policy, manual, article, contract, notes")
    summary: str = Field(description="Concise 3-5 sentence summary")
    key_points: list[str] = Field(description="5-10 most important takeaways")
    entities: list[str] = Field(description="Named people, organisations, products, places")
    topics: list[str] = Field(description="3-8 topic tags")
    dates_and_figures: list[str] = Field(
        default_factory=list, description="Important dates, amounts and metrics mentioned"
    )


def extract_insights(text: str, llm: ChatOpenAI | None = None) -> DocumentInsights:
    llm = llm or ChatOpenAI(model=settings.llm_model, api_key=settings.openai_api_key, temperature=0)
    structured = llm.with_structured_output(DocumentInsights)
    prompt = (
        "Extract structured information from the document below. "
        "Only use facts present in the text.\n\n---\n" + text[:MAX_CHARS]
    )
    return structured.invoke(prompt)
