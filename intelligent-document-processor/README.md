# Intelligent Document Processor

RAG pipeline that turns unstructured **PDF, DOCX, HTML and Markdown** files into structured, searchable content, compares **chunking strategies with RAGAS**, and serves answers through a **FastAPI** service.

**Stack:** Python · LangChain 1.0 · OpenAI API (`text-embedding-3-large`) · Qdrant / Pinecone · FastAPI · RAGAS · Streamlit

## Architecture

```
file ─► loaders ─► extraction (LLM → title, summary, key points, entities, topics)
                └► chunking (recursive | semantic) ─► embeddings ─► vector DB (1 collection / strategy)
                                                                         │
FastAPI /ask ◄── RAG chain (retrieve → grounded answer + sources) ◄──────┘
run_eval.py ─► RAGAS (faithfulness, context precision, context recall) ─► results/eval_results.json ─► dashboard
```

## Quick start

```bash
pip install -e ".[dev]"
cp .env.example .env            # add OPENAI_API_KEY
docker compose up -d            # Qdrant (or set VECTOR_BACKEND=pinecone + PINECONE_API_KEY)

uvicorn idp.api:app --reload    # http://localhost:8000/docs
```

```bash
# ingest (strategy, chunk_size, chunk_overlap are configurable)
curl -F file=@data/sample_docs/remote_work_policy.md -F strategy=recursive \
     -F chunk_size=800 -F chunk_overlap=100 http://localhost:8000/ingest

# ask
curl -X POST http://localhost:8000/ask -H 'Content-Type: application/json' \
     -d '{"question":"How many remote days per week are allowed?","strategy":"recursive"}'

# structured insights for every ingested document
curl http://localhost:8000/documents
```

## Chunking strategies

| Strategy | How it splits | Configurable |
|---|---|---|
| `recursive` | `RecursiveCharacterTextSplitter` on paragraphs → sentences → words | `chunk_size`, `chunk_overlap` |
| `semantic` | `SemanticChunker` breaks where embedding similarity drops; oversized chunks are capped by the recursive splitter | breakpoint threshold, size cap |

## Evaluation & dashboard

```bash
python evaluation/run_eval.py --docs data/sample_docs --questions evaluation/eval_questions.json \
       --strategies recursive semantic --chunk-size 1000 --chunk-overlap 150
streamlit run dashboard/app.py
```

RAGAS metrics: **faithfulness**, **context precision**, **context recall**. Each strategy gets its own isolated collection so runs don't contaminate each other.

> The bundled sample (1 doc, 5 questions) only demonstrates the flow. For meaningful numbers, use your own corpus and 30+ question/reference pairs in `evaluation/eval_questions.json`. Report the scores your own run produces.

## Tests

```bash
pytest
```

## Notes / limitations
- Insight extraction sends up to ~60k characters per document in one call; very long documents are truncated (a map-reduce summariser is a natural extension).
- Scanned PDFs need OCR before ingestion.
