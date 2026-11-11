"""Compare chunking strategies with RAGAS and write results/eval_results.json.

Usage:
  python evaluation/run_eval.py --docs data/sample_docs --questions evaluation/eval_questions.json \
      --strategies recursive semantic --chunk-size 1000 --chunk-overlap 150
"""
import argparse
import json
import time
from pathlib import Path

from langchain_openai import ChatOpenAI
from ragas import EvaluationDataset, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall

from idp.config import ROOT, settings
from idp.loaders import SUPPORTED_EXTENSIONS
from idp.pipeline import ingest_file
from idp.qa import answer_question
from idp.vectorstore import get_embeddings

NON_METRIC = {"user_input", "retrieved_contexts", "response", "reference", "reference_contexts"}


def run_strategy(strategy, files, questions, args, run_tag):
    collection = f"{settings.collection_prefix}-eval-{strategy}-{run_tag}"
    stats = []
    for f in files:
        stats.append(
            ingest_file(f, strategy, args.chunk_size, args.chunk_overlap, collection=collection, with_insights=False)
        )

    samples = []
    for q in questions:
        out = answer_question(q["question"], strategy, args.top_k, collection=collection)
        samples.append(
            {
                "user_input": q["question"],
                "retrieved_contexts": out["contexts"],
                "response": out["answer"],
                "reference": q["reference"],
            }
        )

    llm = LangchainLLMWrapper(ChatOpenAI(model=args.judge_model, temperature=0))
    emb = LangchainEmbeddingsWrapper(get_embeddings())
    metrics = [Faithfulness(), LLMContextPrecisionWithReference(), LLMContextRecall()]
    result = evaluate(EvaluationDataset.from_list(samples), metrics=metrics, llm=llm, embeddings=emb)
    df = result.to_pandas()
    cols = [c for c in df.columns if c not in NON_METRIC]

    return {
        "strategy": strategy,
        "chunks": sum(s["chunks"] for s in stats),
        "avg_chunk_chars": round(sum(s["avg_chunk_chars"] for s in stats) / len(stats), 1),
        "metrics": {c: round(float(df[c].mean()), 4) for c in cols},
        "per_question": df[["user_input"] + cols].to_dict(orient="records"),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--docs", default=str(ROOT / "data" / "sample_docs"))
    p.add_argument("--questions", default=str(ROOT / "evaluation" / "eval_questions.json"))
    p.add_argument("--strategies", nargs="+", default=["recursive", "semantic"])
    p.add_argument("--chunk-size", type=int, default=settings.chunk_size)
    p.add_argument("--chunk-overlap", type=int, default=settings.chunk_overlap)
    p.add_argument("--top-k", type=int, default=settings.top_k)
    p.add_argument("--judge-model", default=settings.llm_model)
    p.add_argument("--out", default=str(ROOT / "results" / "eval_results.json"))
    args = p.parse_args()

    files = [f for f in Path(args.docs).iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS]
    questions = json.loads(Path(args.questions).read_text())
    run_tag = str(int(time.time()))

    results = [run_strategy(s, files, questions, args, run_tag) for s in args.strategies]
    payload = {
        "config": {
            "embedding_model": settings.embedding_model,
            "llm_model": settings.llm_model,
            "judge_model": args.judge_model,
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
            "top_k": args.top_k,
            "num_documents": len(files),
            "num_questions": len(questions),
        },
        "results": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))
    for r in results:
        print(r["strategy"], r["metrics"])
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
