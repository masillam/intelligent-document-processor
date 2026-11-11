"""Streamlit dashboard comparing chunking strategies.

Run: streamlit run dashboard/app.py
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

RESULTS = Path(__file__).resolve().parents[1] / "results" / "eval_results.json"

st.set_page_config(page_title="Chunking Strategy Comparison", layout="wide")
st.title("Chunking strategy comparison")

if not RESULTS.exists():
    st.warning("No results yet. Run `python evaluation/run_eval.py` first.")
    st.stop()

data = json.loads(RESULTS.read_text())
cfg, results = data["config"], data["results"]

with st.sidebar:
    st.header("Run configuration")
    st.json(cfg)

rows = [{"strategy": r["strategy"], "chunks": r["chunks"], "avg_chunk_chars": r["avg_chunk_chars"], **r["metrics"]} for r in results]
df = pd.DataFrame(rows).set_index("strategy")
metric_cols = [c for c in df.columns if c not in {"chunks", "avg_chunk_chars"}]

st.subheader("Retrieval & generation metrics (higher is better)")
st.bar_chart(df[metric_cols])
st.dataframe(df.style.highlight_max(subset=metric_cols, axis=0), use_container_width=True)

best = df[metric_cols].mean(axis=1).idxmax()
st.success(f"Best overall (mean of metrics): **{best}**")

st.subheader("Per-question scores")
choice = st.selectbox("Strategy", [r["strategy"] for r in results])
sel = next(r for r in results if r["strategy"] == choice)
st.dataframe(pd.DataFrame(sel["per_question"]), use_container_width=True)
