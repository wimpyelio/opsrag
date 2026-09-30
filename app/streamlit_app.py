"""Small Streamlit UI for interactive RAG and evaluation."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import project_path, settings
from src.elasticsearch_client import get_client
from src.generation import answer_question

st.set_page_config(page_title="OpsRAG", page_icon="🔎", layout="wide")
st.title("OpsRAG")
st.caption("Multi-source operations knowledge and RAG evaluation lab")

page = st.sidebar.radio("View", ["Ask", "Evaluation"])

if page == "Ask":
    method = st.sidebar.selectbox("Retrieval method", ["BM25", "Vector", "Hybrid"], index=2)
    question = st.text_input("Question")

    if st.button("Search", type="primary", disabled=not question.strip()):
        client = get_client()
        try:
            result = answer_question(question, method, prompt_name="improved", es_client=client)
            st.subheader("Answer")
            st.write(result["answer"])

            st.subheader("Sources")
            if result["sources"]:
                for source in result["sources"]:
                    st.markdown(f"- **{source['title']} — {source['section']}**")
                    if source["source"].startswith(("http://", "https://")):
                        st.caption(source["source"])
            else:
                st.write("No sources returned.")

            st.subheader("Retrieved Chunks")
            for item in result["retrieved_chunks"]:
                with st.expander(f"{item.get('chunk_id', '')} · {item.get('score', '')}"):
                    st.markdown(item["text"])
        except Exception as exc:
            st.error("Request failed. Check the Elasticsearch/Ollama configuration and terminal output.")
            st.caption(f"Error type: {type(exc).__name__}")
        finally:
            client.close()

else:
    st.subheader("Evaluation")
    summary_path = project_path("results", "summary.csv")
    baseline_path = project_path("results", "baseline.csv")
    improved_path = project_path("results", "improved.csv")
    if not summary_path.exists() or summary_path.read_text(encoding="utf-8").strip().count("\n") < 2:
        st.info("Run `python scripts/evaluate.py` after Elasticsearch and Ollama are running to populate this page with measured results.")
    else:
        summary = pd.read_csv(summary_path)
        display = summary.set_index("experiment").T
        display.index = [str(item) for item in display.index]
        display.columns = ["Baseline" if value == "baseline" else "Improved" for value in display.columns]
        st.dataframe(display.loc[["Recall@5", "MRR", "Answer Correctness", "Citation Coverage"]], use_container_width=True)

        st.subheader("Failure categories")
        frames = []
        for label, path in [("Baseline", baseline_path), ("Improved", improved_path)]:
            if path.exists():
                df = pd.read_csv(path)
                frames.append(df["failure_category"].value_counts().rename(label))
        if frames:
            failures = pd.concat(frames, axis=1).fillna(0).astype(int)
            st.dataframe(failures, use_container_width=True)

st.caption(f"Embedding model: `{settings.embedding_model}` · Ollama model: `{settings.ollama_model}`")
