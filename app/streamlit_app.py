"""Small Streamlit UI for interactive RAG and evaluation."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import project_path, settings
from src.elasticsearch_client import check_connection, get_client
from src.generation import answer_question

st.set_page_config(page_title="OpsRAG", page_icon="🔎", layout="wide")
st.title("OpsRAG")
st.caption("Multi-source operations knowledge and RAG evaluation lab")

# ── Sidebar: service health ───────────────────────────────────────────────────
with st.sidebar:
    page = st.radio("View", ["Ask", "Evaluation"])
    st.divider()
    st.caption("**Service status**")
    ok, msg = check_connection()
    if ok:
        st.success(msg, icon="✅")
        # Show indexed document count when connected.
        try:
            _client = get_client()
            try:
                count = _client.count(index=settings.elasticsearch_index).get("count", "?")
            finally:
                _client.close()
            st.caption(f"Index `{settings.elasticsearch_index}`: **{count}** chunks")
        except Exception:
            pass
    else:
        st.error(msg, icon="🔴")
    st.divider()
    st.caption(f"Embedding: `{settings.embedding_model}`")
    st.caption(f"LLM: `{settings.ollama_model}`")

# ── Ask view ──────────────────────────────────────────────────────────────────
if page == "Ask":
    method = st.selectbox(
        "Retrieval method",
        ["BM25", "Vector", "Hybrid"],
        index=2,
        help="BM25: keyword · Vector: dense-vector kNN · Hybrid: Elasticsearch RRF",
    )
    question = st.text_input("Question", placeholder="e.g. What are the goals of the NECP?")

    if st.button("Search", type="primary", disabled=not question.strip()):
        with st.spinner("Retrieving and generating…"):
            client = get_client()
            try:
                result = answer_question(
                    question, method, prompt_name="improved", es_client=client
                )
            except Exception as exc:
                st.error(
                    "Request failed — check Elasticsearch and Ollama are running and"
                    " the configured models are available.",
                    icon="🔴",
                )
                with st.expander("Error details"):
                    st.code(f"{type(exc).__name__}: {exc}")
                st.stop()
            finally:
                client.close()

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

        st.subheader(f"Retrieved chunks ({method})")
        for item in result["retrieved_chunks"]:
            score_str = f"{item['score']:.4f}" if item.get("score") is not None else "n/a"
            with st.expander(f"{item.get('chunk_id', 'unknown')} · score {score_str}"):
                st.markdown(f"**{item.get('title', '')} — {item.get('section', '')}**")
                st.markdown(item["text"])
                if item.get("source", "").startswith(("http://", "https://")):
                    st.caption(item["source"])

# ── Evaluation view ───────────────────────────────────────────────────────────
else:
    st.subheader("Evaluation results")
    summary_path = project_path("results", "summary.csv")
    baseline_path = project_path("results", "baseline.csv")
    improved_path = project_path("results", "improved.csv")

    if not summary_path.exists() or summary_path.read_text(encoding="utf-8").strip().count("\n") < 2:
        st.info(
            "No evaluation results found. Run `python scripts/evaluate.py` after"
            " Elasticsearch and Ollama are running, then reload this page.",
            icon="ℹ️",
        )
    else:
        summary = pd.read_csv(summary_path)
        display = summary.set_index("experiment").T
        display.index = [str(item) for item in display.index]
        display.columns = [
            "Baseline" if value == "baseline" else "Improved"
            for value in display.columns
        ]
        metric_rows = ["Recall@5", "MRR", "Answer Correctness", "Citation Coverage"]
        st.dataframe(
            display.loc[[r for r in metric_rows if r in display.index]],
            use_container_width=True,
        )

        st.subheader("Failure categories")
        frames = []
        for label, path in [("Baseline", baseline_path), ("Improved", improved_path)]:
            if path.exists():
                df = pd.read_csv(path)
                frames.append(df["failure_category"].value_counts().rename(label))
        if frames:
            failures = pd.concat(frames, axis=1).fillna(0).astype(int)
            st.dataframe(failures, use_container_width=True)

        # Per-question drill-down
        st.subheader("Per-question detail")
        tab_base, tab_impr = st.tabs(["Baseline", "Improved"])
        for tab, path, label in [
            (tab_base, baseline_path, "Baseline"),
            (tab_impr, improved_path, "Improved"),
        ]:
            with tab:
                if path.exists():
                    df = pd.read_csv(path)
                    st.dataframe(
                        df[
                            [
                                "id",
                                "question",
                                "recall_at_5",
                                "mrr",
                                "answer_correctness",
                                "citation_coverage",
                                "failure_category",
                            ]
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )
