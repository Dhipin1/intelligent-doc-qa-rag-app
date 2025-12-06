# src/app.py
import os
import sys
import json
from pathlib import Path

# Ensure project root is on PYTHONPATH when running "streamlit run src/app.py"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402 (import after sys.path fix)

from src.rag import RAGPipeline  # noqa: E402
from src.ingest import build_faiss_index  # noqa: E402
from src.config import DATA_RAW_DIR, INDEX_DIR  # noqa: E402

st.set_page_config(page_title="Document QA Chatbot (RAG)", page_icon="📄", layout="wide")
st.title("📄 Intelligent Document QA Chatbot (LLM + RAG) — Groq")


def load_meta() -> dict:
    """Read index metadata if available."""
    meta_path = INDEX_DIR / "meta.json"
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


with st.sidebar:
    st.header("Settings")

    default_model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    top_k = st.slider("Top-K docs", min_value=2, max_value=12, value=6, step=1)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
    model = st.text_input(
        "Groq model id",
        value=default_model,
        help="Examples: 'llama-3.1-8b-instant' (fast) or 'llama-3.3-70b-versatile' (larger).",
    )

    st.caption(f"Data:  {DATA_RAW_DIR}")
    st.caption(f"Index: {INDEX_DIR}")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Rebuild index"):
            with st.spinner("Building index (this may take a while)..."):
                build_faiss_index()
            st.success("Index rebuilt!")
    with col2:
        if st.button("Clear chat"):
            st.session_state.pop("messages", None)
            st.rerun()

    meta = load_meta()
    if meta:
        st.markdown(
            f"Vectors: {meta.get('num_vectors', '?')} • Chunk size: {meta.get('chunk_size', '?')} "
            f"• Embed: {meta.get('embedding_model', '?')}"
        )

    if not os.getenv("GROQ_API_KEY"):
        st.error("GROQ_API_KEY is not set. Add it to your .env and restart the app.")


@st.cache_resource
def load_rag() -> RAGPipeline:
    return RAGPipeline()


if "messages" not in st.session_state:
    st.session_state.messages = []

rag = load_rag()

# Render chat history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Chat input
user_input = st.chat_input("Ask a question about your documents...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating..."):
            try:
                answer, retrieved = rag.ask(
                    user_input,
                    top_k=top_k,
                    model=model,
                    temperature=temperature,
                )
                st.markdown(answer)
                with st.expander("Sources"):
                    for i, d in enumerate(retrieved, 1):
                        st.write(f"{i}. {d.source}#{d.chunk_id} (score={d.score:.3f})")
            except Exception as e:
                st.error(f"Error while generating answer: {e}")
                answer = "An error occurred. Please check logs and try again."

    st.session_state.messages.append({"role": "assistant", "content": answer})