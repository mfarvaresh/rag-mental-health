import streamlit as st
from src.ragmh.chains import run_rag_pipeline
import os

st.set_page_config(page_title="Mental Health RAG Demo", page_icon="🧠", layout="centered")
st.title("🧠 Mental Health RAG System")
st.markdown("""
A Retrieval-Augmented Generation (RAG) system for mental health support.\
Choose your LLM, enter a question, and get evidence-based, empathetic answers with context.
""")

# Sidebar: Model and Source selection
with st.sidebar:
    st.header("Settings")
    llm = st.selectbox(
        "LLM Backend",
        ["ollama", "gpt-3.5-turbo-0125", "gemini-1.5-flash-8b"],
        format_func=lambda x: {
            "ollama": "Ollama (Local)",
            "gpt-3.5-turbo-0125": "OpenAI GPT-3.5",
            "gemini-1.5-flash-8b": "Gemini 1.5"
        }[x],
    )
    source_filter = st.selectbox(
        "Context Source (optional)",
        [None, "counselchat", "reddit", "mind.org.uk"],
        format_func=lambda x: x if x else "All sources"
    )
    st.markdown("---")
    st.markdown("**Tip:** For OpenAI or Gemini, set your API key in a `.env` file or environment variable.")

# Main input
query = st.text_area("Enter your mental health question:", height=80, key="query")

if st.button("Get Answer", type="primary") or (query and st.session_state.get("auto_submit", False)):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving answer..."):
            result = run_rag_pipeline(query, source_filter=source_filter, verbose=False, llm=llm, save_log=False)
        if not result:
            st.error("No response generated. Please check your backend or try a different query.")
        else:
            st.markdown("## 💬 Response")
            st.success(result["response"])
            st.markdown("---")
            st.markdown(f"**⏱️ Response time:** {result['duration_seconds']:.2f} seconds")
            st.markdown(f"**Sources used:** {', '.join(result['sources'])}")
            with st.expander("Show retrieved context snippets"):
                for i, ctx in enumerate(result["contexts"], 1):
                    st.markdown(f"**Context {i}:**\n{ctx}")

# Footer
st.markdown("---")
st.caption("RAG Mental Health Demo | Powered by Ollama, OpenAI, Gemini, ChromaDB | 2025")
