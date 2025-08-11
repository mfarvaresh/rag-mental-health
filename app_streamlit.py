"""
Enhanced Mental Health RAG Streamlit UI (UI_APP.py) with Comparison & Evaluation
Place this file as app_enhanced.py in your project root
Run with: streamlit run app_enhanced.py
"""

import streamlit as st
import sys
from pathlib import Path
import time
from datetime import datetime
import json
import pandas as pd
import plotly.express as px

sys.path.append(str(Path(__file__).parent / "src"))

from ragmh import (
    run_rag_pipeline,
    verify_ollama_connection,
    init_chromadb,
    get_collection_stats,
)
from ragmh.vanilla_llm import generate_vanilla_response, compare_rag_vs_vanilla

st.set_page_config(
    page_title="Mental Health RAG - Compare & Evaluate",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .response-box { background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin: 10px 0; }
    .rag-response { background-color: #e3f2fd; border-left: 4px solid #2196f3; }
    .vanilla-response { background-color: #fce4ec; border-left: 4px solid #e91e63; }
    .source-card { background-color: #e8eaf0; padding: 15px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #4a90e2; }
    .eval-card { background-color: #f5f5f5; padding: 15px; border-radius: 8px; margin: 10px 0; }
    .comparison-metric { text-align: center; padding: 10px; background-color: #ffffff; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .warning-banner { background-color: #fff3cd; border: 1px solid #ffeaa7; color: #856404; padding: 10px; border-radius: 5px; margin-bottom: 20px; }
</style>
""",
    unsafe_allow_html=True,
)

if 'history' not in st.session_state:
    st.session_state.history = []
if 'comparisons' not in st.session_state:
    st.session_state.comparisons = []
if 'evaluations' not in st.session_state:
    st.session_state.evaluations = []
if 'current_comparison' not in st.session_state:
    st.session_state.current_comparison = None

def check_system_status():
    """Check system readiness."""
    status = {}
    try:
        client = init_chromadb()
        collection = client.get_collection("mental_health_rag")
        status['chromadb'] = True
        status['doc_count'] = collection.count()
    except Exception:
        status['chromadb'] = False
        status['doc_count'] = 0
    status['ollama'] = verify_ollama_connection()
    return status

def format_source_name(source):
    """Human-readable source labels."""
    source_map = {
        'counselchat': '👨‍⚕️ Professional Therapist',
        'reddit': '💬 Reddit Community',
        'mind.org.uk': '🏥 Mind.org.uk',
        'pubmed': '📚 PubMed Research',
        'who': '🌍 WHO Guidelines',
    }
    return source_map.get(source, source)

def save_evaluation(comparison_id, rag_ratings, vanilla_ratings, preference, notes):
    """Persist evaluation in session state."""
    evaluation = {
        'id': comparison_id,
        'timestamp': datetime.now().isoformat(),
        'rag_ratings': rag_ratings,
        'vanilla_ratings': vanilla_ratings,
        'preference': preference,
        'notes': notes,
        'rag_avg': sum(rag_ratings.values()) / len(rag_ratings),
        'vanilla_avg': sum(vanilla_ratings.values()) / len(vanilla_ratings),
    }
    st.session_state.evaluations.append(evaluation)
    return evaluation

st.title("🧠 Mental Health RAG - Compare & Evaluate")
st.markdown('<div class="warning-banner">💙 This is an AI assistant for mental health information. For emergencies, please contact crisis services: <b>999</b> (UK) or your local emergency number.</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Settings")
    mode = st.radio("🎯 Mode", ["Single Response", "Compare RAG vs Vanilla", "Evaluate Responses"], help="Choose a mode")
    llm_choice = st.selectbox("🤖 Language Model", ["ollama", "gpt-3.5-turbo-0125", "gemini-1.5-flash-8b"], help="Select the AI model")
    if mode != "Compare RAG vs Vanilla":
        source_filter = st.selectbox("📚 Filter by Source", ["all", "counselchat", "reddit", "mind.org.uk"], format_func=lambda x: "All Sources" if x == "all" else format_source_name(x))
    else:
        source_filter = "all"
    st.header("📊 System Status")
    status = check_system_status()
    col1, col2 = st.columns(2)
    with col1:
        if status['chromadb']:
            st.success("✅ Vector DB")
            st.metric("Documents", status['doc_count'])
        else:
            st.error("❌ Vector DB")
    with col2:
        if status['ollama']:
            st.success("✅ Ollama")
        else:
            st.warning("⚠️ Ollama")

if mode == "Single Response":
    tab1, tab2 = st.tabs(["💬 Chat", "📜 History"])
    with tab1:
        query = st.text_area("How can I help you today?", placeholder="Example: How can I manage anxiety before a job interview?", height=100)
        if st.button("🔍 Get Support", type="primary"):
            if query:
                with st.spinner("🤔 Thinking..."):
                    result = run_rag_pipeline(query, source_filter=None if source_filter == "all" else source_filter, verbose=False, llm=llm_choice, save_log=True)
                if result:
                    st.session_state.history.append({
                        'query': query,
                        'response': result['response'],
                        'sources': result['sources'],
                        'timestamp': datetime.now().isoformat(),
                    })
                    st.markdown("### 💬 Response")
                    st.markdown(result['response'])
                    st.markdown("### 📚 Sources")
                    for s in result['sources']:
                        st.markdown(f"- {format_source_name(s)}")
            else:
                st.warning("Please enter a question.")
    with tab2:
        if st.session_state.history:
            for item in reversed(st.session_state.history):
                st.markdown(f"**{item['timestamp']}** — {item['query']}")
                st.markdown(item['response'])
                st.markdown("**Sources:** " + ", ".join(item['sources']))
                st.markdown("---")
        else:
            st.info("No history yet.")

elif mode == "Compare RAG vs Vanilla":
    query = st.text_input("Enter a question to compare:", placeholder="e.g., How do I cope with panic attacks?")
    if st.button("🔬 Compare") and query:
        with st.spinner("Running comparison..."):
            rag_result = run_rag_pipeline(query, source_filter=None if source_filter == "all" else source_filter, verbose=False, llm=llm_choice, save_log=False)
            vanilla_response = generate_vanilla_response(query, llm=llm_choice)
            comparison = compare_rag_vs_vanilla(query, rag_result, llm=llm_choice, verbose=False)
            st.session_state.comparisons.append({
                'query': query,
                'rag': rag_result['response'],
                'vanilla': vanilla_response,
                'timestamp': datetime.now().isoformat(),
            })
        st.markdown("## Results")
        cols = st.columns(2)
        with cols[0]:
            st.subheader("RAG Response")
            st.markdown(f"<div class='response-box rag-response'>{rag_result['response']}</div>", unsafe_allow_html=True)
        with cols[1]:
            st.subheader("Vanilla LLM Response")
            st.markdown(f"<div class='response-box vanilla-response'>{vanilla_response}</div>", unsafe_allow_html=True)

elif mode == "Evaluate Responses":
    if not st.session_state.comparisons:
        st.info("Run a comparison first to evaluate responses.")
    else:
        last = st.session_state.comparisons[-1]
        st.markdown("### Evaluate the latest comparison")
        st.write(f"Question: {last['query']}")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("RAG")
            st.markdown(last['rag'])
            rag_ratings = {
                'empathy': st.slider("Empathy (RAG)", 1, 5, 4),
                'specificity': st.slider("Specificity (RAG)", 1, 5, 4),
                'safety': st.slider("Safety (RAG)", 1, 5, 5),
            }
        with col2:
            st.subheader("Vanilla")
            st.markdown(last['vanilla'])
            vanilla_ratings = {
                'empathy': st.slider("Empathy (Vanilla)", 1, 5, 3),
                'specificity': st.slider("Specificity (Vanilla)", 1, 5, 3),
                'safety': st.slider("Safety (Vanilla)", 1, 5, 4),
            }
        preference = st.selectbox("Which do you prefer?", ["RAG", "Vanilla", "Tie"])
        notes = st.text_area("Notes (optional)")
        if st.button("💾 Save Evaluation"):
            save_evaluation(
                comparison_id=len(st.session_state.comparisons),
                rag_ratings=rag_ratings,
                vanilla_ratings=vanilla_ratings,
                preference=preference,
                notes=notes,
            )
            st.success("Saved evaluation.")
