"""
Mental Health RAG Streamlit UI
Place this file in your project root or src/ directory
Run with: streamlit run app.py
"""

import streamlit as st
import sys
from pathlib import Path
import time
from datetime import datetime
import json

# Add the src directory to Python path
sys.path.append(str(Path(__file__).parent / "src"))

# Import your RAG modules
from ragmh import (
    run_rag_pipeline,
    verify_ollama_connection,
    init_chromadb,
    get_collection_stats
)

# Page config
st.set_page_config(
    page_title="Mental Health Support Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .response-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .source-card {
        background-color: #e8eaf0;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #4a90e2;
    }
    .stat-metric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .warning-banner {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'llm_status' not in st.session_state:
    st.session_state.llm_status = {}

# Helper functions
def check_system_status():
    """Check if all systems are ready"""
    status = {}
    
    # Check ChromaDB
    try:
        client = init_chromadb()
        collection = client.get_collection("mental_health_rag")
        status['chromadb'] = True
        status['doc_count'] = collection.count()
    except:
        status['chromadb'] = False
        status['doc_count'] = 0
    
    # Check Ollama
    status['ollama'] = verify_ollama_connection()
    
    return status

def format_source_name(source):
    """Format source names for display"""
    source_map = {
        'counselchat': '👨‍⚕️ Professional Therapist',
        'reddit': '💬 Reddit Community',
        'mind.org.uk': '🏥 Mind.org.uk',
        'pubmed': '📚 PubMed Research',
        'who': '🌍 WHO Guidelines'
    }
    return source_map.get(source, source)

# Main UI
st.title("🧠 Mental Health Support Assistant")
st.markdown('<div class="warning-banner">💙 This is an AI assistant for mental-health information. If you need emergency help, dial 999 (UK) or your local emergency number.</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # LLM Selection
    llm_choice = st.selectbox(
        "🤖 Language Model",
        ["ollama", "gpt-3.5-turbo-0125", "gemini-1.5-flash-8b"],
        help="Select the AI model to use"
    )
    
    # Source Filter
    source_filter = st.selectbox(
        "📚 Filter by Source",
        ["all", "counselchat", "reddit", "mind.org.uk"],
        format_func=lambda x: "All Sources" if x == "all" else format_source_name(x)
    )
    
    # Advanced Settings
    with st.expander("🔧 Advanced Settings"):
        top_k = st.slider("Number of contexts to retrieve", 3, 10, 5)
        show_stats = st.checkbox("Show response statistics", value=True)
        save_logs = st.checkbox("Save interaction logs", value=True)
    
    # System Status
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
    
    # Clear History
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()

# Main content area
tab1, tab2, tab3 = st.tabs(["💬 Chat", "📜 History", "📈 Analytics"])

with tab1:
    # Query input
    query = st.text_area(
        "How can I help you today?",
        placeholder="Example: How can I manage anxiety before a job interview?",
        height=100,
        key="query_input"
    )
    
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        submit = st.button("🔍 Get Support", type="primary", use_container_width=True)
    with col2:
        clear = st.button("🔄 Clear", use_container_width=True)
    
    if clear:
        st.session_state.query_input = ""
        st.rerun()
    
    # Process query
    if submit and query:
        with st.spinner("🤔 Thinking..."):
            start_time = time.time()
            
            # Check if LLM needs API key
            if llm_choice != "ollama":
                import os
                if llm_choice == "gpt-3.5-turbo-0125" and not os.getenv("OPENAI_API_KEY"):
                    st.error("❌ OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
                    st.stop()
                elif llm_choice == "gemini-1.5-flash-8b" and not os.getenv("GEMINI_API_KEY"):
                    st.error("❌ Gemini API key not found. Please set GEMINI_API_KEY environment variable.")
                    st.stop()
            
            # Run RAG pipeline
            try:
                result = run_rag_pipeline(
                    query=query,
                    source_filter=source_filter if source_filter != "all" else None,
                    save_log=save_logs,
                    verbose=False,
                    llm=llm_choice
                )
                
                if result:
                    # Add to history
                    st.session_state.history.append({
                        'timestamp': datetime.now(),
                        'query': query,
                        'result': result,
                        'llm': llm_choice,
                        'source_filter': source_filter
                    })
                    
                    # Display response
                    st.markdown("### 💬 Response")
                    st.markdown(f'<div class="response-box">{result["response"]}</div>', unsafe_allow_html=True)
                    
                    # Display sources
                    if result.get('sources'):
                        st.markdown("### 📚 Sources Used")
                        cols = st.columns(2)
                        for i, (source, context) in enumerate(zip(result['sources'], result['contexts'])):
                            with cols[i % 2]:
                                with st.expander(f"{format_source_name(source)}", expanded=False):
                                    st.markdown(f'<div class="source-card">{context[:500]}...</div>', unsafe_allow_html=True)
                    
                    # Display stats
                    if show_stats:
                        st.markdown("### 📊 Response Statistics")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Response Time", f"{result['duration_seconds']:.2f}s")
                        with col2:
                            st.metric("Sources Used", result['num_contexts'])
                        with col3:
                            st.metric("Model", llm_choice)
                        with col4:
                            st.metric("Timestamp", datetime.now().strftime("%H:%M:%S"))
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("💡 Tip: Make sure your backend is properly set up and all dependencies are installed.")

with tab2:
    st.header("📜 Conversation History")
    
    if st.session_state.history:
        # Add export button
        if st.button("💾 Export History"):
            history_json = json.dumps(
                [
                    {
                        'timestamp': h['timestamp'].isoformat(),
                        'query': h['query'],
                        'response': h['result']['response'],
                        'sources': h['result']['sources'],
                        'llm': h['llm']
                    }
                    for h in st.session_state.history
                ],
                indent=2
            )
            st.download_button(
                label="Download JSON",
                data=history_json,
                file_name=f"rag_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        # Display history in reverse order (newest first)
        for item in reversed(st.session_state.history):
            with st.expander(f"🕐 {item['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} - {item['query'][:50]}..."):
                st.markdown(f"**Query:** {item['query']}")
                st.markdown(f"**Model:** {item['llm']} | **Filter:** {item['source_filter']}")
                st.markdown(f"**Response:** {item['result']['response']}")
                st.markdown(f"**Sources:** {', '.join([format_source_name(s) for s in item['result']['sources']])}")
    else:
        st.info("No conversation history yet. Start by asking a question!")

with tab3:
    st.header("📈 System Analytics")
    
    if status['chromadb']:
        try:
            client = init_chromadb()
            collection = client.get_collection("mental_health_rag")
            stats = get_collection_stats(collection)
            
            st.markdown("### 📊 Vector Database Statistics")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Documents", stats['total_documents'])
                
                # Source distribution
                st.markdown("#### Document Sources")
                for source, count in stats['sources'].items():
                    st.progress(count / stats['total_documents'], text=f"{format_source_name(source)}: {count}")
            
            with col2:
                if st.session_state.history:
                    # Query statistics
                    st.markdown("#### Session Statistics")
                    st.metric("Total Queries", len(st.session_state.history))
                    
                    # Average response time
                    avg_time = sum(h['result']['duration_seconds'] for h in st.session_state.history) / len(st.session_state.history)
                    st.metric("Avg Response Time", f"{avg_time:.2f}s")
                    
                    # Model usage
                    model_usage = {}
                    for h in st.session_state.history:
                        model = h['llm']
                        model_usage[model] = model_usage.get(model, 0) + 1
                    
                    st.markdown("#### Model Usage")
                    for model, count in model_usage.items():
                        st.write(f"{model}: {count} queries")
        except Exception as e:
            st.error(f"Could not load analytics: {str(e)}")
    else:
        st.warning("⚠️ Vector database not initialized. Run setup first!")

# Footer
st.markdown("---")
st.markdown("💙 Remember: This AI assistant provides information and support, but is not a replacement for professional mental health care.")