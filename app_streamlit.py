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

# Add the src directory to Python path
sys.path.append(str(Path(__file__).parent / "src"))

# Import your RAG modules
from ragmh import (
    run_rag_pipeline,
    verify_ollama_connection,
    init_chromadb,
    get_collection_stats
)
from ragmh.vanilla_llm import generate_vanilla_response, compare_rag_vs_vanilla

# Page config
st.set_page_config(
    page_title="Mental Health RAG - Compare & Evaluate",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS
st.markdown("""
<style>
    .response-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .rag-response {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .vanilla-response {
        background-color: #fce4ec;
        border-left: 4px solid #e91e63;
    }
    .source-card {
        background-color: #e8eaf0;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #4a90e2;
    }
    .eval-card {
        background-color: #f5f5f5;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .comparison-metric {
        text-align: center;
        padding: 10px;
        background-color: #ffffff;
        border-radius: 8px;
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
if 'comparisons' not in st.session_state:
    st.session_state.comparisons = []
if 'evaluations' not in st.session_state:
    st.session_state.evaluations = []
if 'current_comparison' not in st.session_state:
    st.session_state.current_comparison = None

# Helper functions
def check_system_status():
    """Check if all systems are ready"""
    status = {}
    
    try:
        client = init_chromadb()
        collection = client.get_collection("mental_health_rag")
        status['chromadb'] = True
        status['doc_count'] = collection.count()
    except:
        status['chromadb'] = False
        status['doc_count'] = 0
    
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

def save_evaluation(comparison_id, rag_ratings, vanilla_ratings, preference, notes):
    """Save evaluation data"""
    evaluation = {
        'id': comparison_id,
        'timestamp': datetime.now().isoformat(),
        'rag_ratings': rag_ratings,
        'vanilla_ratings': vanilla_ratings,
        'preference': preference,
        'notes': notes,
        'rag_avg': sum(rag_ratings.values()) / len(rag_ratings),
        'vanilla_avg': sum(vanilla_ratings.values()) / len(vanilla_ratings)
    }
    st.session_state.evaluations.append(evaluation)
    return evaluation

# Main UI
st.title("🧠 Mental Health RAG - Compare & Evaluate")
st.markdown('<div class="warning-banner">💙 This is an AI assistant for mental health information. For emergencies, please contact crisis services: <b>999</b> (UK) or your local emergency number.</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Mode Selection
    mode = st.radio(
        "🎯 Mode",
        ["Single Response", "Compare RAG vs Vanilla", "Evaluate Responses"],
        help="Choose between single RAG response or comparison mode"
    )
    
    # LLM Selection
    llm_choice = st.selectbox(
        "🤖 Language Model",
        ["ollama", "gpt-3.5-turbo-0125", "gemini-1.5-flash-8b"],
        help="Select the AI model to use"
    )
    
    # Source Filter (only for RAG mode)
    if mode != "Compare RAG vs Vanilla":
        source_filter = st.selectbox(
            "📚 Filter by Source",
            ["all", "counselchat", "reddit", "mind.org.uk"],
            format_func=lambda x: "All Sources" if x == "all" else format_source_name(x)
        )
    else:
        source_filter = "all"
    
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

# Main content area
if mode == "Single Response":
    tab1, tab2 = st.tabs(["💬 Chat", "📜 History"])
    
    with tab1:
        query = st.text_area(
            "How can I help you today?",
            placeholder="Example: How can I manage anxiety before a job interview?",
            height=100
        )
        
        if st.button("🔍 Get Support", type="primary"):
            if query:
                with st.spinner("🤔 Thinking..."):
                    try:
                        result = run_rag_pipeline(
                            query=query,
                            source_filter=source_filter if source_filter != "all" else None,
                            save_log=True,
                            verbose=False,
                            llm=llm_choice
                        )
                        
                        if result:
                            st.session_state.history.append({
                                'timestamp': datetime.now(),
                                'query': query,
                                'result': result,
                                'llm': llm_choice
                            })
                            
                            st.markdown("### 💬 Response")
                            st.markdown(f'<div class="response-box rag-response">{result["response"]}</div>', unsafe_allow_html=True)
                            
                            with st.expander("📚 View Sources"):
                                for source, context in zip(result['sources'], result['contexts']):
                                    st.markdown(f"**{format_source_name(source)}**")
                                    st.markdown(f'<div class="source-card">{context[:300]}...</div>', unsafe_allow_html=True)
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Response Time", f"{result['duration_seconds']:.2f}s")
                            with col2:
                                st.metric("Sources Used", result['num_contexts'])
                            with col3:
                                st.metric("Model", llm_choice)
                    
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
    
    with tab2:
        st.header("📜 Response History")
        if st.session_state.history:
            for item in reversed(st.session_state.history[-10:]):
                with st.expander(f"🕐 {item['timestamp'].strftime('%H:%M:%S')} - {item['query'][:50]}..."):
                    st.write(item['result']['response'])

elif mode == "Compare RAG vs Vanilla":
    st.header("🔬 Compare RAG vs Vanilla LLM")
    
    query = st.text_area(
        "Enter your question to compare responses:",
        placeholder="Example: What are some coping strategies for social anxiety?",
        height=100
    )
    
    if st.button("🔍 Generate Comparison", type="primary"):
        if query:
            with st.spinner("🤔 Generating both responses..."):
                try:
                    # Generate RAG response
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 📚 RAG Response")
                        st.caption("*With context from knowledge base*")
                    
                    with col2:
                        st.markdown("### 🤖 Vanilla LLM Response")
                        st.caption("*Without any context*")
                    
                    # RAG Response
                    rag_start = time.time()
                    rag_result = run_rag_pipeline(
                        query=query,
                        save_log=False,
                        verbose=False,
                        llm=llm_choice
                    )
                    rag_time = time.time() - rag_start
                    
                    # Vanilla Response
                    vanilla_start = time.time()
                    vanilla_response = generate_vanilla_response(
                        query=query,
                        llm=llm_choice
                    )
                    vanilla_time = time.time() - vanilla_start
                    
                    # Display side by side
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f'<div class="response-box rag-response">{rag_result["response"]}</div>', unsafe_allow_html=True)
                        st.metric("Response Time", f"{rag_time:.2f}s")
                        st.metric("Sources Used", rag_result['num_contexts'])
                        
                        with st.expander("📚 Context Sources"):
                            for source in rag_result['sources']:
                                st.write(f"• {format_source_name(source)}")
                    
                    with col2:
                        st.markdown(f'<div class="response-box vanilla-response">{vanilla_response}</div>', unsafe_allow_html=True)
                        st.metric("Response Time", f"{vanilla_time:.2f}s")
                        st.metric("Sources Used", "0 (No RAG)")
                    
                    # Save comparison
                    comparison = {
                        'id': datetime.now().strftime('%Y%m%d_%H%M%S'),
                        'query': query,
                        'rag_response': rag_result['response'],
                        'vanilla_response': vanilla_response,
                        'rag_time': rag_time,
                        'vanilla_time': vanilla_time,
                        'rag_sources': rag_result['sources'],
                        'llm': llm_choice,
                        'timestamp': datetime.now()
                    }
                    st.session_state.comparisons.append(comparison)
                    st.session_state.current_comparison = comparison
                    
                    # Comparison Metrics
                    st.markdown("### 📊 Comparison Metrics")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.markdown('<div class="comparison-metric">', unsafe_allow_html=True)
                        st.metric("Speed Difference", f"{rag_time/vanilla_time:.1f}x slower")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown('<div class="comparison-metric">', unsafe_allow_html=True)
                        st.metric("RAG Length", f"{len(rag_result['response'].split())} words")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown('<div class="comparison-metric">', unsafe_allow_html=True)
                        st.metric("Vanilla Length", f"{len(vanilla_response.split())} words")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col4:
                        st.markdown('<div class="comparison-metric">', unsafe_allow_html=True)
                        st.metric("Context Chunks", rag_result['num_contexts'])
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Quick Evaluation
                    st.markdown("### ⭐ Quick Evaluation")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if st.button("👍 RAG is Better", use_container_width=True):
                            st.success("Preference recorded!")
                    
                    with col2:
                        if st.button("👍 Vanilla is Better", use_container_width=True):
                            st.success("Preference recorded!")
                    
                    with col3:
                        if st.button("🤝 Both are Good", use_container_width=True):
                            st.success("Preference recorded!")
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

elif mode == "Evaluate Responses":
    st.header("📋 Response Evaluation")
    
    # Evaluation criteria
    criteria = {
        "empathy": "Shows understanding and compassion",
        "specificity": "Provides concrete, actionable advice",
        "safety": "Avoids harmful suggestions, includes disclaimers",
        "professionalism": "Suggests professional help when appropriate",
        "clarity": "Clear, easy to understand language"
    }
    
    if st.session_state.current_comparison:
        comp = st.session_state.current_comparison
        st.info(f"📝 Evaluating responses for: *{comp['query']}*")
        
        # Create evaluation form
        with st.form("evaluation_form"):
            st.markdown("### Rate each response on the following criteria (1-5 scale):")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📚 RAG Response")
                rag_ratings = {}
                for criterion, description in criteria.items():
                    rag_ratings[criterion] = st.slider(
                        f"{criterion.capitalize()} - {description}",
                        1, 5, 3,
                        key=f"rag_{criterion}"
                    )
            
            with col2:
                st.markdown("#### 🤖 Vanilla Response")
                vanilla_ratings = {}
                for criterion, description in criteria.items():
                    vanilla_ratings[criterion] = st.slider(
                        f"{criterion.capitalize()} - {description}",
                        1, 5, 3,
                        key=f"vanilla_{criterion}"
                    )
            
            # Overall preference
            st.markdown("### Overall Preference")
            preference = st.radio(
                "Which response would you recommend to a user?",
                ["RAG Response", "Vanilla Response", "Both Equally Good", "Neither"]
            )
            
            # Additional notes
            notes = st.text_area(
                "Additional comments (optional):",
                placeholder="Any specific observations about the responses..."
            )
            
            # Submit button
            if st.form_submit_button("Submit Evaluation", type="primary"):
                eval_data = save_evaluation(
                    comp['id'],
                    rag_ratings,
                    vanilla_ratings,
                    preference,
                    notes
                )
                st.success("✅ Evaluation saved!")
                st.balloons()
                
                # Show summary
                st.markdown("### 📊 Evaluation Summary")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("RAG Average", f"{eval_data['rag_avg']:.2f}/5.0")
                
                with col2:
                    st.metric("Vanilla Average", f"{eval_data['vanilla_avg']:.2f}/5.0")
    
    else:
        st.warning("⚠️ Please generate a comparison first using 'Compare RAG vs Vanilla' mode")
    
    # Show evaluation history
    if st.session_state.evaluations:
        st.markdown("### 📈 Evaluation History")
        
        # Convert to DataFrame for analysis
        eval_df = pd.DataFrame(st.session_state.evaluations)
        
        # Average scores
        col1, col2, col3 = st.columns(3)
        with col1:
            avg_rag = eval_df['rag_avg'].mean()
            st.metric("Avg RAG Score", f"{avg_rag:.2f}/5.0")
        
        with col2:
            avg_vanilla = eval_df['vanilla_avg'].mean()
            st.metric("Avg Vanilla Score", f"{avg_vanilla:.2f}/5.0")
        
        with col3:
            pref_counts = eval_df['preference'].value_counts()
            if 'RAG Response' in pref_counts:
                st.metric("RAG Preferred", f"{pref_counts['RAG Response']} times")
        
        # Visualization
        if len(eval_df) > 1:
            fig = px.box(
                eval_df,
                y=['rag_avg', 'vanilla_avg'],
                labels={'value': 'Average Score', 'variable': 'Response Type'},
                title="Score Distribution: RAG vs Vanilla"
            )
            st.plotly_chart(fig, use_container_width=True)

# Export functionality
with st.sidebar:
    st.markdown("---")
    st.header("📤 Export Data")
    
    if st.button("💾 Export Comparisons"):
        if st.session_state.comparisons:
            data = json.dumps(st.session_state.comparisons, default=str, indent=2)
            st.download_button(
                label="Download Comparisons JSON",
                data=data,
                file_name=f"comparisons_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    if st.button("📊 Export Evaluations"):
        if st.session_state.evaluations:
            eval_df = pd.DataFrame(st.session_state.evaluations)
            csv = eval_df.to_csv(index=False)
            st.download_button(
                label="Download Evaluations CSV",
                data=csv,
                file_name=f"evaluations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

# Footer
st.markdown("---")
st.markdown("💙 This tool is for evaluation purposes. Always consult mental health professionals for serious concerns.")
