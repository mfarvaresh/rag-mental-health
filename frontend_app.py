
import streamlit as st
import sys
from pathlib import Path
import time

# Add backend src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Import RAG pipeline
from ragmh import run_rag_pipeline

# App configuration
st.set_page_config(page_title="Mental Health RAG Chatbot", page_icon="💬", layout="wide")

# Sidebar options
st.sidebar.title("⚙️ Settings")

# Light/dark mode toggle
theme = st.sidebar.radio("Theme", ["Light", "Dark"], horizontal=True)
if theme == "Dark":
    st.markdown("<style>body { background-color: #0e1117; color: white; }</style>", unsafe_allow_html=True)

# Context + history settings
max_history = st.sidebar.slider("Max chat history", 1, 20, 10)
max_context = st.sidebar.slider("Max context size", 1, 10, 3)

# Reset chat history button
if st.sidebar.button("🗑️ Reset Chat"):
    st.session_state.chat_history = []

#model = st.sidebar.selectbox("Choose a model", ["Mistral", "Gemma", "Other (mocked)"])

# Source toggle
show_sources = st.sidebar.toggle("Show source contexts", value=False)

# Title
st.markdown("<h2 style='text-align: center;'>Hi! What can I help you with today?</h2>", unsafe_allow_html=True)

#st.title("Mental Health Chatbot")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat input
user_query = st.text_input("Ask me anything about mental health...")
        
if user_query:
    # Save user message
    st.session_state.chat_history.append({"role": "user", "content": user_query})

    # Generate RAG response
    with st.spinner("Thinking..."):
        try:
            result = run_rag_pipeline(user_query)
            response = result["response"] if isinstance(result, dict) else str(result)
            sources = result.get("contexts", []) if isinstance(result, dict) else []
        except Exception as e:
            response = f"⚠️ Error: {e}"
            sources = []
    
    # Simulate streamed response (typing effect)
    streamed = ""
    with st.chat_message("assistant"):
        msg_box = st.empty()
        for token in response.split():
            streamed += token + " "
            msg_box.markdown(streamed)
            time.sleep(0.03)

        # Show sources if toggled
        if show_sources and sources:
            st.markdown("---")
            st.markdown("**📚 Sources:**")
            for src in sources:
                st.info(src)

    # Save bot response
    st.session_state.chat_history.append({"role": "assistant", "content": response})

    # Enforce history limit
    st.session_state.chat_history = st.session_state.chat_history[-2*max_history:]
    
# Display previous chat history (user + assistant)
for message in st.session_state.chat_history[:-1]:  # skip last because it's already shown
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


#if user_query:
    #with st.spinner("Generating answer..."):
        #result = run_rag_pipeline(user_query)
        #response_text = result["response"] if isinstance(result, dict) else result
        #st.write(response_text)
