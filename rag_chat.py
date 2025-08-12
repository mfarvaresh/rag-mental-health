import sys
import time
from io import BytesIO
from pathlib import Path
from datetime import datetime
import base64
import streamlit as st

# Add backend src to path
sys.path.append(str(Path(__file__).parent / "src"))
from ragmh import run_rag_pipeline  # your RAG fn

# ---------- Optional TTS (audio output only) ----------
try:
    from gtts import gTTS  # pip install gTTS
    HAS_TTS = True
except Exception:
    HAS_TTS = False

# ---------- Page & Styles ----------
st.set_page_config(page_title="Chatbot", page_icon="./assets/logo.png", layout="wide")

st.markdown("""
<style>
/* Buttons */
.stButton>button, .stDownloadButton>button {
    background-color: #003926;
    color: white !important;
    border: none;
    border-radius: 10px;
    height: 40px;
    width: 100%;
    font-weight: 600;
}
.stButton>button:hover, .stDownloadButton>button:hover {
    background-color: #002a1d;
    color: white !important;
}

/* Message rows */
.msg{display:flex;margin:6px 0}
.msg.user{justify-content:flex-end}
.msg.assistant{justify-content:flex-start}

/* Bubbles */
.bubble{
  padding:12px 14px; border-radius:16px; line-height:1.5;
  word-wrap:break-word; max-width:75%;
}
.bubble.user { margin-right:0; }
.bubble.assistant { margin-left:0; }

/* Divider line */
.divider{height:1px;background:linear-gradient(90deg,transparent,rgba(125,125,125,.3),transparent);margin:8px 0 4px}

/* Default theme bubbles (kept) */
@media (prefers-color-scheme: light){
  .bubble.user{background:#dbeafe;border:1px solid #93c5fd;color:#1e3a8a}
  .bubble.assistant{background:#f1f5f9;border:1px solid #e2e8f0;color:#0f172a}
  .bubble.source{background:#fff7ed;border:1px solid #fed7aa;color:#7c2d12}
}
@media (prefers-color-scheme: dark){
  .bubble.user{background:#1e3a8a;border:1px solid #3b82f6;color:#e5e7eb}
  .bubble.assistant{background:#111827;border:1px solid #374151;color:#e5e7eb}
  .bubble.source{background:#1f2937;border:1px solid #374151;color:#e5e7eb}
}

/* Final overrides: your chosen look */
/* User bubble: green bg, white text, NO border */
.bubble.user {
  background:#478D76 !important;
  color:#ffffff !important;
  border:none !important;
  border-radius:16px !important;
}
/* Assistant bubble: remove border, keep its bg */
.bubble.assistant {
  border:none !important;
  border-radius:16px !important;
}

/* Page background + header */
[data-testid="stAppViewContainer"] { background-color: #71AA97 !important; }
[data-testid="stHeader"] {
    background-color: #71AA97 !important;
    box-shadow: none !important;
    border-bottom: none !important;
}

/* Bottom area behind chat input */
[data-testid="stBottomBlockContainer"] { background-color: #71AA97 !important; }

/* Chat input styling (no red hover borders) */
[data-testid="stChatInput"] textarea {
    color: #111 !important;
    border: 1px solid transparent !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.22) !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #2b4d40 !important; }

/* Tip/notification box: white text on darker overlay */
div[data-baseweb="notification"] {
    background-color: rgba(0, 0, 0, 0.4) !important;
    color: white !important;
    box-shadow: none !important;
}
div[data-baseweb="notification"] a {
    color: #ffffff !important;
    font-weight: 600;
    text-decoration: underline;
}
</style>
""", unsafe_allow_html=True)

# ---------- Defaults & Session ----------
DEFAULT_MAX_HISTORY = 15
DEFAULT_MAX_CONTEXT = 3
DEFAULT_SHOW_SOURCES = False

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []   # list[{role, content, ts, audio?}]
if "page" not in st.session_state:
    st.session_state.page = "Home"       # landing page
if "max_history" not in st.session_state:
    st.session_state.max_history = DEFAULT_MAX_HISTORY
if "max_context" not in st.session_state:
    st.session_state.max_context = DEFAULT_MAX_CONTEXT
if "show_sources" not in st.session_state:
    st.session_state.show_sources = DEFAULT_SHOW_SOURCES

# --- Handle navigation via query param (logo click) ---
try:
    # Streamlit >= 1.33
    if (st.query_params.get("nav") or "") == "home":
        st.session_state.page = "Home"
        st.query_params.clear()
except Exception:
    # Older Streamlit fallback
    qp = st.experimental_get_query_params()
    if qp.get("nav", [""])[0] == "home":
        st.session_state.page = "Home"
        st.experimental_set_query_params()

# ---------- Helpers ----------
def _load_logo_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def add_message(role: str, content: str, audio_bytes: bytes | None = None):
    msg = {"role": role, "content": content, "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    if audio_bytes:
        msg["audio"] = audio_bytes
    st.session_state.chat_history.append(msg)
    N = st.session_state.max_history
    st.session_state.chat_history = st.session_state.chat_history[-2 * N:]

def render_message(role: str, content: str, audio_bytes: bytes | None = None):
    who = "user" if role == "user" else "assistant"
    st.markdown(f"<div class='msg {who}'><div class='bubble {who}'>{content}</div></div>",
                unsafe_allow_html=True)
    if audio_bytes:
        st.audio(audio_bytes, format="audio/mp3")
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

def streamed_markdown(text: str, delay: float = 0.02):
    holder = st.empty()
    acc = ""
    for tok in text.split():
        acc += tok + " "
        holder.markdown(f"<div class='msg assistant'><div class='bubble assistant'>{acc}</div></div>",
                        unsafe_allow_html=True)
        time.sleep(delay)

def call_rag(query: str, max_context: int):
    import inspect
    try:
        sig = inspect.signature(run_rag_pipeline)
        if "max_chunks" in sig.parameters:
            return run_rag_pipeline(query, max_chunks=max_context)
        else:
            return run_rag_pipeline(query)
    except TypeError:
        return run_rag_pipeline(query)

def synthesize_tts(text: str):
    """Return BytesIO MP3 or None."""
    if not HAS_TTS or not text.strip():
        return None
    try:
        buf = BytesIO()
        gTTS(text=text, lang="en", tld="co.uk").write_to_fp(buf)  # tweak tld for accent
        buf.seek(0)
        return buf
    except Exception:
        return None

def build_markdown_transcript():
    lines = ["# Chat Transcript\n"]
    for m in st.session_state.chat_history:
        who = "You" if m["role"] == "user" else "Assistant"
        lines.append(f"**{who} — {m.get('ts','')}**\n\n{m['content']}\n\n---\n")
    return "\n".join(lines)

# ---------- Sidebar: logo + Chat button + actions ----------
with st.sidebar:
    # Clickable logo at the very top (left corner)
    LOGO_PATH = "assets/logo.png"
    try:
        _logo_b64 = _load_logo_base64(LOGO_PATH)
        st.markdown(
            f"""
            <div style="display:flex; justify-content:center; margin:6px 0 8px;">
              <a href="?nav=home" title="Go to Home">
                <img src="data:image/png;base64,{_logo_b64}" style="height:70px; border-radius:8px; cursor:pointer;"
                     onclick="parent.window.location.search='?nav=home';" />
              </a>
            </div>
            """,
            unsafe_allow_html=True
        )
    except Exception:
        if st.button("🏠 Home", use_container_width=True):
            st.session_state.page = "Home"

    if st.button("Chat", use_container_width=True):
        st.session_state.page = "Chat"

    if st.button("Delete Chat", use_container_width=True):
        st.session_state.chat_history = []
        try:
            st.toast("Chat cleared.")
        except Exception:
            pass
        st.rerun()

    st.download_button(
        "Download Chat",
        data=build_markdown_transcript().encode("utf-8"),
        file_name="chat_transcript.md",
        mime="text/markdown",
        use_container_width=True,
    )

# ---------- Pages ----------
def render_home():
    st.markdown("<h2 style='text-align:center; margin-top:0;'>Welcome!</h2>", unsafe_allow_html=True)
    st.write("""
    This is your mental-health chatbot. Use the **Chat** page to ask questions.

    **What you can do here:**
    - Ask questions about anxiety, stress, mood, sleep, and coping strategies.
    - Get concise, supportive answers grounded in trusted sources.
    - Download your conversation anytime.
    """)

    st.info("Tip: Click **Delete Chat** in the sidebar to clear everything.")

    # --- UK Crisis Banner (under the tip) ---
    st.markdown(
        """
        <div style="
            background:#0b3b32;
            color:#fff;
            padding:14px 16px;
            border-radius:10px;
            margin: 8px 0 18px 0;
            box-shadow: 0 2px 6px rgba(0,0,0,.25);
            ">
          <div style="font-weight:700; font-size:16px; margin-bottom:6px;">
            If you’re in immediate danger, call <u>999</u> now.
          </div>
          <div style="line-height:1.55; font-size:15px;">
            • <strong>Samaritans</strong> (24/7): <a style="color:#fff;" href="tel:116123">116 123</a> |
              <a style="color:#fff;" href="https://www.samaritans.org/how-we-can-help/contact-samaritan/" target="_blank">Chat/Email</a><br>
            • <strong>SHOUT</strong> (24/7 text): Text <strong>SHOUT</strong> to 
              <a style="color:#fff;" href="sms:85258?&body=SHOUT">85258</a><br>
            • <strong>NHS 111</strong>: <a style="color:#fff;" href="tel:111">111</a> |
              <a style="color:#fff;" href="https://111.nhs.uk/" target="_blank">111.nhs.uk</a><br>
            • <strong>PAPYRUS HOPELINE247</strong> (suicide prevention): 
              <a style="color:#fff;" href="tel:+448000684141">0800 068 4141</a> |
              <a style="color:#fff;" href="https://www.papyrus-uk.org/papyrus-hopeline247/" target="_blank">Web</a><br>
            • <strong>CALM</strong> (Campaign Against Living Miserably): 
              <a style="color:#fff;" href="tel:+448005858585">0800 58 58 58</a> |
              <a style="color:#fff;" href="https://www.thecalmzone.net/get-support" target="_blank">Webchat</a><br>
            • <strong>Mind</strong> urgent help: 
              <a style="color:#fff;" href="tel:+443001233393">0300 123 3393</a> |
              <a style="color:#fff;" href="https://www.mind.org.uk/need-urgent-help/" target="_blank">mind.org.uk/need-urgent-help</a>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_chat():
    st.markdown("<h2 style='text-align:center; margin-top:0;'>Hi! What can I help you with today?</h2>",
                unsafe_allow_html=True)

    # Render history first
    for msg in st.session_state.chat_history:
        render_message(msg["role"], msg["content"], msg.get("audio"))

    # Input
    user_text = st.chat_input("Ask me anything about mental health...")
    if user_text:
        with st.chat_message("user"):
            st.markdown(f"<div class='msg user'><div class='bubble user'>{user_text}</div></div>",
                        unsafe_allow_html=True)
        add_message("user", user_text)

        with st.spinner("Thinking…"):
            try:
                result = call_rag(user_text, st.session_state.max_context)
                response = result["response"] if isinstance(result, dict) else str(result)
                sources = result.get("contexts", []) if isinstance(result, dict) else []
            except Exception as e:
                response = f"⚠️ Error: {e}"
                sources = []

        with st.chat_message("assistant"):
            streamed_markdown(response, delay=0.02)
            audio_buf = synthesize_tts(response)
            audio_bytes = audio_buf.getvalue() if audio_buf else None
            if audio_bytes:
                st.audio(audio_bytes, format="audio/mp3")
            if st.session_state.show_sources and sources:
                st.markdown(
                    "<div class='bubble source'><b>📚 Sources:</b><br>" +
                    "<br><br>".join(map(str, sources)) + "</div>",
                    unsafe_allow_html=True
                )
                st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        # Save assistant message with audio so it persists
        add_message("assistant", response, audio_bytes)

# ---------- Router ----------
if st.session_state.page == "Home":
    render_home()
else:
    render_chat()
