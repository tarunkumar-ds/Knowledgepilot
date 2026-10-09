import html as html_lib
import os
import re
import time
from pathlib import Path

import requests
import streamlit as st

from src.rag import (
    ingest_files,
    retrieve,
    answer_question,
    list_indexed_files,
)

# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

SUGGESTIONS = [
    "Summarize the main points",
    "Extract the key findings",
    "Explain the core concepts",
]

st.set_page_config(
    page_title="KnowledgePilot · Enterprise RAG",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# DESIGN SYSTEM (CSS)
# =========================================================

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --bg-main: #F5F7FB;
    --bg-surface: #FFFFFF;
    --bg-soft: #F8FAFC;
    --text-primary: #0F172A;
    --text-secondary: #334155;
    --text-muted: #64748B;
    --border-color: #E2E8F0;
    --border-strong: #CBD5E1;
    --accent: #4F46E5;
    --accent-hover: #4338CA;
    --accent-light: #EEF2FF;
    --accent-border: #C7D2FE;
    --success-bg: #ECFDF5;
    --success-text: #065F46;
    --success-border: #A7F3D0;
    --danger-bg: #FEF2F2;
    --danger-text: #991B1B;
    --danger-border: #FECACA;
    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 16px;
    --shadow-sm: 0 1px 2px rgba(15, 23, 42, 0.05);
    --shadow-md: 0 4px 12px -2px rgba(15, 23, 42, 0.08);
}

html, body, .stApp, .stApp * {
    font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}
/* keep icon fonts intact */
.stApp [data-testid="stIconMaterial"],
.stApp .material-icons,
.stApp [class*="material-symbols"] {
    font-family: "Material Symbols Rounded", "Material Icons" !important;
}

.stApp { background: var(--bg-main) !important; color: var(--text-primary) !important; }
[data-testid="stHeader"] { background: transparent !important; }
#MainMenu, footer { visibility: hidden; }

.block-container {
    max-width: 1500px;
    padding: 1.75rem 2rem 6rem !important;
}

/* ---------- Typography ---------- */
h1, h2, h3, h4 { color: var(--text-primary) !important; letter-spacing: -0.02em !important; }
.stApp p, .stApp li { color: var(--text-secondary); line-height: 1.65; font-size: 14.5px; }
.stApp [data-testid="stWidgetLabel"] p { color: var(--text-secondary) !important; font-size: 13px; font-weight: 500; }
.stApp [data-testid="stCaptionContainer"] { color: var(--text-muted) !important; }

/* ---------- Smooth entrance ---------- */
@keyframes fadeUp { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: none; } }
[data-testid="stChatMessage"], .card, .source-card, .page-header { animation: fadeUp .25s ease-out both; }

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background: var(--bg-surface) !important;
    border-right: 1px solid var(--border-color);
}
[data-testid="stSidebar"] > div:first-child { padding: 0.5rem 0.5rem; }

.brand { display: flex; align-items: center; gap: 12px; margin: 0.25rem 0 0.5rem; }
.brand-logo {
    width: 40px; height: 40px; border-radius: 10px;
    background: linear-gradient(135deg, #6366F1, #4338CA);
    color: #fff; display: flex; align-items: center; justify-content: center;
    font-size: 18px; box-shadow: 0 4px 10px rgba(79, 70, 229, .35);
}
.brand-title { font-size: 15.5px; font-weight: 700; color: var(--text-primary); line-height: 1.2; }
.brand-subtitle { font-size: 11.5px; color: var(--text-muted); }

.section-title {
    font-size: 11px; font-weight: 700; text-transform: uppercase;
    letter-spacing: .08em; color: var(--text-muted); margin: 1.4rem 0 .55rem;
}

/* ---------- Inputs (typing text always visible) ---------- */
.stApp input, .stApp textarea {
    color: var(--text-primary) !important;
    -webkit-text-fill-color: var(--text-primary) !important;
    caret-color: var(--accent) !important;
    opacity: 1 !important;
}
.stApp [data-baseweb="input"] > div,
.stApp [data-baseweb="select"] > div {
    background-color: var(--bg-surface) !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: var(--radius-sm) !important;
    box-shadow: var(--shadow-sm) !important;
    transition: border-color .15s, box-shadow .15s;
}
.stApp [data-baseweb="select"] * { color: var(--text-primary) !important; }
.stApp [data-baseweb="input"] > div:focus-within,
.stApp [data-baseweb="select"] > div:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(79, 70, 229, .15) !important;
}
[data-baseweb="popover"] ul[role="listbox"] li { color: var(--text-primary) !important; }

/* File uploader */
[data-testid="stFileUploaderDropzone"] {
    background: var(--bg-soft) !important;
    border: 1.5px dashed var(--accent-border) !important;
    border-radius: var(--radius-md) !important;
    transition: background .15s, border-color .15s;
}
[data-testid="stFileUploaderDropzone"]:hover { background: var(--accent-light) !important; border-color: var(--accent) !important; }
[data-testid="stFileUploaderDropzone"] * { color: var(--text-secondary) !important; }

/* ---------- Chat input ---------- */
[data-testid="stBottom"], [data-testid="stBottom"] > div,
[data-testid="stBottomBlockContainer"] { background: transparent !important; }

/* outer pill */
.stApp [data-testid="stChatInput"] {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: 16px !important;
    box-shadow: var(--shadow-md) !important;
    padding: 4px 6px !important;
    transition: border-color .15s, box-shadow .15s;
}
.stApp [data-testid="stChatInput"]:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(79, 70, 229, .15), var(--shadow-md) !important;
}
/* every inner layer: white, borderless, no theme-colored outline (fixes dark box) */
.stApp [data-testid="stChatInput"] > div,
.stApp [data-testid="stChatInput"] div[data-baseweb],
.stApp [data-testid="stChatInput"] div[data-baseweb] > div,
.stApp [data-testid="stChatInputTextArea"] {
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
}
/* the text you type */
.stApp [data-testid="stChatInput"] textarea,
.stApp [data-testid="stChatInputTextArea"] {
    background: transparent !important;
    color: #0F172A !important;
    -webkit-text-fill-color: #0F172A !important;
    caret-color: var(--accent) !important;
    font-size: 15px !important;
    opacity: 1 !important;
}
.stApp [data-testid="stChatInput"] textarea::placeholder,
.stApp [data-testid="stChatInputTextArea"]::placeholder {
    color: var(--text-muted) !important;
    -webkit-text-fill-color: var(--text-muted) !important;
    opacity: 1 !important;
}
/* send button: brand indigo instead of the default red */
.stApp [data-testid="stChatInputSubmitButton"],
.stApp [data-testid="stChatInput"] button {
    background: var(--accent) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    transition: background .15s, transform .15s;
}
.stApp [data-testid="stChatInputSubmitButton"] svg,
.stApp [data-testid="stChatInput"] button svg { fill: #fff !important; color: #fff !important; }
.stApp [data-testid="stChatInputSubmitButton"]:hover,
.stApp [data-testid="stChatInput"] button:hover { background: var(--accent-hover) !important; transform: translateY(-1px); }
.stApp [data-testid="stChatInputSubmitButton"]:disabled,
.stApp [data-testid="stChatInput"] button:disabled { background: #E2E8F0 !important; }
.stApp [data-testid="stChatInputSubmitButton"]:disabled svg,
.stApp [data-testid="stChatInput"] button:disabled svg { fill: #94A3B8 !important; color: #94A3B8 !important; }

/* ---------- Buttons (full width, no deprecated params needed) ---------- */
.stApp div.stButton > button {
    width: 100%;
    background: var(--bg-surface) !important;
    color: var(--text-secondary) !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: var(--radius-sm) !important;
    font-weight: 600 !important; font-size: 13px !important;
    min-height: 40px; box-shadow: var(--shadow-sm);
    transition: all .15s ease;
}
.stApp div.stButton > button:hover {
    background: var(--accent-light) !important; color: var(--accent) !important;
    border-color: var(--accent-border) !important; transform: translateY(-1px);
}
.stApp div.stButton > button:disabled { opacity: .5; transform: none; }
.stApp div.stButton > button[kind="primary"],
.stApp div.stButton > button[data-testid="stBaseButton-primary"] {
    background: var(--accent) !important; color: #fff !important; border-color: var(--accent) !important;
}
.stApp div.stButton > button[kind="primary"]:hover,
.stApp div.stButton > button[data-testid="stBaseButton-primary"]:hover {
    background: var(--accent-hover) !important; color: #fff !important;
}

/* ---------- Page header ---------- */
.page-header { display: flex; justify-content: space-between; align-items: flex-end; gap: 16px; flex-wrap: wrap; margin-bottom: 1.1rem; }
.page-title { font-size: 26px; font-weight: 700; color: var(--text-primary); letter-spacing: -.02em; margin: 0; }
.page-sub { font-size: 14px; color: var(--text-muted); margin: 2px 0 0; }
.chips { display: flex; gap: 8px; flex-wrap: wrap; }
.chip {
    display: inline-flex; align-items: center; gap: 6px;
    background: var(--bg-surface); border: 1px solid var(--border-color); color: var(--text-secondary);
    border-radius: 999px; padding: 4px 11px; font-size: 12px; font-weight: 600; box-shadow: var(--shadow-sm);
}
.chip.ok { background: var(--success-bg); color: var(--success-text); border-color: var(--success-border); }
.chip.bad { background: var(--danger-bg); color: var(--danger-text); border-color: var(--danger-border); }
.dot { width: 7px; height: 7px; border-radius: 50%; background: currentColor; }

/* ---------- Cards ---------- */
.card {
    background: var(--bg-surface); border: 1px solid var(--border-color);
    border-radius: var(--radius-lg); padding: 22px 24px; box-shadow: var(--shadow-sm); margin-bottom: 14px;
}
.card h4 { margin: 0 0 6px; font-size: 17px; }
.card p { margin: 0; }
.card.center { text-align: center; padding: 28px 18px; }
.card.center p { color: var(--text-muted); font-size: 13px; }
.card.warn { background: var(--danger-bg); border-color: var(--danger-border); }
.card.warn h4, .card.warn p { color: var(--danger-text) !important; }
.card code { background: rgba(15,23,42,.06); padding: 1px 6px; border-radius: 6px; font-size: 12.5px; }

.steps { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-top: 16px; }
.step { background: var(--bg-soft); border: 1px solid var(--border-color); border-radius: var(--radius-md); padding: 14px; }
.step-num {
    width: 24px; height: 24px; border-radius: 50%; background: var(--accent-light); color: var(--accent);
    font-weight: 700; font-size: 12px; display: inline-flex; align-items: center; justify-content: center; margin-bottom: 8px;
}
.step b { display: block; color: var(--text-primary); font-size: 13.5px; margin-bottom: 2px; }
.step span { color: var(--text-muted); font-size: 12.5px; }
@media (max-width: 800px) { .steps { grid-template-columns: 1fr; } }

/* ---------- Chat bubbles ---------- */
[data-testid="stChatMessage"] {
    background: var(--bg-surface); border: 1px solid var(--border-color);
    border-radius: var(--radius-lg); padding: 14px 18px; box-shadow: var(--shadow-sm); margin-bottom: 10px;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: var(--accent-light); border-color: var(--accent-border);
}

/* ---------- Sources ---------- */
.source-card {
    background: var(--bg-surface); border: 1px solid var(--border-color);
    border-radius: var(--radius-md); padding: 12px 14px; margin-bottom: 10px;
    transition: border-color .15s, box-shadow .15s;
}
.source-card:hover { border-color: var(--accent-border); box-shadow: var(--shadow-md); }
.source-header { display: flex; justify-content: space-between; align-items: center; gap: 8px; margin-bottom: 6px; }
.source-title { font-size: 12.5px; font-weight: 650; color: var(--text-primary); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.source-rank { color: var(--accent); margin-right: 6px; }
.source-meta { font-size: 11px; color: var(--text-muted); white-space: nowrap; background: var(--bg-soft); border-radius: 6px; padding: 2px 7px; }
.source-snippet { font-size: 12.5px; color: var(--text-secondary); line-height: 1.55; }

/* ---------- Sidebar doc list ---------- */
.doc-item {
    display: flex; align-items: center; gap: 8px; padding: 7px 10px; margin-bottom: 6px;
    background: var(--bg-soft); border: 1px solid var(--border-color); border-radius: var(--radius-sm);
    font-size: 13px; color: var(--text-primary);
}
.doc-item span:last-child { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
</style>
""",
    unsafe_allow_html=True,
)

# =========================================================
# SESSION STATE
# =========================================================

st.session_state.setdefault("messages", [])
st.session_state.setdefault("last_sources", [])
st.session_state.setdefault("pending_question", None)
st.session_state.setdefault("uploader_key", 0)
st.session_state.setdefault("flash", None)

# =========================================================
# HELPERS
# =========================================================


def html(markup: str):
    """Render HTML on a single line so Markdown never turns it into a code block."""
    st.markdown(re.sub(r"\s*\n\s*", " ", markup.strip()), unsafe_allow_html=True)


def esc(value) -> str:
    return html_lib.escape(str(value), quote=True)


@st.cache_data(ttl=10, show_spinner=False)
def check_ollama():
    """Return (is_running, installed_model_names). Cached so reruns stay fast."""
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        response.raise_for_status()
        models = [item.get("name", "") for item in response.json().get("models", [])]
        return True, models
    except (requests.RequestException, ValueError):
        return False, []


def model_is_installed(name: str, installed: list) -> bool:
    if not name:
        return False
    names = {m for m in installed} | {m.removesuffix(":latest") for m in installed}
    return name in names or name.removesuffix(":latest") in names


def snippet(text: str, limit: int = 340) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    return esc(text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "…")


def render_sources(sources):
    if not sources:
        st.caption("No sources available.")
        return
    for idx, source in enumerate(sources, start=1):
        distance = source.get("distance")
        meta = f'<span class="source-meta" title="Vector distance (lower = closer match)">d = {distance:.3f}</span>' if isinstance(distance, (int, float)) else ""
        html(
            f"""
            <div class="source-card">
                <div class="source-header">
                    <span class="source-title"><span class="source-rank">#{idx:02d}</span>{esc(source.get('file', 'Unknown'))}</span>
                    {meta}
                </div>
                <div class="source-snippet">{snippet(source.get('text', ''))}</div>
            </div>
            """
        )


def stream_text(text: str):
    """Yield the answer word by word for a smooth typing effect (capped at ~2.5s)."""
    tokens = re.findall(r"\S+\s*", text)
    delay = min(0.015, 2.5 / max(len(tokens), 1))
    for token in tokens:
        yield token
        time.sleep(delay)


def show_error(error):
    st.error("Something went wrong while processing your request.")
    with st.expander("Diagnostic details"):
        st.exception(error)


# =========================================================
# STATUS & SIDEBAR
# =========================================================

ollama_running, installed_models = check_ollama()

with st.sidebar:
    html(
        """
        <div class="brand">
            <div class="brand-logo">✦</div>
            <div>
                <div class="brand-title">KnowledgePilot</div>
                <div class="brand-subtitle">Enterprise RAG Engine</div>
            </div>
        </div>
        """
    )

    if st.session_state.flash:
        st.toast(st.session_state.flash, icon="✅")
        st.session_state.flash = None

    html('<div class="section-title">Knowledge Ingestion</div>')

    uploads = st.file_uploader(
        "Upload files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        key=f"document_upload_{st.session_state.uploader_key}",
        label_visibility="collapsed",
    )

    if st.button("Process & Index Documents", type="primary", disabled=not uploads):
        try:
            with st.spinner(f"Indexing {len(uploads)} document(s)…"):
                count = ingest_files(uploads, DATA_DIR)
            if count > 0:
                st.session_state.flash = f"Indexed {count} chunks successfully."
                st.session_state.uploader_key += 1  # clears the uploader
                st.rerun()
            else:
                st.warning("No parseable content detected in the selected files.")
        except Exception as error:
            show_error(error)

    indexed_files = list_indexed_files(DATA_DIR)

    html(f'<div class="section-title">Indexed Documents ({len(indexed_files)})</div>')
    if indexed_files:
        for filename in indexed_files:
            html(f'<div class="doc-item" title="{esc(filename)}"><span>📄</span><span>{esc(filename)}</span></div>')
    else:
        st.caption("No documents indexed yet.")

    html('<div class="section-title">Model Settings</div>')

    if installed_models:
        if st.session_state.get("ollama_model") not in installed_models:
            st.session_state.ollama_model = (
                DEFAULT_MODEL if DEFAULT_MODEL in installed_models else installed_models[0]
            )
        model = st.selectbox("Ollama model", installed_models, key="ollama_model")
        if not model_is_installed(DEFAULT_MODEL, installed_models):
            st.caption(f"Default `{DEFAULT_MODEL}` is not installed. Run `ollama pull {DEFAULT_MODEL}`.")
    else:
        st.session_state.setdefault("ollama_model", DEFAULT_MODEL)
        model = st.text_input("Ollama model", key="ollama_model").strip()

    top_k = st.slider(
        "Passages to retrieve",
        min_value=2,
        max_value=8,
        value=4,
        help="More passages give the model more context but can slow down answers.",
    )

    html('<div class="section-title">System Status</div>')
    if ollama_running:
        html('<span class="chip ok"><span class="dot"></span>Ollama online</span>')
    else:
        html('<span class="chip bad"><span class="dot"></span>Ollama offline</span>')
        st.write("")
        if st.button("Retry connection"):
            check_ollama.clear()
            st.rerun()

# =========================================================
# READINESS
# =========================================================

model_ok = (not installed_models) or model_is_installed(model, installed_models)
ready = bool(ollama_running and model and model_ok and indexed_files)

if not ollama_running:
    input_hint = "Ollama is offline. Start it to begin chatting."
elif not indexed_files:
    input_hint = "Upload and index a document to start asking questions."
elif not model_ok:
    input_hint = "Selected model is not installed."
else:
    input_hint = "Ask a question about your documents…"

# Chat input lives at the top level so it stays pinned to the bottom of the page.
typed_question = st.chat_input(input_hint, disabled=not ready)
pending = st.session_state.pending_question
question = typed_question or (pending if ready else None)
st.session_state.pending_question = None

# =========================================================
# MAIN WORKSPACE
# =========================================================

main_col, ref_col = st.columns([2.4, 1], gap="large")

with main_col:
    status_chip = (
        '<span class="chip ok"><span class="dot"></span>Online</span>'
        if ollama_running
        else '<span class="chip bad"><span class="dot"></span>Offline</span>'
    )
    queries_asked = sum(m["role"] == "user" for m in st.session_state.messages) + (1 if question else 0)
    html(
        f"""
        <div class="page-header">
            <div>
                <h1 class="page-title">Workspace</h1>
                <p class="page-sub">Ask questions about your private documents with context-aware retrieval.</p>
            </div>
            <div class="chips">
                {status_chip}
                <span class="chip">📄 {len(indexed_files)} document{'s' if len(indexed_files) != 1 else ''}</span>
                <span class="chip">🧠 {esc(model or 'No model')}</span>
                <span class="chip">💬 {queries_asked} quer{'y' if queries_asked == 1 else 'ies'}</span>
            </div>
        </div>
        """
    )

    # ---- Empty / blocked states ----
    if not ollama_running:
        html(
            f"""
            <div class="card warn">
                <h4>Ollama isn't reachable</h4>
                <p>Start the Ollama app or run <code>ollama serve</code>, then press <b>Retry connection</b> in the sidebar.
                Expected at <code>{esc(OLLAMA_URL)}</code>.</p>
            </div>
            """
        )
    elif not model_ok:
        html(
            f"""
            <div class="card warn">
                <h4>Model not installed</h4>
                <p>Run <code>ollama pull {esc(model)}</code> or pick an installed model in the sidebar.</p>
            </div>
            """
        )

    if not indexed_files:
        html(
            """
            <div class="card">
                <h4>Welcome to KnowledgePilot</h4>
                <p>Get answers from your own documents, fully local and private. Three quick steps:</p>
                <div class="steps">
                    <div class="step"><div class="step-num">1</div><b>Upload</b><span>Add PDF or TXT files in the sidebar.</span></div>
                    <div class="step"><div class="step-num">2</div><b>Index</b><span>Click “Process &amp; Index Documents”.</span></div>
                    <div class="step"><div class="step-num">3</div><b>Ask</b><span>Type a question and get cited answers.</span></div>
                </div>
            </div>
            """
        )
    elif not st.session_state.messages and not question:
        st.markdown("**Try asking**")
        cols = st.columns(len(SUGGESTIONS))
        for col, text in zip(cols, SUGGESTIONS):
            with col:
                if st.button(text, key=f"sugg_{text}", disabled=not ready):
                    st.session_state.pending_question = text
                    st.rerun()

    # ---- Conversation history ----
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander(f"Referenced passages ({len(message['sources'])})"):
                    render_sources(message["sources"])

    # ---- New question ----
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching documents and composing an answer…"):
                    passages = retrieve(question, DATA_DIR, top_k=top_k)
                    if passages:
                        response_text = answer_question(question, passages, model=model)
                        sources = passages
                    else:
                        response_text = (
                            "I couldn't find anything relevant in your indexed documents. "
                            "Try rephrasing your question or indexing more files."
                        )
                        sources = []

                st.write_stream(stream_text(response_text))
                if sources:
                    with st.expander(f"Referenced passages ({len(sources)})"):
                        render_sources(sources)

                st.session_state.messages.append(
                    {"role": "assistant", "content": response_text, "sources": sources}
                )
                st.session_state.last_sources = sources

            except Exception as error:
                # Keep the transcript consistent: every question gets a reply.
                fallback = "⚠️ I couldn't complete that request. Please check that Ollama is running and try again."
                st.markdown(fallback)
                show_error(error)
                st.session_state.messages.append(
                    {"role": "assistant", "content": fallback, "sources": []}
                )

# =========================================================
# REFERENCES PANEL
# =========================================================

with ref_col:
    latest_sources = st.session_state.last_sources
    html(
        f"""
        <h3 style="margin:0 0 2px;font-size:18px;">References</h3>
        <p style="margin:0 0 14px;font-size:13px;color:var(--text-muted);">
            {f'{len(latest_sources)} passage(s) used for the latest answer' if latest_sources else 'Passages used for the latest answer'}
        </p>
        """
    )

    if latest_sources:
        render_sources(latest_sources)
    else:
        html(
            """
            <div class="card center">
                <div style="font-size:26px;margin-bottom:6px;">🔎</div>
                <p>No sources yet.<br>Ask a question to see the matched passages here.</p>
            </div>
            """
        )

    if st.button("Clear conversation", disabled=not st.session_state.messages):
        st.session_state.messages = []
        st.session_state.last_sources = []
        st.session_state.pending_question = None
        st.rerun()