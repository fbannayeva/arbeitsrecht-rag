"""
Streamlit UI for Arbeitsrecht RAG — German Labor Law Assistant.
Run: streamlit run app.py
"""

import time
import uuid

import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Arbeitsrecht Assistent",
    page_icon="⚖️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=EB+Garamond:wght@400;600&family=DM+Sans:wght@400;500&display=swap');

    html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    }
    h1, h2, h3 {
        font-family: 'EB Garamond', serif;
}
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 1px solid #d4c9a8;
    }
    .answer-box {
        background: #faf8f3;
        border-left: 4px solid #8b6914;
        padding: 1.2rem 1.5rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
        font-size: 0.97rem;
        line-height: 1.7;
        color: #1a1a1a;
    }
    .disclaimer {
        background: #fff8e1;
        border: 1px solid #ffe082;
        border-radius: 8px;
        padding: 0.7rem 1rem;
        font-size: 0.82rem;
        color: #6d5e00;
        margin-top: 1rem;
    }
    .metric-card {
        background: white;
        border: 1px solid #e8e0cc;
        border-radius: 8px;
        padding: 0.8rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("# Arbeitsrecht Assistent")
    st.markdown("*KI-gestützter Assistent für deutsches Arbeitsrecht — KSchG · BGB · AGG · ArbZG · MuSchG · EntgFG*")

with col2:
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        if r.status_code == 200:
            st.success("API online", icon="✅")
        else:
            st.warning("API problem")
    except Exception:
        st.error("API offline")

st.divider()

# ---------------------------------------------------------------------------
# Example questions
# ---------------------------------------------------------------------------
EXAMPLE_QUESTIONS = [
    "Wie lange ist die gesetzliche Kündigungsfrist nach 5 Jahren?",
    "Was gilt beim Kündigungsschutz in der Probezeit?",
    "Wie viele Urlaubstage habe ich gesetzlich Anspruch?",
    "Was ist der Unterschied zwischen fristloser und ordentlicher Kündigung?",
    "Welche Rechte haben Schwangere am Arbeitsplatz?",
    "Wie lange wird Gehalt bei Krankheit weitergezahlt?",
]

st.markdown("### Beispielfragen")
cols = st.columns(3)
selected_example = None
for i, q in enumerate(EXAMPLE_QUESTIONS):
    with cols[i % 3]:
        if st.button(q, key=f"ex_{i}", use_container_width=True):
            selected_example = q

st.divider()

# ---------------------------------------------------------------------------
# Chat interface
# ---------------------------------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "history" not in st.session_state:
    st.session_state.history = []

question_input = st.text_input(
    "Ihre Frage zum Arbeitsrecht:",
    value=selected_example or "",
    placeholder="z.B. Wie lange ist die Kündigungsfrist nach 3 Jahren Betriebszugehörigkeit?",
    key="question_input",
)

col_ask, col_clear = st.columns([1, 5])
with col_ask:
    ask_clicked = st.button("Fragen ▶", type="primary", use_container_width=True)
with col_clear:
    if st.button("Verlauf löschen", use_container_width=False):
        st.session_state.history = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

# ---------------------------------------------------------------------------
# Query execution
# ---------------------------------------------------------------------------
if ask_clicked and question_input.strip():
    with st.spinner("Durchsuche Arbeitsrecht-Datenbank…"):
        t0 = time.monotonic()
        try:
            response = requests.post(
                f"{API_URL}/query",
                json={"question": question_input, "session_id": st.session_state.session_id},
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            st.session_state.history.insert(0, {
                "question": question_input,
                "answer": data["answer"],
                "latency_ms": data["latency_ms"],
                "trace_url": data.get("trace_url"),
                "disclaimer": data["disclaimer"],
            })
        except requests.exceptions.ConnectionError:
            st.error(" API nicht erreichbar. Starten Sie den Server: `uvicorn src.api.main:app --reload`")
        except Exception as e:
            st.error(f"Fehler: {e}")

# ---------------------------------------------------------------------------
# Display history
# ---------------------------------------------------------------------------
for entry in st.session_state.history:
    st.markdown(f"**❓ {entry['question']}**")

    st.markdown(f"""<div class="answer-box">{entry['answer'].replace(chr(10), '<br>')}</div>""",
                unsafe_allow_html=True)

    mcols = st.columns(3)
    with mcols[0]:
        st.metric("Antwortzeit", f"{entry['latency_ms']} ms")
    with mcols[1]:
        if entry.get("trace_url"):
            st.markdown(f"[ Langfuse Trace]({entry['trace_url']})")

    st.markdown(f"""<div class="disclaimer"> {entry['disclaimer']}</div>""",
                unsafe_allow_html=True)
    st.divider()

# ---------------------------------------------------------------------------
# Sidebar: indexed sources
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Indexierte Quellen")
    try:
        sources_r = requests.get(f"{API_URL}/sources", timeout=2)
        if sources_r.status_code == 200:
            for s in sources_r.json()["sources"]:
                st.markdown(f"**{s['name']}** — {s['full']}")
    except Exception:
        for name in ["KSchG", "BGB", "AGG", "ArbZG", "MuSchG", "EntgFG"]:
            st.markdown(f"• {name}")

    st.divider()
    st.markdown("### Über dieses Projekt")
    st.markdown("""
Portfolio-Projekt demonstriert **Agentic RAG** für juristische Dokumente:

-  Deutsche Rechtsdaten
- Qdrant Vektordatenbank
- CrossEncoder Reranking
- Langfuse Observability
- FastAPI Backend
    """)
