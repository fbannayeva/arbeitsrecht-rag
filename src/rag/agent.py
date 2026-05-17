"""
Agentic RAG pipeline for German labor law queries.

Architecture:
  Query → Dense retrieval (Qdrant) → CrossEncoder rerank
        → Llama 3.3 70B via Groq (answer with citations) → Langfuse trace

LLM: Meta Llama 3.3 70B via Groq (free tier, open-source)
Embeddings: paraphrase-multilingual-mpnet (local, no API key needed)
"""

import logging
import os
from typing import Any

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool
from langchain_groq import ChatGroq
from sentence_transformers import CrossEncoder

from src.ingestion.vectorstore import load_vector_store

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reranker (local, no API key)
# ---------------------------------------------------------------------------
_reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank(query: str, docs: list, top_k: int = 5) -> list:
    """Rerank retrieved documents by query relevance."""
    if not docs:
        return docs
    pairs = [(query, doc.page_content) for doc in docs]
    scores = _reranker.predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:top_k]]


# ---------------------------------------------------------------------------
# Retrieval tools
# ---------------------------------------------------------------------------
_vector_store = None


def get_vector_store():
    global _vector_store
    if _vector_store is None:
        _vector_store = load_vector_store()
    return _vector_store


@tool
def search_arbeitsrecht(query: str) -> str:
    """
    Search German labor law (Arbeitsrecht) documents.
    Returns relevant paragraphs with source citations.
    Use for questions about Kündigung, Urlaubsanspruch, Arbeitszeit,
    Mutterschutz, Diskriminierung, Entgeltfortzahlung, etc.
    """
    vs = get_vector_store()
    raw_docs = vs.similarity_search(query, k=10)
    reranked = rerank(query, raw_docs, top_k=5)

    results = []
    for doc in reranked:
        m = doc.metadata
        citation = f"[{m['source']} {m['paragraph']} – {m.get('title', '')}]"
        results.append(f"{citation}\n{doc.page_content}")

    return "\n\n---\n\n".join(results) if results else "Keine relevanten Paragraphen gefunden."


@tool
def explain_paragraph(paragraph_ref: str) -> str:
    """
    Retrieve and explain a specific law paragraph by reference.
    Example input: 'KSchG § 1' or 'AGG § 7'
    """
    vs = get_vector_store()
    query = f"Paragraph {paragraph_ref}"
    docs = vs.similarity_search(query, k=3)

    for doc in docs:
        m = doc.metadata
        ref = f"{m.get('source', '')} {m.get('paragraph', '')}".strip()
        if paragraph_ref.lower() in ref.lower():
            return f"**{ref} – {m.get('title', '')}**\n\n{doc.page_content}\n\nQuelle: {m.get('url', '')}"

    return f"Paragraph '{paragraph_ref}' nicht gefunden. Bitte prüfen Sie die Schreibweise."


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """Du bist ein präziser Legal-Tech-Assistent für deutsches Arbeitsrecht.

Du hilfst dabei, Fragen zu arbeitsrechtlichen Themen zu beantworten, indem du
die relevanten Gesetze (KSchG, BGB, AGG, ArbZG, MuSchG, EntgFG) durchsuchst.

Wichtige Regeln:
- Antworte immer auf Deutsch
- Zitiere immer die genauen Paragraphen (z.B. § 1 KSchG)
- Weise darauf hin, dass dies keine Rechtsberatung ist
- Bei komplexen Fällen empfehle einen Rechtsanwalt
- Strukturiere Antworten klar mit Absätzen

Beispiel-Themen: Kündigungsschutz, Urlaubsrecht, Arbeitszeit, Mutterschutz,
Diskriminierungsverbot, Lohnfortzahlung im Krankheitsfall."""


def build_agent() -> AgentExecutor:
    """Build LangChain tool-calling agent with Groq (Llama 3.1 70B) and Langfuse observability."""
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",  # open-source, free
        temperature=0,
        api_key=os.environ["GROQ_API_KEY"],
    )

    tools = [search_arbeitsrecht, explain_paragraph]

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=4)


def run_query(question: str, session_id: str = "default") -> dict[str, Any]:
    """
    Run a legal query through the RAG agent.
    Returns answer + Langfuse trace URL for observability.
    """
    callbacks = []
    trace_url = None

    # Langfuse tracing — optional, only if keys are set
    if os.environ.get("LANGFUSE_PUBLIC_KEY"):
        try:
            from langfuse.callback import CallbackHandler
            handler = CallbackHandler(session_id=session_id, user_id="arbeitsrecht-rag")
            callbacks.append(handler)
            trace_url = handler.get_trace_url()
        except Exception as e:
            logger.warning(f"Langfuse tracing unavailable: {e}")

    agent = build_agent()
    result = agent.invoke(
        {"input": question},
        config={"callbacks": callbacks},
    )

    return {
        "answer": result["output"],
        "question": question,
        "trace_url": trace_url,
    }


if __name__ == "__main__":
    result = run_query("Wie lange ist die gesetzliche Kündigungsfrist nach 5 Jahren Betriebszugehörigkeit?")
    print(result["answer"])