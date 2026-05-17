"""
FastAPI backend for Arbeitsrecht RAG Agent.
"""

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.rag.agent import run_query

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Arbeitsrecht RAG API…")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Arbeitsrecht RAG API",
    description="Agentic RAG over German labor law (KSchG, BGB, AGG, ArbZG, MuSchG)",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=1000,
                      examples=["Wie lange ist die Kündigungsfrist nach 3 Jahren?"])
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class QueryResponse(BaseModel):
    answer: str
    question: str
    session_id: str
    latency_ms: int
    trace_url: str | None = None
    disclaimer: str = (
        "Diese Antwort dient nur zur allgemeinen Information und stellt keine "
        "Rechtsberatung dar. Bei konkreten Rechtsfragen wenden Sie sich bitte "
        "an einen zugelassenen Rechtsanwalt."
    )


class HealthResponse(BaseModel):
    status: str
    version: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health():
    return {"status": "ok", "version": "1.0.0"}


@app.post("/query", response_model=QueryResponse)
async def query_law(request: QueryRequest):
    """
    Ask a question about German labor law.
    The agent retrieves relevant paragraphs and generates a cited answer.
    """
    logger.info(f"Query [{request.session_id}]: {request.question[:80]}")
    t0 = time.monotonic()

    try:
        result = run_query(request.question, session_id=request.session_id)
    except Exception as e:
        logger.error(f"Agent error: {e}")
        raise HTTPException(status_code=500, detail="Fehler bei der Verarbeitung der Anfrage.")

    latency = int((time.monotonic() - t0) * 1000)
    logger.info(f"Answered in {latency}ms")

    return QueryResponse(
        answer=result["answer"],
        question=request.question,
        session_id=request.session_id,
        latency_ms=latency,
        trace_url=result.get("trace_url"),
    )


@app.get("/sources")
async def list_sources():
    """List all indexed legal sources."""
    return {
        "sources": [
            {"name": "KSchG", "full": "Kündigungsschutzgesetz", "url": "https://www.gesetze-im-internet.de/kschg/"},
            {"name": "BGB", "full": "Bürgerliches Gesetzbuch (Arbeitsrecht §§ 611–630)", "url": "https://www.gesetze-im-internet.de/bgb/"},
            {"name": "AGG", "full": "Allgemeines Gleichbehandlungsgesetz", "url": "https://www.gesetze-im-internet.de/agg/"},
            {"name": "ArbZG", "full": "Arbeitszeitgesetz", "url": "https://www.gesetze-im-internet.de/arbzg/"},
            {"name": "MuSchG", "full": "Mutterschutzgesetz", "url": "https://www.gesetze-im-internet.de/muschg_2018/"},
            {"name": "EntgFG", "full": "Entgeltfortzahlungsgesetz", "url": "https://www.gesetze-im-internet.de/entgfg/"},
        ]
    }
