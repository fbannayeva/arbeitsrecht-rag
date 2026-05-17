"""
Tests for Arbeitsrecht RAG pipeline.
"""

import pytest
from unittest.mock import MagicMock, patch

from src.ingestion.loader import clean_text, LegalDocument


# ---------------------------------------------------------------------------
# Ingestion tests
# ---------------------------------------------------------------------------

def test_clean_text_removes_extra_whitespace():
    raw = "Dies ist   ein Test\n\nmit    mehreren  Leerzeichen."
    result = clean_text(raw)
    assert "  " not in result


def test_clean_text_normalizes_paragraphs():
    raw = "(1) Erster Absatz. (2) Zweiter Absatz."
    result = clean_text(raw)
    assert "\n(2)" in result


def test_legal_document_creation():
    doc = LegalDocument(
        source="KSchG",
        paragraph="§ 1",
        title="Sozial ungerechtfertigte Kündigungen",
        text="Die Kündigung des Arbeitsverhältnisses gegenüber einem Arbeitnehmer...",
        url="https://www.gesetze-im-internet.de/kschg/",
    )
    assert doc.source == "KSchG"
    assert doc.doc_type == "Gesetz"


# ---------------------------------------------------------------------------
# API tests
# ---------------------------------------------------------------------------

def test_health_endpoint(test_client):
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_sources_endpoint(test_client):
    response = test_client.get("/sources")
    assert response.status_code == 200
    sources = response.json()["sources"]
    names = [s["name"] for s in sources]
    assert "KSchG" in names
    assert "AGG" in names


def test_query_validation(test_client):
    response = test_client.post("/query", json={"question": "Hi"})
    assert response.status_code == 422


@pytest.fixture
def test_client():
    from fastapi.testclient import TestClient
    from src.api.main import app
    return TestClient(app)


# ---------------------------------------------------------------------------
# RAG pipeline tests (mocked)
# ---------------------------------------------------------------------------

def test_rerank_returns_top_k():
    from src.rag.agent import rerank

    mock_docs = [MagicMock(page_content=f"Text {i}") for i in range(10)]

    with patch("src.rag.agent._reranker") as mock_reranker:
        mock_reranker.predict.return_value = [float(i) for i in range(10)]
        result = rerank("Kündigung", mock_docs, top_k=3)

    assert len(result) == 3


def test_query_response_has_disclaimer():
    """Verify disclaimer is always present in API responses."""
    from src.api.main import QueryResponse
    resp = QueryResponse(
        answer="Test Antwort",
        question="Test Frage",
        session_id="test-123",
        latency_ms=100,
    )
    assert "Rechtsberatung" in resp.disclaimer
