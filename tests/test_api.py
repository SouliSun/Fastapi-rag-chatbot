"""Unit and integration tests for the FastAPI service in api.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api import create_app
from config import AppConfig
from generation import GenerationResult
from retrieval import FAQRetriever, RetrievalDiagnostics, SearchResult


@pytest.fixture
def test_config():
    return AppConfig(
        groq_api_key="gsk-test-secret-key-12345",
        chroma_db_path="./test_chroma",
        collection_name="test_support_faqs",
        max_query_length=50,
        similarity_threshold=0.75,
        default_top_k=3,
        fallback_message="Default fallback for testing.",
    )


@pytest.fixture
def sample_search_result():
    return SearchResult(
        id="faq-101",
        question="How do I track my order?",
        answer="You can track your order using the order number sent to your email.",
        score=0.92,
        distance=0.08,
    )


@pytest.fixture
def mock_retriever(sample_search_result):
    retriever = MagicMock(spec=FAQRetriever)
    retriever.config = AppConfig(max_query_length=50, similarity_threshold=0.75, default_top_k=3)

    def search_fn(query, k=None, threshold=None):
        active_thresh = threshold if threshold is not None else 0.75
        active_k = k if k is not None else 3
        diag = RetrievalDiagnostics(
            query_length=len(query),
            requested_k=active_k,
            effective_k=1,
            total_candidates=1,
            accepted_candidates=1 if sample_search_result.score >= active_thresh else 0,
            top_score=sample_search_result.score,
            latency_ms=12.5,
            threshold_applied=active_thresh,
            is_accepted=sample_search_result.score >= active_thresh,
        )
        return [sample_search_result], diag

    retriever.search.side_effect = search_fn
    retriever.search_with_history.side_effect = lambda query, history=None, k=None, threshold=None: retriever.search(query, k=k, threshold=threshold)
    retriever.filter_relevant.side_effect = lambda results, thresh, max_items=None: [
        r for r in results if r.score >= thresh
    ][:max_items]
    return retriever


@pytest.fixture
def mock_generator(sample_search_result, test_config):
    generator = MagicMock()
    generator.config = test_config

    def generate_fn(query, relevant_faqs, history=None):
        if not relevant_faqs:
            return GenerationResult(
                answer=test_config.fallback_message,
                fallback=True,
                fallback_reason="No qualifying FAQs passed the similarity threshold.",
                retrieved_faqs=[],
                latency_ms=45.0,
            )
        return GenerationResult(
            answer="Here is the tracked shipment information.",
            fallback=False,
            fallback_reason=None,
            retrieved_faqs=relevant_faqs,
            latency_ms=150.0,
            prompt_tokens=42,
            completion_tokens=15,
        )

    generator.generate.side_effect = generate_fn
    return generator


@pytest.fixture
def mock_collection():
    coll = MagicMock()
    coll.count.return_value = 42
    return coll


@pytest.fixture
def client(test_config, mock_retriever, mock_generator, mock_collection):
    app = create_app(custom_config=test_config)
    app.state.retriever = mock_retriever
    app.state.generator = mock_generator
    app.state.collection = mock_collection
    with TestClient(app) as test_client:
        yield test_client


# ---------------------------------------------------------------------------
# Health Probe Tests
# ---------------------------------------------------------------------------

def test_health_endpoint_healthy(client, test_config):
    """GET /health returns 200 and accurate telemetry when components are operational."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["indexed_faqs"] == 42
    assert data["collection_name"] == test_config.collection_name
    assert data["embedding_model"] == test_config.embedding_model_name
    assert data["test_retrieval_ok"] is True
    assert data["error"] is None


def test_health_endpoint_unhealthy_on_error(test_config, mock_retriever, mock_generator):
    """GET /health returns 503 when the collection or retriever throws an error."""
    app = create_app(custom_config=test_config)
    failing_collection = MagicMock()
    failing_collection.count.side_effect = RuntimeError("ChromaDB connection lost")

    app.state.retriever = mock_retriever
    app.state.generator = mock_generator
    app.state.collection = failing_collection

    with TestClient(app) as test_client:
        resp = test_client.get("/health")
        assert resp.status_code == 503
        data = resp.json()
        assert data["status"] == "unhealthy"
        assert "failed" in data["error"].lower()


def test_health_endpoint_fallback_to_ingest_health_check(test_config):
    """GET /health falls back to ingest.health_check() if state singletons are absent."""
    app = create_app(custom_config=test_config)
    with patch("api.ensure_indexed", side_effect=RuntimeError("Index not ready")):
        with patch("api.health_check") as mock_hc:
            mock_hc.return_value = {
                "status": "healthy",
                "indexed_faqs": 100,
                "database_path": "./test_chroma",
                "collection_name": "test_support_faqs",
                "embedding_model": "BAAI/bge-small-en-v1.5",
                "test_retrieval_ok": True,
            }
            with TestClient(app) as test_client:
                resp = test_client.get("/health")
                assert resp.status_code == 200
                data = resp.json()
                assert data["status"] == "healthy"
                assert data["indexed_faqs"] == 100


# ---------------------------------------------------------------------------
# Chat Query Tests
# ---------------------------------------------------------------------------

def test_chat_endpoint_successful_answer(client):
    """POST /chat successfully returns an LLM-generated answer and telemetry."""
    payload = {"question": "How do I track my order?"}
    resp = client.post("/chat", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["answer"] == "Here is the tracked shipment information."
    assert data["fallback"] is False
    assert data["fallback_reason"] is None
    assert len(data["contexts"]) == 1
    assert data["contexts"][0]["id"] == "faq-101"
    assert data["contexts"][0]["score"] == 0.92
    assert data["prompt_tokens"] == 42
    assert data["completion_tokens"] == 15

    # Check retrieval diagnostics
    diag = data["diagnostics"]
    assert diag is not None
    assert diag["is_accepted"] is True
    assert diag["top_score"] == 0.92
    assert diag["accepted_candidates"] == 1


def test_chat_endpoint_fallback_on_low_similarity(client, test_config, mock_retriever):
    """POST /chat triggers fallback answer when candidates fail similarity threshold."""
    low_score_result = SearchResult(
        id="faq-999",
        question="Unrelated question",
        answer="Unrelated answer",
        score=0.45,
        distance=0.55,
    )
    mock_retriever.search.side_effect = lambda query, k=None, threshold=None: (
        [low_score_result],
        RetrievalDiagnostics(
            query_length=len(query),
            requested_k=k or 3,
            effective_k=1,
            total_candidates=1,
            accepted_candidates=0,
            top_score=0.45,
            latency_ms=10.0,
            threshold_applied=threshold or 0.75,
            is_accepted=False,
        ),
    )

    payload = {"question": "Do you sell lunar spaceships?"}
    resp = client.post("/chat", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["fallback"] is True
    assert data["answer"] == test_config.fallback_message
    assert "No qualifying FAQs passed" in data["fallback_reason"]
    assert len(data["contexts"]) == 0
    assert len(data["results"]) == 1


def test_chat_endpoint_blank_and_whitespace_questions(client):
    """POST /chat rejects empty and whitespace-only queries with HTTP 422."""
    for empty_input in ["", "   ", "\t\n  "]:
        resp = client.post("/chat", json={"question": empty_input})
        assert resp.status_code == 422
        assert "detail" in resp.json()


def test_chat_endpoint_too_long_query(client, test_config):
    """POST /chat rejects queries exceeding max_query_length with HTTP 400."""
    long_question = "A" * (test_config.max_query_length + 10)
    resp = client.post("/chat", json={"question": long_question})
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert "exceeds maximum allowed limit" in detail


def test_chat_endpoint_malformed_json_and_types(client):
    """POST /chat validates data types and returns HTTP 422 on invalid schema."""
    # Missing required question
    resp = client.post("/chat", json={"k": 5})
    assert resp.status_code == 422

    # Invalid k type
    resp = client.post("/chat", json={"question": "Valid query", "k": "invalid_int"})
    assert resp.status_code == 422

    # Negative k
    resp = client.post("/chat", json={"question": "Valid query", "k": -1})
    assert resp.status_code == 422

    # Threshold > 1.0
    resp = client.post("/chat", json={"question": "Valid query", "threshold": 1.5})
    assert resp.status_code == 422


def test_chat_endpoint_custom_k_and_threshold(client, mock_retriever):
    """POST /chat respects custom k and threshold parameters."""
    client.post("/chat", json={"question": "Check my order", "k": 5, "threshold": 0.88})

    call_args = mock_retriever.search.call_args
    assert call_args is not None
    args, kwargs = call_args
    assert args[0] == "Check my order"
    assert kwargs["k"] == 5
    assert kwargs["threshold"] == 0.88


# ---------------------------------------------------------------------------
# Error Boundaries & Secret Protection Tests
# ---------------------------------------------------------------------------

def test_retrieval_failure_recovers_gracefully(client, test_config, mock_retriever):
    """Service returns safe fallback without crashing if retrieval raises an exception."""
    mock_retriever.search.side_effect = RuntimeError("ChromaDB index corrupted")

    resp = client.post("/chat", json={"question": "What is the warranty period?"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["fallback"] is True
    assert data["answer"] == test_config.fallback_message
    assert "Retrieval service encountered an unexpected error." in data["fallback_reason"]


def test_generation_failure_recovers_gracefully(client, test_config, mock_generator):
    """Service returns safe fallback without crashing if generation raises an exception."""
    mock_generator.generate.side_effect = RuntimeError("OpenAI provider timeout")

    resp = client.post("/chat", json={"question": "What is the warranty period?"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["fallback"] is True
    assert data["answer"] == test_config.fallback_message
    assert "Generation service encountered an unexpected error." in data["fallback_reason"]


def test_secret_is_never_leaked(client, test_config, mock_generator):
    """Ensure the API key is never exposed in responses or errors."""
    mock_generator.generate.side_effect = Exception(
        f"API Error with key {test_config.groq_api_key}"
    )

    resp = client.post("/chat", json={"question": "Valid query"})
    raw_response_text = resp.text
    assert test_config.groq_api_key not in raw_response_text

    # Also check /health response
    health_resp = client.get("/health")
    assert test_config.groq_api_key not in health_resp.text


def test_lifespan_startup_initializes_singletons(test_config):
    """Verify lifespan context initializes collection, retriever, and generator."""
    app = create_app(custom_config=test_config)
    mock_coll = MagicMock()
    with patch("api.ensure_indexed", return_value=mock_coll) as mock_ei:
        with patch("api.AnswerGenerator") as mock_gen_cls:
            with TestClient(app):
                assert app.state.collection is mock_coll
                assert app.state.retriever is not None
                assert app.state.generator is not None
            mock_ei.assert_called_once()
            mock_gen_cls.assert_called_once_with(test_config)


def test_cors_configuration(client):
    """Verify CORS headers are returned for allowed local dev origins and rejected for untrusted origins."""
    # Allowed local Vite origin
    resp = client.options(
        "/chat",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:5173"
    assert resp.headers.get("access-control-allow-credentials") is None

    # Untrusted origin
    resp_untrusted = client.options(
        "/chat",
        headers={
            "Origin": "http://evil-tracker.example.com",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert resp_untrusted.headers.get("access-control-allow-origin") is None


def test_chat_generates_conversation_id_when_omitted(client):
    """Verify that omitting conversation_id generates a valid conversation_id."""
    resp = client.post("/chat", json={"question": "Where is my receipt?"})
    assert resp.status_code == 200
    data = resp.json()
    assert "conversation_id" in data
    assert data["conversation_id"].startswith("conv_")


def test_chat_preserves_valid_conversation_id(client):
    """Verify that providing a valid conversation_id preserves it in the response."""
    conv_id = "user-session-42"
    resp = client.post("/chat", json={"question": "Where is my receipt?", "conversation_id": conv_id})
    assert resp.status_code == 200
    data = resp.json()
    assert data["conversation_id"] == conv_id


def test_chat_rejects_invalid_conversation_id(client):
    """Verify that an invalid conversation_id format returns HTTP 400."""
    resp = client.post(
        "/chat",
        json={"question": "Where is my receipt?", "conversation_id": "invalid/id!@#"},
    )
    assert resp.status_code == 400
    assert "Invalid conversation_id" in resp.json()["detail"]

    # Also test exceeding maximum length (64 chars)
    resp_long = client.post(
        "/chat",
        json={"question": "Where is my receipt?", "conversation_id": "a" * 65},
    )
    assert resp_long.status_code == 400
    assert "exceeds maximum allowed length" in resp_long.json()["detail"]


def test_chat_passes_history_to_retriever_and_generator_on_followup(
    client, mock_retriever, mock_generator
):
    """Verify multi-turn flow passes accumulated conversation history to retriever and generator."""
    conv_id = "conv_multiturn_1"

    # Turn 1
    resp1 = client.post(
        "/chat",
        json={"question": "How do I track my order?", "conversation_id": conv_id},
    )
    assert resp1.status_code == 200

    # Turn 2: Follow-up question
    resp2 = client.post(
        "/chat",
        json={"question": "Can I do that on mobile?", "conversation_id": conv_id},
    )
    assert resp2.status_code == 200

    # Verify generator.generate was called on turn 2 with history
    last_gen_call = mock_generator.generate.call_args
    assert last_gen_call is not None
    _, kwargs = last_gen_call
    # In kwargs or args: generate(body.question, relevant, history=history)
    history_passed = kwargs.get("history") or (
        last_gen_call[0][2] if len(last_gen_call[0]) > 2 else None
    )
    assert history_passed is not None
    assert len(history_passed) == 1
    assert history_passed[0].user_query == "How do I track my order?"


def test_chat_conversation_isolation(client, mock_generator):
    """Verify distinct conversation IDs maintain isolated histories."""
    # Conversation A turn 1
    client.post(
        "/chat",
        json={"question": "What is your return policy?", "conversation_id": "conv_user_A"},
    )

    # Conversation B turn 1
    client.post(
        "/chat",
        json={"question": "Do you ship internationally?", "conversation_id": "conv_user_B"},
    )

    # Check that Conv B's call only received empty history (its own first turn)
    last_gen_call = mock_generator.generate.call_args
    history_passed = last_gen_call.kwargs.get("history") or (
        last_gen_call.args[2] if len(last_gen_call.args) > 2 else None
    )
    assert history_passed is None or len(history_passed) == 0


def test_chat_fallback_still_triggered_without_faqs_despite_history(
    client, mock_retriever, test_config
):
    """Ensure that history is not used as a substitute for factual retrieval."""
    conv_id = "conv_no_rag_substitute"

    # Turn 1 succeeds with FAQ
    client.post(
        "/chat",
        json={"question": "How do I track my order?", "conversation_id": conv_id},
    )

    # Turn 2: Mock retrieval returning no relevant results (score below threshold)
    low_score_result = SearchResult(
        id="faq-low",
        question="Unrelated",
        answer="Unrelated",
        score=0.20,
        distance=0.80,
    )
    mock_retriever.search.side_effect = lambda query, k=None, threshold=None: (
        [low_score_result],
        RetrievalDiagnostics(
            query_length=len(query),
            requested_k=k or 3,
            effective_k=1,
            total_candidates=1,
            accepted_candidates=0,
            top_score=0.20,
            latency_ms=10.0,
            threshold_applied=threshold or 0.75,
            is_accepted=False,
        ),
    )

    resp2 = client.post(
        "/chat",
        json={"question": "What is the secret recipe?", "conversation_id": conv_id},
    )
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert data2["fallback"] is True
    assert data2["answer"] == test_config.fallback_message



