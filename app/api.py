"""FastAPI service for the Customer Support FAQ RAG Chatbot.

Provides HTTP endpoints for health probes and conversational question-answering
backed by ChromaDB semantic search and OpenAI-compatible Groq LLM generation.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, List, Optional

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

try:
    from app.config import AppConfig, get_config
    from app.generation import AnswerGenerator, GenerationResult
    from app.ingest import ensure_indexed, get_default_generator, get_default_retriever, health_check
    from app.memory import ConversationStore, ConversationTurn
    from app.retrieval import FAQRetriever, RetrievalDiagnostics, SearchResult
except ImportError:
    from config import AppConfig, get_config
    from generation import AnswerGenerator, GenerationResult
    from ingest import ensure_indexed, get_default_generator, get_default_retriever, health_check
    from memory import ConversationStore, ConversationTurn
    from retrieval import FAQRetriever, RetrievalDiagnostics, SearchResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    """Incoming chat request payload."""

    question: str = Field(
        ...,
        description="The customer inquiry or support question.",
        examples=["How do I track my order?"],
    )
    conversation_id: Optional[str] = Field(
        default=None,
        description="Optional conversation identifier to maintain multi-turn chat context.",
        examples=["conv_3f8b1c4a9d2e"],
    )
    k: Optional[int] = Field(
        default=None,
        ge=1,
        le=20,
        description="Optional number of FAQ candidates to retrieve. Defaults to default_top_k.",
    )
    threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional similarity threshold for accepting candidates. Defaults to similarity_threshold.",
    )

    @field_validator("question")
    @classmethod
    def validate_question_not_blank(cls, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("Question must be a string.")
        cleaned = " ".join(value.strip().split())
        if not cleaned:
            raise ValueError("Question cannot be empty or whitespace-only.")
        return cleaned


class RetrievedFAQ(BaseModel):
    """Serialized representation of a retrieved FAQ document."""

    id: str = Field(..., description="Unique FAQ identifier.")
    question: str = Field(..., description="Indexed FAQ question.")
    answer: str = Field(..., description="Indexed FAQ answer.")
    score: float = Field(..., description="Calculated cosine similarity score.")
    distance: float = Field(..., description="Raw vector distance from ChromaDB.")


class RetrievalDiagnosticsModel(BaseModel):
    """Structured search telemetry for RAG observability."""

    query_length: int = Field(..., description="Length of the normalized user query.")
    requested_k: int = Field(..., description="Requested top-k parameter.")
    effective_k: int = Field(..., description="Actual k evaluated against available corpus.")
    total_candidates: int = Field(..., description="Total candidates returned by vector search.")
    accepted_candidates: int = Field(..., description="Number of candidates passing similarity threshold.")
    top_score: Optional[float] = Field(default=None, description="Similarity score of the top candidate.")
    latency_ms: float = Field(..., description="Vector retrieval latency in milliseconds.")
    threshold_applied: float = Field(..., description="Similarity threshold applied.")
    is_accepted: bool = Field(..., description="Whether top candidate met or exceeded the threshold.")
    error: Optional[str] = Field(default=None, description="Error message if retrieval encountered issues.")


class ChatResponse(BaseModel):
    """Outgoing response for a chat query."""

    conversation_id: str = Field(
        ...,
        description="The stable conversation identifier maintaining multi-turn context.",
    )
    answer: str = Field(..., description="Generated answer or polite fallback message.")
    fallback: bool = Field(..., description="True if no context passed threshold or generation failed.")
    fallback_reason: Optional[str] = Field(default=None, description="Detailed explanation for fallback status.")
    contexts: List[RetrievedFAQ] = Field(
        default_factory=list,
        description="Filtered FAQ items provided as verified context to the LLM.",
    )
    results: List[RetrievedFAQ] = Field(
        default_factory=list,
        description="All top-k retrieved FAQ candidates before filtering.",
    )
    latency_ms: float = Field(..., description="Total execution latency in milliseconds.")
    prompt_tokens: Optional[int] = Field(default=None, description="Input prompt tokens consumed if available.")
    completion_tokens: Optional[int] = Field(default=None, description="Completion tokens consumed if available.")
    diagnostics: Optional[RetrievalDiagnosticsModel] = Field(
        default=None,
        description="Retrieval performance diagnostics.",
    )


class HealthResponse(BaseModel):
    """Service health and RAG readiness telemetry."""

    status: str = Field(..., description="'healthy' or 'unhealthy'.")
    indexed_faqs: Optional[int] = Field(default=None, description="Number of indexed FAQs in ChromaDB.")
    database_path: Optional[str] = Field(default=None, description="Configured ChromaDB storage path.")
    collection_name: Optional[str] = Field(default=None, description="Target collection name.")
    embedding_model: Optional[str] = Field(default=None, description="Embedding model name.")
    test_retrieval_ok: Optional[bool] = Field(default=None, description="Whether test retrieval succeeded.")
    error: Optional[str] = Field(default=None, description="Error detail if status is unhealthy.")


# ---------------------------------------------------------------------------
# State Helpers
# ---------------------------------------------------------------------------

def get_app_config(app: FastAPI) -> AppConfig:
    if getattr(app.state, "config", None) is not None:
        return app.state.config
    return get_config()


def get_app_retriever(app: FastAPI) -> FAQRetriever:
    if getattr(app.state, "retriever", None) is not None:
        return app.state.retriever
    return get_default_retriever()


def get_app_generator(app: FastAPI) -> AnswerGenerator:
    if getattr(app.state, "generator", None) is not None:
        return app.state.generator
    return get_default_generator()


_DEFAULT_MEMORY_STORE: Optional[ConversationStore] = None


def get_app_memory_store(app: FastAPI) -> ConversationStore:
    global _DEFAULT_MEMORY_STORE
    if getattr(app.state, "memory_store", None) is not None:
        return app.state.memory_store
    if _DEFAULT_MEMORY_STORE is None:
        cfg = get_app_config(app)
        _DEFAULT_MEMORY_STORE = ConversationStore(
            max_conversations=cfg.max_active_conversations,
            max_turns_per_conversation=cfg.max_history_turns,
            ttl_seconds=cfg.conversation_ttl_seconds,
        )
    return _DEFAULT_MEMORY_STORE


# ---------------------------------------------------------------------------
# Lifespan Management
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize ChromaDB index and RAG singletons once on application startup."""
    logger.info("Initializing FAQ RAG FastAPI service...")
    cfg = getattr(app.state, "config", None) or get_config()
    app.state.config = cfg

    # Only initialize if not already provided (e.g., via test fixtures or custom injection)
    if getattr(app.state, "collection", None) is None or getattr(app.state, "retriever", None) is None:
        try:
            coll = ensure_indexed()
            app.state.collection = coll
            app.state.retriever = FAQRetriever(coll, cfg)
            logger.info("ChromaDB collection and FAQRetriever initialized successfully.")
        except Exception as exc:
            logger.error("Failed to initialize ChromaDB or FAQRetriever: %s", exc)
            app.state.collection = None
            app.state.retriever = None

    if getattr(app.state, "generator", None) is None:
        try:
            app.state.generator = AnswerGenerator(cfg)
            logger.info("AnswerGenerator initialized successfully.")
        except Exception as exc:
            logger.error("Failed to initialize AnswerGenerator: %s", exc)
            app.state.generator = None

    if getattr(app.state, "memory_store", None) is None:
        app.state.memory_store = ConversationStore(
            max_conversations=cfg.max_active_conversations,
            max_turns_per_conversation=cfg.max_history_turns,
            ttl_seconds=cfg.conversation_ttl_seconds,
        )
        logger.info("ConversationStore initialized successfully.")

    yield

    logger.info("Shutting down FAQ RAG FastAPI service.")


# ---------------------------------------------------------------------------
# Application Factory
# ---------------------------------------------------------------------------

def create_app(custom_config: Optional[AppConfig] = None) -> FastAPI:
    """Create and configure the FastAPI application instance."""
    app = FastAPI(
        title="Customer Support FAQ RAG API",
        description=(
            "Retrieval-Augmented Generation (RAG) service for customer support FAQs. "
            "Note: This service does not include authentication or rate limiting by default. "
            "Deploy behind an API gateway or reverse proxy for production exposure."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    if custom_config is not None:
        app.state.config = custom_config

    # Configure CORS for local development origins (avoiding wildcard CORS and credentials)
    cors_origins = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    # Safe error handlers preventing stack trace or credential leakage
    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error("Unhandled internal server exception: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "An internal server error occurred. Please contact support."},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        # Return clean validation error messages without leaking internal structures
        errors = [
            {"loc": err.get("loc"), "msg": err.get("msg"), "type": err.get("type")}
            for err in exc.errors()
        ]
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": errors},
        )

    # -----------------------------------------------------------------------
    # Endpoints
    # -----------------------------------------------------------------------

    @app.get(
        "/health",
        response_model=HealthResponse,
        responses={
            200: {"description": "RAG service and ChromaDB are healthy."},
            503: {"description": "Service is unhealthy or index is unavailable."},
        },
        summary="Service and RAG readiness health check",
    )
    def health_endpoint(request: Request, response: Response) -> HealthResponse:
        """Probe service readiness and verify ChromaDB connectivity."""
        cfg = get_app_config(request.app)
        coll = getattr(request.app.state, "collection", None)
        retriever = getattr(request.app.state, "retriever", None)

        # Fall back to ingest.health_check() if state singletons are absent
        if coll is None or retriever is None:
            diag = health_check()
            if diag.get("status") != "healthy":
                response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return HealthResponse(
                status=diag.get("status", "unhealthy"),
                indexed_faqs=diag.get("indexed_faqs"),
                database_path=diag.get("database_path", cfg.chroma_db_path),
                collection_name=diag.get("collection_name", cfg.collection_name),
                embedding_model=diag.get("embedding_model", cfg.embedding_model_name),
                test_retrieval_ok=diag.get("test_retrieval_ok", False),
                error=diag.get("error"),
            )

        try:
            count = coll.count()
            # Perform a lightweight search test to verify embeddings and indexing
            sample_results, _ = retriever.search("health check query", k=1)
            return HealthResponse(
                status="healthy",
                indexed_faqs=count,
                database_path=cfg.chroma_db_path,
                collection_name=cfg.collection_name,
                embedding_model=cfg.embedding_model_name,
                test_retrieval_ok=True,
            )
        except Exception as exc:
            logger.error("Health check execution failed: %s", exc)
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return HealthResponse(
                status="unhealthy",
                indexed_faqs=None,
                database_path=cfg.chroma_db_path,
                collection_name=cfg.collection_name,
                embedding_model=cfg.embedding_model_name,
                test_retrieval_ok=False,
                error="RAG retrieval or database check failed.",
            )

    @app.post(
        "/chat",
        response_model=ChatResponse,
        responses={
            200: {"description": "Query processed successfully."},
            400: {"description": "Query rejected (e.g., query length exceeds maximum)."},
            422: {"description": "Validation error (e.g., blank or whitespace question)."},
            500: {"description": "Internal server error."},
        },
        summary="Process a customer inquiry and return an answer",
    )
    def chat_endpoint(request: Request, body: ChatRequest) -> ChatResponse:
        """Receive a customer question, retrieve top FAQ candidates, and generate an answer."""
        cfg = get_app_config(request.app)

        # Enforce maximum query length constraint with clear HTTP 400
        if len(body.question) > cfg.max_query_length:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Query length ({len(body.question)}) exceeds maximum allowed limit "
                    f"of {cfg.max_query_length} characters."
                ),
            )

        # 0. Conversation session management
        memory_store = get_app_memory_store(request.app)
        try:
            session, conv_id = memory_store.get_or_create(body.conversation_id)
            history = session.get_history()
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            )

        # Privacy-conscious logging: do not log raw queries unless explicitly enabled
        if cfg.log_raw_queries:
            logger.info(
                "Chat request: conv_id=%s, query='%s', k=%s, threshold=%s, history_turns=%d",
                conv_id,
                body.question,
                body.k,
                body.threshold,
                len(history),
            )
        else:
            logger.info(
                "Chat request: conv_id=%s, query_length=%d, k=%s, threshold=%s, history_turns=%d",
                conv_id,
                len(body.question),
                body.k,
                body.threshold,
                len(history),
            )

        retriever = get_app_retriever(request.app)
        generator = get_app_generator(request.app)

        active_threshold = (
            body.threshold if body.threshold is not None else cfg.similarity_threshold
        )
        active_k = body.k if body.k is not None else cfg.default_top_k

        # 1. Retrieval with multi-turn history resolution and failure boundary
        try:
            if history and hasattr(retriever, "search_with_history"):
                results, diagnostics = retriever.search_with_history(
                    body.question, history=history, k=active_k, threshold=active_threshold
                )
            else:
                results, diagnostics = retriever.search(
                    body.question, k=active_k, threshold=active_threshold
                )
        except Exception as exc:
            logger.error("Retrieval failed in /chat: %s", exc)
            return ChatResponse(
                conversation_id=conv_id,
                answer=cfg.fallback_message,
                fallback=True,
                fallback_reason="Retrieval service encountered an unexpected error.",
                contexts=[],
                results=[],
                latency_ms=0.0,
                diagnostics=None,
            )

        relevant = retriever.filter_relevant(results, active_threshold, max_items=active_k)

        # 2. Generation with multi-turn context and failure boundary
        try:
            gen_res = generator.generate(body.question, relevant, history=history)
        except Exception as exc:
            logger.error("Generation failed in /chat: %s", exc)
            diag_model = (
                RetrievalDiagnosticsModel(**diagnostics.__dict__) if diagnostics else None
            )
            return ChatResponse(
                conversation_id=conv_id,
                answer=cfg.fallback_message,
                fallback=True,
                fallback_reason="Generation service encountered an unexpected error.",
                contexts=[RetrievedFAQ(**r.to_dict()) for r in relevant],
                results=[RetrievedFAQ(**r.to_dict()) for r in results],
                latency_ms=0.0,
                diagnostics=diag_model,
            )

        # 3. Record interaction turn into session memory
        memory_store.record_turn(
            conv_id,
            user_query=body.question,
            assistant_answer=gen_res.answer,
            fallback=gen_res.fallback,
        )

        diag_model = (
            RetrievalDiagnosticsModel(**diagnostics.__dict__) if diagnostics else None
        )

        return ChatResponse(
            conversation_id=conv_id,
            answer=gen_res.answer,
            fallback=gen_res.fallback,
            fallback_reason=gen_res.fallback_reason,
            contexts=[RetrievedFAQ(**r.to_dict()) for r in relevant],
            results=[RetrievedFAQ(**r.to_dict()) for r in results],
            latency_ms=gen_res.latency_ms,
            prompt_tokens=gen_res.prompt_tokens,
            completion_tokens=gen_res.completion_tokens,
            diagnostics=diag_model,
        )

    return app


# Module-level default application instance for Uvicorn: `uvicorn api:app`
app = create_app()
