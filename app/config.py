"""Centralized configuration for the Customer Support FAQ RAG Chatbot."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

# Load .env file automatically if present
load_dotenv()


@dataclass(frozen=True)
class AppConfig:
    """Application configuration with validation and secret masking."""

    # LLM & Provider Settings
    groq_api_key: str = field(
        default_factory=lambda: os.getenv("GROQ_API_KEY", "").strip()
    )
    groq_base_url: str = field(
        default_factory=lambda: os.getenv(
            "GROQ_BASE_URL", "https://api.groq.com/openai/v1"
        ).strip()
    )
    llm_model: str = field(
        default_factory=lambda: os.getenv(
            "LLM_MODEL", "openai/gpt-oss-20b"
        ).strip()
    )
    temperature: float = field(
        default_factory=lambda: float(os.getenv("TEMPERATURE", "0.2"))
    )
    max_completion_tokens: int = field(
        default_factory=lambda: int(os.getenv("MAX_COMPLETION_TOKENS", "1024"))
    )
    llm_timeout_seconds: float = field(
        default_factory=lambda: float(os.getenv("LLM_TIMEOUT_SECONDS", "30.0"))
    )
    llm_max_retries: int = field(
        default_factory=lambda: int(os.getenv("LLM_MAX_RETRIES", "3"))
    )

    # Embedding & ChromaDB Settings
    embedding_model_name: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5"
        ).strip()
    )
    chroma_db_path: str = field(
        default_factory=lambda: os.getenv("CHROMA_DB_PATH", "./chroma_db").strip()
    )
    collection_name: str = field(
        default_factory=lambda: os.getenv(
            "COLLECTION_NAME", "customer_support_faqs"
        ).strip()
    )

    # Retrieval & Threshold Settings
    default_top_k: int = field(
        default_factory=lambda: int(os.getenv("DEFAULT_TOP_K", "3"))
    )
    similarity_threshold: float = field(
        default_factory=lambda: float(os.getenv("SIMILARITY_THRESHOLD", "0.75"))
    )
    max_query_length: int = field(
        default_factory=lambda: int(os.getenv("MAX_QUERY_LENGTH", "1000"))
    )

    # Fallback and Safety
    fallback_message: str = (
        "Sorry, I couldn't find anything about that in our FAQs. "
        "Please contact our support team for help."
    )

    # Logging and Privacy
    log_level: str = field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO").upper()
    )
    log_raw_queries: bool = field(
        default_factory=lambda: os.getenv("LOG_RAW_QUERIES", "false").lower()
        in {"true", "1", "yes"}
    )

    # Conversation Memory Settings
    max_history_turns: int = field(
        default_factory=lambda: int(os.getenv("MAX_HISTORY_TURNS", "5"))
    )
    max_active_conversations: int = field(
        default_factory=lambda: int(os.getenv("MAX_ACTIVE_CONVERSATIONS", "1000"))
    )
    conversation_ttl_seconds: float = field(
        default_factory=lambda: float(os.getenv("CONVERSATION_TTL_SECONDS", "3600.0"))
    )

    def validate(self, require_llm_key: bool = True) -> None:
        """Validate critical configuration settings.

        Raises:
            ValueError: If configuration values are invalid or out of bounds.
        """
        if require_llm_key and not self.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. Please set it in your environment or .env file."
            )

        if not self.collection_name:
            raise ValueError("COLLECTION_NAME cannot be empty.")

        if self.default_top_k <= 0:
            raise ValueError(f"DEFAULT_TOP_K must be positive, got {self.default_top_k}.")

        if not (0.0 <= self.similarity_threshold <= 1.0):
            raise ValueError(
                f"SIMILARITY_THRESHOLD must be between 0.0 and 1.0, got {self.similarity_threshold}."
            )

        if self.max_query_length <= 0:
            raise ValueError("MAX_QUERY_LENGTH must be positive.")

        if self.llm_timeout_seconds <= 0:
            raise ValueError("LLM_TIMEOUT_SECONDS must be positive.")

        if self.llm_max_retries < 0:
            raise ValueError("LLM_MAX_RETRIES must be non-negative.")

        if self.max_history_turns <= 0:
            raise ValueError("MAX_HISTORY_TURNS must be positive.")

        if self.max_active_conversations <= 0:
            raise ValueError("MAX_ACTIVE_CONVERSATIONS must be positive.")

        if self.conversation_ttl_seconds <= 0:
            raise ValueError("CONVERSATION_TTL_SECONDS must be positive.")

    def __repr__(self) -> str:
        """Mask secrets when stringified or logged."""
        masked_key = (
            f"{self.groq_api_key[:4]}...{self.groq_api_key[-4:]}"
            if len(self.groq_api_key) >= 8
            else ("***" if self.groq_api_key else "")
        )
        return (
            f"AppConfig(llm_model={self.llm_model!r}, "
            f"embedding_model={self.embedding_model_name!r}, "
            f"chroma_db_path={self.chroma_db_path!r}, "
            f"collection={self.collection_name!r}, "
            f"threshold={self.similarity_threshold}, "
            f"top_k={self.default_top_k}, "
            f"groq_api_key={masked_key!r})"
        )


def get_config(**overrides) -> AppConfig:
    """Helper to instantiate AppConfig with optional parameter overrides."""
    if not overrides:
        return AppConfig()
    return AppConfig(**overrides)

