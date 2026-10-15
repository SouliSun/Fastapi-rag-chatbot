"""FAQ RAG Assistant core package."""

from app.config import AppConfig, get_config
from app.memory import ConversationStore, ConversationSession, ConversationTurn
from app.retrieval import FAQRetriever, SearchResult, RetrievalDiagnostics
from app.generation import AnswerGenerator, GenerationResult
from app.api import create_app, app

__all__ = [
    "AppConfig",
    "get_config",
    "ConversationStore",
    "ConversationSession",
    "ConversationTurn",
    "FAQRetriever",
    "SearchResult",
    "RetrievalDiagnostics",
    "AnswerGenerator",
    "GenerationResult",
    "create_app",
    "app",
]

