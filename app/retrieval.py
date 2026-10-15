"""Retrieval pipeline and result modeling for the Customer Support FAQ RAG Chatbot."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import chromadb

try:
    from app.config import AppConfig, get_config
except ImportError:
    from config import AppConfig, get_config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchResult:
    """Standardized representation of a single retrieved FAQ record."""
    id: str
    question: str
    answer: str
    score: float
    distance: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "answer": self.answer,
            "score": self.score,
            "distance": self.distance,
        }


@dataclass(frozen=True)
class RetrievalDiagnostics:
    """Structured telemetry captured during a search operation."""
    query_length: int
    requested_k: int
    effective_k: int
    total_candidates: int
    accepted_candidates: int
    top_score: Optional[float]
    latency_ms: float
    threshold_applied: float
    is_accepted: bool
    error: Optional[str] = None


class FAQRetriever:
    """Manages semantic search, ChromaDB queries, and threshold evaluation."""

    def __init__(self, collection: chromadb.Collection, config: Optional[AppConfig] = None):
        self.collection = collection
        self.config = config or get_config()

    def validate_query(self, query: Any) -> str:
        """Validate input query and return normalized string.

        Raises:
            ValueError: If query is empty, non-string, or exceeds max_query_length.
        """
        if not isinstance(query, str):
            raise ValueError(f"Query must be a string, got {type(query).__name__}")

        cleaned = " ".join(query.strip().split())
        if not cleaned:
            raise ValueError("Query cannot be empty or whitespace-only.")

        if len(cleaned) > self.config.max_query_length:
            raise ValueError(
                f"Query length ({len(cleaned)}) exceeds maximum allowed "
                f"({self.config.max_query_length})."
            )

        return cleaned

    def validate_k(self, k: int, collection_count: int) -> int:
        """Validate and clamp k between 1 and available collection size."""
        if not isinstance(k, int) or k <= 0:
            raise ValueError(f"k must be a positive integer, got {k}")

        if collection_count == 0:
            return 0

        return min(k, collection_count)

    def search(
        self,
        query: str,
        k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> Tuple[List[SearchResult], RetrievalDiagnostics]:
        """Perform semantic retrieval against the ChromaDB FAQ collection.

        Distance conversion:
            ChromaDB configured with hnsw:space='cosine' returns cosine distance:
                distance = 1.0 - cos_sim(u, v)
            We convert this to cosine similarity via:
                similarity = 1.0 - distance
            Similarity ranges from -1.0 to 1.0 (with 1.0 being an exact match).

        Args:
            query: The user question text.
            k: Number of candidates to retrieve. Defaults to config.default_top_k.
            threshold: Minimum similarity to accept top result. Defaults to config.similarity_threshold.

        Returns:
            Tuple of (list_of_all_retrieved_results, retrieval_diagnostics)
        """
        start_time = time.perf_counter()
        active_threshold = (
            threshold if threshold is not None else self.config.similarity_threshold
        )
        requested_k = k if k is not None else self.config.default_top_k

        # 1. Validate query
        try:
            norm_query = self.validate_query(query)
        except ValueError as exc:
            elapsed = (time.perf_counter() - start_time) * 1000.0
            return [], RetrievalDiagnostics(
                query_length=len(str(query or "")),
                requested_k=requested_k,
                effective_k=0,
                total_candidates=0,
                accepted_candidates=0,
                top_score=None,
                latency_ms=round(elapsed, 2),
                threshold_applied=active_threshold,
                is_accepted=False,
                error=str(exc),
            )

        # 2. Check collection count
        count = self.collection.count()
        if count == 0:
            elapsed = (time.perf_counter() - start_time) * 1000.0
            logger.warning("Search called on an empty collection.")
            return [], RetrievalDiagnostics(
                query_length=len(norm_query),
                requested_k=requested_k,
                effective_k=0,
                total_candidates=0,
                accepted_candidates=0,
                top_score=None,
                latency_ms=round(elapsed, 2),
                threshold_applied=active_threshold,
                is_accepted=False,
                error="Collection is empty",
            )

        effective_k = self.validate_k(requested_k, count)

        # 3. Query ChromaDB
        try:
            raw_res = self.collection.query(
                query_texts=[norm_query],
                n_results=effective_k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            elapsed = (time.perf_counter() - start_time) * 1000.0
            logger.error("ChromaDB query failed: %s", exc)
            return [], RetrievalDiagnostics(
                query_length=len(norm_query),
                requested_k=requested_k,
                effective_k=effective_k,
                total_candidates=0,
                accepted_candidates=0,
                top_score=None,
                latency_ms=round(elapsed, 2),
                threshold_applied=active_threshold,
                is_accepted=False,
                error=f"Database query error: {type(exc).__name__}",
            )

        # 4. Safe parsing of ChromaDB response
        results: List[SearchResult] = []
        ids = (raw_res.get("ids") or [[]])[0]
        docs = (raw_res.get("documents") or [[]])[0]
        metas = (raw_res.get("metadatas") or [[]])[0]
        dists = (raw_res.get("distances") or [[]])[0]

        for item_id, doc, meta, dist in zip(ids, docs, metas, dists):
            meta_dict = meta if isinstance(meta, dict) else {}
            answer = meta_dict.get("answer", "")
            raw_dist = float(dist) if dist is not None else 1.0
            similarity = float(1.0 - raw_dist)

            results.append(
                SearchResult(
                    id=str(item_id),
                    question=str(doc or ""),
                    answer=str(answer or ""),
                    score=similarity,
                    distance=raw_dist,
                )
            )

        elapsed = (time.perf_counter() - start_time) * 1000.0

        # 5. Evaluate acceptance against threshold
        top_score = results[0].score if results else None
        is_accepted = bool(top_score is not None and top_score >= active_threshold)
        accepted_count = sum(1 for r in results if r.score >= active_threshold)

        diagnostics = RetrievalDiagnostics(
            query_length=len(norm_query),
            requested_k=requested_k,
            effective_k=effective_k,
            total_candidates=len(results),
            accepted_candidates=accepted_count,
            top_score=top_score,
            latency_ms=round(elapsed, 2),
            threshold_applied=active_threshold,
            is_accepted=is_accepted,
            error=None,
        )

        return results, diagnostics

    @staticmethod
    def filter_relevant(
        results: List[SearchResult], threshold: float, max_items: int = 3
    ) -> List[SearchResult]:
        """Single source of truth for selecting qualifying context items."""
        return [r for r in results if r.score >= threshold][:max_items]

    def search_with_history(
        self,
        query: str,
        history: Optional[List[Any]] = None,
        k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> Tuple[List[SearchResult], RetrievalDiagnostics]:
        """Perform semantic retrieval with multi-turn conversation context.

        When history is present, executes primary search and contextualized
        search (using the most recent turn's user question), merging results
        by highest similarity score.

        If history is empty or None, directly executes standard search().
        """
        if not history:
            return self.search(query, k=k, threshold=threshold)

        primary_results, primary_diag = self.search(query, k=k, threshold=threshold)
        if primary_diag.error:
            return primary_results, primary_diag

        # Extract most recent user query from history
        prev_user_query = None
        for turn in reversed(history):
            uq = getattr(turn, "user_query", None) or (
                turn.get("user_query") if isinstance(turn, dict) else None
            )
            if uq and isinstance(uq, str) and uq.strip():
                prev_user_query = uq.strip()
                break

        if not prev_user_query:
            return primary_results, primary_diag

        contextual_query = f"{prev_user_query} {query}"
        context_results, context_diag = self.search(
            contextual_query, k=k, threshold=threshold
        )

        # Merge candidate pools taking maximum score for any duplicate FAQ
        merged_map: dict[str, SearchResult] = {}
        for r in primary_results:
            merged_map[r.id] = r

        for r in context_results:
            if r.id not in merged_map or r.score > merged_map[r.id].score:
                merged_map[r.id] = r

        merged_sorted = sorted(merged_map.values(), key=lambda x: x.score, reverse=True)
        active_k = k if k is not None else self.config.default_top_k
        final_results = merged_sorted[:active_k]

        active_threshold = (
            threshold if threshold is not None else self.config.similarity_threshold
        )
        top_score = final_results[0].score if final_results else None
        is_accepted = bool(top_score is not None and top_score >= active_threshold)
        accepted_count = sum(1 for r in final_results if r.score >= active_threshold)
        total_latency = round(primary_diag.latency_ms + context_diag.latency_ms, 2)

        diagnostics = RetrievalDiagnostics(
            query_length=len(query),
            requested_k=primary_diag.requested_k,
            effective_k=len(final_results),
            total_candidates=len(final_results),
            accepted_candidates=accepted_count,
            top_score=top_score,
            latency_ms=total_latency,
            threshold_applied=active_threshold,
            is_accepted=is_accepted,
            error=None,
        )

        return final_results, diagnostics


