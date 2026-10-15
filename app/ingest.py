"""ChromaDB ingestion, index lifecycle management, and high-level RAG orchestration."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple

import chromadb
from chromadb.utils import embedding_functions
import pandas as pd

try:
    from scripts.clean_data import clean_faq_dataframe, compute_faq_id, get_clean_df, _LazyDFProxy
except ImportError:
    from clean_data import clean_faq_dataframe, compute_faq_id, get_clean_df, _LazyDFProxy

try:
    from app.config import AppConfig, get_config
    from app.generation import AnswerGenerator, GenerationResult
    from app.retrieval import FAQRetriever, SearchResult
except ImportError:
    from config import AppConfig, get_config
    from generation import AnswerGenerator, GenerationResult
    from retrieval import FAQRetriever, SearchResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Constants preserved for backward compatibility
config = get_config()
MODEL_NAME = config.embedding_model_name
DB_PATH = config.chroma_db_path
COLLECTION_NAME = config.collection_name
LLM_MODEL = config.llm_model
GROQ_BASE_URL = config.groq_base_url
GROQ_API_KEY = config.groq_api_key
FALLBACK_MESSAGE = config.fallback_message

# Cleaned dataframe proxy (loaded lazily on demand, NOT at import time)
df = _LazyDFProxy()

# Metadata file tracking index state and fingerprint
FINGERPRINT_FILE_NAME = "index_manifest.json"
INDEX_SCHEMA_VERSION = "2.0"


def compute_dataset_fingerprint(
    faq_df: pd.DataFrame,
    embedding_model_name: str,
    distance_metric: str = "cosine",
    schema_version: str = INDEX_SCHEMA_VERSION,
) -> str:
    """Compute a cryptographic hash of all FAQ pairs and index configuration."""
    hasher = hashlib.sha256()
    hasher.update(embedding_model_name.encode("utf-8"))
    hasher.update(distance_metric.encode("utf-8"))
    hasher.update(schema_version.encode("utf-8"))

    # Sort deterministically by question
    sorted_df = faq_df.sort_values(by="question").reset_index(drop=True)
    for _, row in sorted_df.iterrows():
        hasher.update(str(row["question"]).strip().encode("utf-8"))
        hasher.update(str(row["answer"]).strip().encode("utf-8"))

    return hasher.hexdigest()


def get_manifest_path(db_path: str) -> Path:
    return Path(db_path) / FINGERPRINT_FILE_NAME


def read_index_manifest(db_path: str) -> Optional[dict]:
    manifest_path = get_manifest_path(db_path)
    if not manifest_path.exists():
        return None
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("Could not read index manifest: %s", exc)
        return None


def write_index_manifest(
    db_path: str,
    fingerprint: str,
    collection_name: str,
    record_count: int,
    embedding_model: str,
) -> None:
    manifest_path = get_manifest_path(db_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "fingerprint": fingerprint,
        "collection_name": collection_name,
        "record_count": record_count,
        "embedding_model": embedding_model,
        "schema_version": INDEX_SCHEMA_VERSION,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def get_chroma_client(db_path: Optional[str] = None) -> chromadb.PersistentClient:
    """Instantiate a persistent ChromaDB client."""
    target_path = db_path or config.chroma_db_path
    Path(target_path).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=target_path)


def get_embedding_function(model_name: Optional[str] = None):
    """Instantiate the ChromaDB sentence transformer embedding function singleton."""
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=model_name or config.embedding_model_name
    )


def build_collection(
    client: chromadb.PersistentClient,
    collection_name: str,
    embed_fn: Any,
) -> chromadb.Collection:
    """Get or create collection with cosine space."""
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )


def populate_collection(
    collection: chromadb.Collection,
    faq_df: pd.DataFrame,
    batch_size: int = 128,
) -> None:
    """Batch-upsert FAQ documents and metadata into ChromaDB."""
    ids: List[str] = []
    documents: List[str] = []
    metadatas: List[dict] = []

    for _, row in faq_df.iterrows():
        f_id = row.get("faq_id") or compute_faq_id(row["question"])
        ids.append(f_id)
        documents.append(str(row["question"]))
        metadatas.append({"answer": str(row["answer"])})

    total = len(ids)
    for i in range(0, total, batch_size):
        chunk_ids = ids[i : i + batch_size]
        chunk_docs = documents[i : i + batch_size]
        chunk_metas = metadatas[i : i + batch_size]
        collection.upsert(
            ids=chunk_ids,
            documents=chunk_docs,
            metadatas=chunk_metas,
        )
    logger.info("Indexed %d records into collection '%s'.", total, collection.name)


def rebuild_index(
    faq_df: pd.DataFrame,
    db_path: Optional[str] = None,
    collection_name: Optional[str] = None,
    embed_model: Optional[str] = None,
) -> chromadb.Collection:
    """Safe, deterministic full rebuild of the ChromaDB index."""
    target_db = db_path or config.chroma_db_path
    target_col = collection_name or config.collection_name
    target_model = embed_model or config.embedding_model_name

    logger.info("Performing full rebuild of collection '%s' at '%s'...", target_col, target_db)
    client = get_chroma_client(target_db)
    embed_fn = get_embedding_function(target_model)

    # Delete existing collection if present to prevent any stale orphaned documents
    try:
        client.delete_collection(target_col)
        logger.info("Deleted existing collection '%s'.", target_col)
    except Exception:
        pass  # Collection did not exist

    collection = client.create_collection(
        name=target_col,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    populate_collection(collection, faq_df)

    # Validate post-index integrity
    actual_count = collection.count()
    expected_count = len(faq_df)
    if actual_count != expected_count:
        raise RuntimeError(
            f"Index validation failed: expected {expected_count} records, found {actual_count}."
        )

    # Write manifest
    fp = compute_dataset_fingerprint(faq_df, target_model)
    write_index_manifest(target_db, fp, target_col, actual_count, target_model)
    logger.info("Index manifest saved successfully (fingerprint=%s).", fp[:12])
    return collection


def ensure_indexed(
    faq_df: Optional[pd.DataFrame] = None,
    db_path: Optional[str] = None,
    collection_name: Optional[str] = None,
    embed_model: Optional[str] = None,
    force_rebuild: bool = False,
) -> chromadb.Collection:
    """Ensure the ChromaDB collection matches the current FAQ dataset and configuration.

    Uses deterministic cryptographic fingerprinting to detect added, modified,
    reordered, or deleted records, as well as embedding model changes.
    """
    target_df = faq_df if faq_df is not None else get_clean_df()
    target_db = db_path or config.chroma_db_path
    target_col = collection_name or config.collection_name
    target_model = embed_model or config.embedding_model_name

    current_fingerprint = compute_dataset_fingerprint(target_df, target_model)
    manifest = read_index_manifest(target_db)

    client = get_chroma_client(target_db)
    embed_fn = get_embedding_function(target_model)

    # Check if index is already up to date
    if not force_rebuild and manifest is not None:
        if (
            manifest.get("fingerprint") == current_fingerprint
            and manifest.get("collection_name") == target_col
            and manifest.get("schema_version") == INDEX_SCHEMA_VERSION
        ):
            try:
                coll = client.get_collection(target_col, embedding_function=embed_fn)
                if coll.count() == len(target_df):
                    logger.info(
                        "Collection '%s' is up-to-date (%d items, fingerprint=%s).",
                        target_col,
                        coll.count(),
                        current_fingerprint[:12],
                    )
                    return coll
            except Exception:
                pass  # Fall through to rebuild if collection missing in Chroma

    # Fingerprint mismatch or missing collection -> rebuild
    logger.info("Dataset or configuration changed; synchronizing index...")
    return rebuild_index(target_df, target_db, target_col, target_model)


# Initialize default collection singleton lazily
_COLLECTION_INSTANCE: Optional[chromadb.Collection] = None
_RETRIEVER_INSTANCE: Optional[FAQRetriever] = None
_GENERATOR_INSTANCE: Optional[AnswerGenerator] = None


def get_default_retriever() -> FAQRetriever:
    global _COLLECTION_INSTANCE, _RETRIEVER_INSTANCE
    if _RETRIEVER_INSTANCE is None:
        _COLLECTION_INSTANCE = ensure_indexed()
        _RETRIEVER_INSTANCE = FAQRetriever(_COLLECTION_INSTANCE, config)
    return _RETRIEVER_INSTANCE


def get_default_generator() -> AnswerGenerator:
    global _GENERATOR_INSTANCE
    if _GENERATOR_INSTANCE is None:
        _GENERATOR_INSTANCE = AnswerGenerator(config)
    return _GENERATOR_INSTANCE


# Lazy Collection Proxy so `from ingest import collection` works seamlessly without import overhead
class _LazyCollectionProxy:
    def __getattr__(self, name: str) -> Any:
        global _COLLECTION_INSTANCE
        if _COLLECTION_INSTANCE is None:
            _COLLECTION_INSTANCE = ensure_indexed()
        return getattr(_COLLECTION_INSTANCE, name)


collection = _LazyCollectionProxy()


# Public functions preserved for backward compatibility
def search(
    query: str,
    k: Optional[int] = None,
    threshold: Optional[float] = None,
) -> Tuple[Optional[str], List[dict]]:
    """Backward-compatible search function.

    Returns:
        (top_answer_or_None, list_of_result_dicts)
    """
    retriever = get_default_retriever()
    active_threshold = threshold if threshold is not None else config.similarity_threshold
    active_k = k if k is not None else config.default_top_k

    results, diagnostics = retriever.search(
        query, k=active_k, threshold=active_threshold
    )
    result_dicts = [r.to_dict() for r in results]

    if not results or results[0].score < active_threshold:
        return None, result_dicts
    return results[0].answer, result_dicts


def generate_answer(
    query: str,
    k: Optional[int] = None,
    threshold: Optional[float] = None,
    temperature: Optional[float] = None,
    debug: bool = False,
    history: Optional[List[Any]] = None,
) -> dict:
    """Backward-compatible answer generation function.

    Returns:
        Dict with keys: answer, results, contexts, fallback, latency_ms
    """
    retriever = get_default_retriever()
    generator = get_default_generator()

    active_threshold = threshold if threshold is not None else config.similarity_threshold
    active_k = k if k is not None else config.default_top_k

    if history and hasattr(retriever, "search_with_history"):
        results, diagnostics = retriever.search_with_history(
            query, history=history, k=active_k, threshold=active_threshold
        )
    else:
        results, diagnostics = retriever.search(
            query, k=active_k, threshold=active_threshold
        )

    if debug:
        print(f"[Search Debug] Query: '{query}'")
        for r in results:
            accepted = "ACCEPT" if r.score >= active_threshold else "REJECT"
            print(f"  {r.score:.3f} [{accepted}] {r.question}")

    # Single source of truth for context filtering
    relevant = retriever.filter_relevant(results, active_threshold, max_items=active_k)

    gen_res: GenerationResult = generator.generate(query, relevant, history=history)

    return {
        "answer": gen_res.answer,
        "results": [r.to_dict() for r in results],
        "contexts": [r.to_dict() for r in relevant],
        "fallback": gen_res.fallback,
        "latency_ms": gen_res.latency_ms,
        "prompt_tokens": gen_res.prompt_tokens,
        "completion_tokens": gen_res.completion_tokens,
        "diagnostics": diagnostics,
    }


def health_check() -> dict:
    """Verify that the database and indexing layers are operational."""
    try:
        coll = ensure_indexed()
        count = coll.count()
        sample_res, _ = FAQRetriever(coll, config).search("test", k=1)
        return {
            "status": "healthy",
            "indexed_faqs": count,
            "database_path": config.chroma_db_path,
            "collection_name": config.collection_name,
            "embedding_model": config.embedding_model_name,
            "test_retrieval_ok": True,
        }
    except Exception as exc:
        return {
            "status": "unhealthy",
            "error": str(exc),
        }


def main():
    parser = argparse.ArgumentParser(description="Customer Support FAQ RAG Chatbot")
    parser.add_argument("--reindex", action="store_true", help="Force a complete re-index of the dataset")
    parser.add_argument("--health", action="store_true", help="Run system health check and exit")
    parser.add_argument("--query", type=str, help="Run a single test query and print the result")
    args = parser.parse_args()

    if args.health:
        h = health_check()
        print(json.dumps(h, indent=2))
        sys.exit(0 if h["status"] == "healthy" else 1)

    if args.reindex:
        logger.info("Forced reindex requested via CLI.")
        ensure_indexed(force_rebuild=True)
        print("Reindexing complete.")
        return

    if args.query:
        out = generate_answer(args.query, debug=True)
        print(f"\nQuery: {args.query}")
        print(f"Bot: {out['answer']}\n")
        return

    print("\nCustomer Support FAQ Bot Ready. Type 'exit' or 'quit' to end.\n")
    ensure_indexed()

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break

        try:
            out = generate_answer(user_input, debug=False)
            print(f"\nBot: {out['answer']}\n")
        except Exception as err:
            logger.error("CLI error: %s", err)
            print(f"\nBot: {config.fallback_message}\n")


if __name__ == "__main__":
    main()