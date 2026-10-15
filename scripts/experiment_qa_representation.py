"""Reproducible experiment comparing Question-Only vs. Question+Answer document representations.

Hypothesis:
    Question-only embedding aligns directly with customer query intent.
    Question+Answer combined embedding might capture vocabulary appearing only
    in the answer, but might dilute query alignment and increase false positive leakage.
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Dict, List, Tuple

import chromadb
import numpy as np
import pandas as pd

from clean_data import get_clean_df
from config import get_config
from ingest import (
    ensure_indexed,
    get_chroma_client,
    get_embedding_function,
)
from retrieval import FAQRetriever, SearchResult

logger = logging.getLogger(__name__)


def create_qa_collection(
    client: chromadb.PersistentClient,
    faq_df: pd.DataFrame,
    embed_fn: Any,
    collection_name: str = "customer_support_faqs_qa_representation",
) -> chromadb.Collection:
    """Build a comparison collection embedding 'Question: ... \nAnswer: ...'."""
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    ids = []
    documents = []
    metadatas = []

    for _, row in faq_df.iterrows():
        f_id = row["faq_id"]
        q = row["question"]
        a = row["answer"]
        # Combined Q+A text representation
        doc_text = f"Question: {q}\nAnswer: {a}"

        ids.append(f_id)
        documents.append(doc_text)
        metadatas.append({"question": q, "answer": a})

    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    return collection


def evaluate_retrieval_performance(
    retriever: FAQRetriever,
    val_items: List[dict],
    k: int = 3,
) -> Dict[str, float]:
    """Calculate retrieval metrics (Hit@1, Hit@k, MRR) for in-scope queries."""
    norm = lambda s: " ".join(str(s or "").lower().split())

    hits_at_1 = []
    hits_at_k = []
    reciprocal_ranks = []
    in_scope_scores = []
    ood_scores = []

    for item in val_items:
        q = item["user_input"]
        is_answerable = bool(item.get("answerable", True))
        source_questions = {norm(sq) for sq in item.get("source_questions", [])}

        results, _ = retriever.search(q, k=k, threshold=-1.0)
        top_score = results[0].score if results else 0.0

        if is_answerable:
            in_scope_scores.append(top_score)
            retrieved_qs = []
            for r in results:
                raw_q = r.question
                if raw_q.startswith("Question: ") and "\nAnswer:" in raw_q:
                    raw_q = raw_q.split("\nAnswer:")[0].replace("Question: ", "").strip()
                retrieved_qs.append(norm(raw_q))

            # Hit@1
            match_at_1 = bool(retrieved_qs and retrieved_qs[0] in source_questions)
            hits_at_1.append(match_at_1)

            # Hit@k
            ranks = [i for i, rq in enumerate(retrieved_qs, start=1) if rq in source_questions]
            hits_at_k.append(bool(ranks))
            reciprocal_ranks.append(1.0 / ranks[0] if ranks else 0.0)
        else:
            ood_scores.append(top_score)

    return {
        "Hit@1": float(np.mean(hits_at_1)) if hits_at_1 else 0.0,
        f"Hit@{k}": float(np.mean(hits_at_k)) if hits_at_k else 0.0,
        "MRR": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
        "mean_in_scope_score": float(np.mean(in_scope_scores)) if in_scope_scores else 0.0,
        "mean_ood_score": float(np.mean(ood_scores)) if ood_scores else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Question-Only vs Q+A representation")
    parser.add_argument("--eval-set", type=str, default="ragas_eval_set.json", help="Path to evaluation queries")
    parser.add_argument("--k", type=int, default=3, help="Top-k evaluation parameter")
    args = parser.parse_args()

    config = get_config()
    df = get_clean_df()

    with open(args.eval_set, "r", encoding="utf-8") as f:
        val_items = json.load(f)

    print(f"Loaded {len(val_items)} test queries.")
    client = get_chroma_client()
    embed_fn = get_embedding_function()

    # 1. Evaluate Baseline (Question-Only)
    print("\n--- Evaluating Baseline: Question-Only Embedding ---")
    base_coll = ensure_indexed()
    base_retriever = FAQRetriever(base_coll, config)
    base_metrics = evaluate_retrieval_performance(base_retriever, val_items, k=args.k)

    # 2. Evaluate Experiment (Question + Answer Combined)
    print("\n--- Evaluating Experiment: Question + Answer Combined Embedding ---")
    qa_coll = create_qa_collection(client, df, embed_fn)
    qa_retriever = FAQRetriever(qa_coll, config)
    qa_metrics = evaluate_retrieval_performance(qa_retriever, val_items, k=args.k)

    # 3. Report Comparison Table
    comparison = pd.DataFrame([
        {"Metric": "Hit@1", "Question-Only (Baseline)": f"{base_metrics['Hit@1']:.2%}", "Q+A Combined": f"{qa_metrics['Hit@1']:.2%}"},
        {"Metric": f"Hit@{args.k}", "Question-Only (Baseline)": f"{base_metrics[f'Hit@{args.k}']:.2%}", "Q+A Combined": f"{qa_metrics[f'Hit@{args.k}']:.2%}"},
        {"Metric": "MRR", "Question-Only (Baseline)": f"{base_metrics['MRR']:.3f}", "Q+A Combined": f"{qa_metrics['MRR']:.3f}"},
        {"Metric": "Mean In-Scope Score", "Question-Only (Baseline)": f"{base_metrics['mean_in_scope_score']:.3f}", "Q+A Combined": f"{qa_metrics['mean_in_scope_score']:.3f}"},
        {"Metric": "Mean Out-of-Scope Score", "Question-Only (Baseline)": f"{base_metrics['mean_ood_score']:.3f}", "Q+A Combined": f"{qa_metrics['mean_ood_score']:.3f}"},
    ])

    print("\n=== Representation Experiment Results ===")
    print(comparison.to_string(index=False))

    print("\n=== Analysis and Architectural Decision ===")
    if base_metrics[f"Hit@{args.k}"] >= qa_metrics[f"Hit@{args.k}"]:
        print(
            "RESULT: Baseline Question-Only representation performs equally or better.\n"
            "Recommendation: Retain Question-Only representation as production default.\n"
            "Reasoning: User questions match the intent and style of FAQ questions more directly,\n"
            "avoiding vector dilution from long answer paragraphs."
        )
    else:
        print(
            "RESULT: Combined Q+A produced higher retrieval scores on this dataset.\n"
            "Recommendation: Conduct manual review before switching production defaults to\n"
            "ensure general out-of-scope queries do not match accidental words in answers."
        )


if __name__ == "__main__":
    main()

