"""Reproducible relevance-threshold calibration and evaluation workflow.

Key Concepts & Limitations:
    1. Acceptance vs. Retrieval Correctness:
       - Acceptance metrics evaluate the binary threshold filter: does the query's similarity
         score clear threshold tau?
         * answerable_coverage: fraction of answerable queries that pass threshold.
         * unanswerable_rejection_rate: fraction of unanswerable/OOD queries rejected.
         * acceptance_precision: fraction of all accepted queries that are answerable.
       - Retrieval metrics evaluate ranking accuracy against labeled ground truth source FAQs
         (computed on answerable queries only):
         * Hit@1, Hit@k, and MRR.
       - An accepted answerable query that retrieves the wrong source FAQ counts toward
         answerable coverage, but NOT as a retrieval hit.
    2. Cosine similarity in embedding space is an uncalibrated geometric ranking signal,
       NOT a probability of answer correctness.
    3. Tuning data (development set) must remain strictly separate from held-out evaluation
       data. The threshold is selected using the development set only; the held-out set is
       evaluated once to measure performance on unseen queries.
    4. Small sample sizes (e.g., 45 queries total, or strata with 0-5 items) mean results
       are preliminary indicators and should not be claimed as an unbiased representation of
       real production user traffic.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from clean_data import normalize_text
from config import get_config
from ingest import ensure_indexed
from retrieval import FAQRetriever

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CategorySplitReport:
    """Detailed split diagnostic for a specific query category."""
    category: str
    total_count: int
    dev_count: int
    test_count: int
    is_too_small_to_split: bool
    warning: Optional[str] = None


def partition_dev_and_test_sets(
    items: List[dict],
    dev_ratio: float = 0.7,
    seed: int = 42,
) -> Tuple[List[dict], List[dict], List[CategorySplitReport]]:
    """Perform deterministic stratified partitioning by query_type and answerability.

    Args:
        items: List of labeled evaluation items.
        dev_ratio: Fraction of data assigned to development/tuning (default: 0.70).
        seed: Random seed for deterministic shuffling.

    Returns:
        (dev_set, test_set, category_reports)
    """
    if not (0.0 < dev_ratio < 1.0):
        raise ValueError(f"dev_ratio must be between 0.0 and 1.0 (exclusive), got {dev_ratio}")

    rng = random.Random(seed)
    buckets: Dict[str, List[dict]] = {}

    for item in items:
        q_type = str(item.get("query_type", "single_hop"))
        ans = "ans" if item.get("answerable", True) else "una"
        key = f"{q_type}_{ans}"
        buckets.setdefault(key, []).append(item)

    dev_set: List[dict] = []
    test_set: List[dict] = []
    reports: List[CategorySplitReport] = []

    for key, group in sorted(buckets.items()):
        shuffled = list(group)
        rng.shuffle(shuffled)
        total = len(shuffled)

        if total == 1:
            # Cannot be split across both sets; assign to dev so tuner observes it
            dev_set.extend(shuffled)
            rep = CategorySplitReport(
                category=key,
                total_count=1,
                dev_count=1,
                test_count=0,
                is_too_small_to_split=True,
                warning=(
                    f"Category '{key}' has only 1 example. Assigned to development set; "
                    "held-out test set contains 0 examples for this category."
                ),
            )
            reports.append(rep)
            logger.warning(rep.warning)
            continue

        # For total >= 2, ensure at least 1 in dev and at least 1 in test
        cut = max(1, min(total - 1, int(round(total * dev_ratio))))
        dev_subset = shuffled[:cut]
        test_subset = shuffled[cut:]

        dev_set.extend(dev_subset)
        test_set.extend(test_subset)

        warning = None
        if total < 5:
            warning = (
                f"Category '{key}' has only {total} examples ({len(dev_subset)} dev, {len(test_subset)} test). "
                "Per-category metrics will have high variance and cannot support standalone conclusions."
            )
            logger.info(warning)

        rep = CategorySplitReport(
            category=key,
            total_count=total,
            dev_count=len(dev_subset),
            test_count=len(test_subset),
            is_too_small_to_split=False,
            warning=warning,
        )
        reports.append(rep)

    return dev_set, test_set, reports


def build_evaluation_cache(
    val_items: List[dict],
    retriever: FAQRetriever,
    k: int = 3,
) -> List[dict]:
    """Execute raw retrieval for all queries once, storing candidate ranks and scores."""
    eval_cache = []
    norm = lambda s: " ".join(str(s or "").lower().split())

    for item in val_items:
        q = item["user_input"]
        # Search with threshold=-1.0 to retrieve unfiltered top-k candidates
        results, diag = retriever.search(q, k=k, threshold=-1.0)
        top_score = results[0].score if results else -1.0

        is_answerable = bool(item.get("answerable", True))
        source_questions = [norm(sq) for sq in item.get("source_questions", [])]

        matched_rank = None
        if is_answerable and source_questions and results:
            for rank_idx, res in enumerate(results, start=1):
                if norm(res.question) in source_questions:
                    matched_rank = rank_idx
                    break

        hit_at_1 = (matched_rank == 1)
        hit_at_k = (matched_rank is not None)
        reciprocal_rank = (1.0 / matched_rank) if matched_rank is not None else 0.0

        eval_cache.append({
            "id": item.get("id"),
            "query": q,
            "query_type": item.get("query_type", "unknown"),
            "answerable": is_answerable,
            "top_score": top_score,
            "matched_rank": matched_rank,
            "hit_at_1": hit_at_1,
            "hit_at_k": hit_at_k,
            "reciprocal_rank": reciprocal_rank,
        })

    return eval_cache


def compute_metrics_from_cache(
    eval_cache: List[dict],
    threshold: float,
) -> Dict[str, Any]:
    """Compute decoupled acceptance and retrieval metrics for a specified threshold."""
    in_scope_items = [c for c in eval_cache if c["answerable"]]
    ood_items = [c for c in eval_cache if not c["answerable"]]

    total_answerable = len(in_scope_items)

    # 1. Baseline retrieval quality on all answerable queries (unfiltered)
    if total_answerable > 0:
        base_hit1 = sum(1 for c in in_scope_items if c["hit_at_1"]) / total_answerable
        base_hitk = sum(1 for c in in_scope_items if c["hit_at_k"]) / total_answerable
        base_mrr = sum(c["reciprocal_rank"] for c in in_scope_items) / total_answerable
    else:
        base_hit1 = None
        base_hitk = None
        base_mrr = None

    # 2. Acceptance gating for given threshold
    accepted_answerable = 0
    false_refusals = 0
    hit1_accepted_count = 0
    hitk_accepted_count = 0
    rr_accepted_sum = 0.0

    for c in in_scope_items:
        if c["top_score"] >= threshold:
            accepted_answerable += 1
            if c["hit_at_1"]:
                hit1_accepted_count += 1
            if c["hit_at_k"]:
                hitk_accepted_count += 1
            rr_accepted_sum += c["reciprocal_rank"]
        else:
            false_refusals += 1

    # In-domain unanswerable vs. out-of-scope breakdown
    in_domain_unans_items = [c for c in ood_items if c.get("query_type") == "unanswerable"]
    oos_items = [c for c in ood_items if c.get("query_type") == "out_of_scope"]

    unans_in_domain_count = len(in_domain_unans_items)
    oos_count = len(oos_items)

    unans_in_domain_rejected = sum(1 for c in in_domain_unans_items if c["top_score"] < threshold)
    unans_in_domain_leakage = sum(1 for c in in_domain_unans_items if c["top_score"] >= threshold)

    oos_rejected = sum(1 for c in oos_items if c["top_score"] < threshold)
    oos_leakage = sum(1 for c in oos_items if c["top_score"] >= threshold)

    unans_in_domain_rejection_rate = (
        (unans_in_domain_rejected / unans_in_domain_count) if unans_in_domain_count > 0 else None
    )
    oos_rejection_rate = (
        (oos_rejected / oos_count) if oos_count > 0 else None
    )

    combined_unans_total = unans_in_domain_count + oos_count
    combined_unans_rejected = unans_in_domain_rejected + oos_rejected
    combined_unans_leakage = unans_in_domain_leakage + oos_leakage
    combined_rejection_rate = (
        (combined_unans_rejected / combined_unans_total) if combined_unans_total > 0 else None
    )

    total_accepted = accepted_answerable + combined_unans_leakage

    # Acceptance metrics (gate filtering only)
    coverage = (accepted_answerable / total_answerable) if total_answerable > 0 else None
    precision = (accepted_answerable / total_accepted) if total_accepted > 0 else 0.0

    if coverage is not None and (precision + coverage) > 0:
        acceptance_f1 = (2 * precision * coverage) / (precision + coverage)
    else:
        acceptance_f1 = 0.0

    # 3. Retrieval correctness among accepted answerable queries
    # If no answerable queries passed the threshold, denominator is 0 -> return None (N/A)
    # If accepted queries exist but none matched ground truth -> return 0.0
    if accepted_answerable > 0:
        hit1_among_accepted = hit1_accepted_count / accepted_answerable
        hitk_among_accepted = hitk_accepted_count / accepted_answerable
        mrr_among_accepted = rr_accepted_sum / accepted_answerable
    else:
        hit1_among_accepted = None
        hitk_among_accepted = None
        mrr_among_accepted = None

    strict_hit1_accepted_rate = (hit1_accepted_count / total_answerable) if total_answerable > 0 else None

    return {
        "threshold": round(threshold, 2),
        # Acceptance gating
        "acceptance_precision": round(precision, 3),
        "answerable_coverage": round(coverage, 3) if coverage is not None else None,
        "acceptance_f1": round(acceptance_f1, 3),
        "unanswerable_rejection_rate": round(combined_rejection_rate, 3) if combined_rejection_rate is not None else None,
        "unanswerable_combined_rejection_rate": round(combined_rejection_rate, 3) if combined_rejection_rate is not None else None,
        "unanswerable_in_domain_rejection_rate": round(unans_in_domain_rejection_rate, 3) if unans_in_domain_rejection_rate is not None else None,
        "out_of_scope_rejection_rate": round(oos_rejection_rate, 3) if oos_rejection_rate is not None else None,
        "accepted_answerable_count": accepted_answerable,
        "false_refusal_count": false_refusals,
        "unanswerable_leakage_count": combined_unans_leakage,
        "unanswerable_rejection_count": combined_unans_rejected,
        "unanswerable_in_domain_count": unans_in_domain_count,
        "unanswerable_in_domain_rejected_count": unans_in_domain_rejected,
        "unanswerable_in_domain_leakage_count": unans_in_domain_leakage,
        "out_of_scope_count": oos_count,
        "out_of_scope_rejected_count": oos_rejected,
        "out_of_scope_leakage_count": oos_leakage,
        "total_answerable": total_answerable,
        "total_unanswerable": combined_unans_total,
        # Unfiltered retrieval quality
        "baseline_hit_at_1": round(base_hit1, 3) if base_hit1 is not None else None,
        "baseline_hit_at_k": round(base_hitk, 3) if base_hitk is not None else None,
        "baseline_mrr": round(base_mrr, 3) if base_mrr is not None else None,
        # Retrieval quality among accepted queries
        "hit1_among_accepted": round(hit1_among_accepted, 3) if hit1_among_accepted is not None else None,
        "hitk_among_accepted": round(hitk_among_accepted, 3) if hitk_among_accepted is not None else None,
        "mrr_among_accepted": round(mrr_among_accepted, 3) if mrr_among_accepted is not None else None,
        "hit1_accepted_count": hit1_accepted_count,
        "hitk_accepted_count": hitk_accepted_count,
        "strict_hit1_accepted_rate": round(strict_hit1_accepted_rate, 3) if strict_hit1_accepted_rate is not None else None,
    }


def run_threshold_sweep(
    val_items: List[dict],
    retriever: FAQRetriever,
    thresholds: List[float],
    k: int = 3,
) -> pd.DataFrame:
    """Evaluate retrieval acceptance and correctness across multiple thresholds."""
    eval_cache = build_evaluation_cache(val_items, retriever, k=k)
    rows = [compute_metrics_from_cache(eval_cache, t) for t in thresholds]
    return pd.DataFrame(rows)


def evaluate_single_threshold(
    items: List[dict],
    retriever: FAQRetriever,
    threshold: float,
    k: int = 3,
) -> Dict[str, Any]:
    """Compute performance metrics for a specific fixed threshold."""
    eval_cache = build_evaluation_cache(items, retriever, k=k)
    return compute_metrics_from_cache(eval_cache, threshold)


def _find_data_file(path_str: str) -> str:
    if os.path.exists(path_str):
        return path_str
    candidates = [
        os.path.join("data", path_str),
        os.path.join(os.path.dirname(__file__), "..", "data", path_str),
        os.path.join(os.path.dirname(__file__), "data", path_str),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return path_str


def main():
    parser = argparse.ArgumentParser(description="Calibrate and evaluate RAG relevance threshold")
    parser.add_argument("--eval-set", type=str, default="ragas_eval_set.json", help="Path to master labeled evaluation set")
    parser.add_argument("--dev-file", type=str, default="eval_dev_tuning_set.json", help="Path to development/tuning set file")
    parser.add_argument("--test-file", type=str, default="eval_held_out_test_set.json", help="Path to held-out test set file")
    parser.add_argument("--k", type=int, default=3, help="Top-k retrieval parameter")
    parser.add_argument("--dev-ratio", type=float, default=0.7, help="Proportion allocated to development/tuning split (default: 0.70)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic splitting")
    parser.add_argument("--repartition", action="store_true", help="Force re-partitioning of master evaluation set into dev and test files")
    parser.add_argument("--output-csv", type=str, default="threshold_calibration_report.csv", help="Output report CSV")
    args = parser.parse_args()

    args.eval_set = _find_data_file(args.eval_set)
    args.dev_file = _find_data_file(args.dev_file)
    args.test_file = _find_data_file(args.test_file)

    dev_items: List[dict] = []
    test_items: List[dict] = []
    split_reports: List[CategorySplitReport] = []

    dev_exists = os.path.exists(args.dev_file)
    test_exists = os.path.exists(args.test_file)

    if dev_exists and test_exists and not args.repartition:
        print(f"\n[1/3] Loading existing frozen evaluation splits:")
        print(f"  -> Development / Tuning Set: '{args.dev_file}'")
        with open(args.dev_file, "r", encoding="utf-8") as f:
            dev_items = json.load(f)
        print(f"  -> Held-Out Test Set:        '{args.test_file}'")
        with open(args.test_file, "r", encoding="utf-8") as f:
            test_items = json.load(f)
        print(f"  (Note: Using existing files to preserve split isolation. Use --repartition to re-split)")
    else:
        if not os.path.exists(args.eval_set):
            raise SystemExit(f"Master evaluation set '{args.eval_set}' not found.")
        with open(args.eval_set, "r", encoding="utf-8") as f:
            all_items = json.load(f)

        print(f"\n[1/3] Partitioning master evaluation set '{args.eval_set}' (dev_ratio={args.dev_ratio}, seed={args.seed}):")
        dev_items, test_items, split_reports = partition_dev_and_test_sets(
            all_items, dev_ratio=args.dev_ratio, seed=args.seed
        )

        with open(args.dev_file, "w", encoding="utf-8") as f:
            json.dump(dev_items, f, indent=2)
        with open(args.test_file, "w", encoding="utf-8") as f:
            json.dump(test_items, f, indent=2)

        print(f"  -> Saved development set:  '{args.dev_file}' ({len(dev_items)} items)")
        print(f"  -> Saved held-out test set: '{args.test_file}' ({len(test_items)} items)")

    # Print composition and small-category warnings
    dev_ans = sum(1 for x in dev_items if x.get('answerable', True))
    dev_unans = sum(1 for x in dev_items if not x.get('answerable', True) and x.get('query_type') == 'unanswerable')
    dev_oos = sum(1 for x in dev_items if not x.get('answerable', True) and x.get('query_type') == 'out_of_scope')

    test_ans = sum(1 for x in test_items if x.get('answerable', True))
    test_unans = sum(1 for x in test_items if not x.get('answerable', True) and x.get('query_type') == 'unanswerable')
    test_oos = sum(1 for x in test_items if not x.get('answerable', True) and x.get('query_type') == 'out_of_scope')

    print(f"\nEvaluation Split Composition:")
    print(f"  Development / Tuning Set ({len(dev_items)} total):")
    print(f"    - In-scope answerable:     {dev_ans}")
    print(f"    - In-domain unanswerable:  {dev_unans}")
    print(f"    - Out-of-scope:            {dev_oos}")
    print(f"  Held-Out Test Set ({len(test_items)} total):")
    print(f"    - In-scope answerable:     {test_ans}")
    print(f"    - In-domain unanswerable:  {test_unans}")
    print(f"    - Out-of-scope:            {test_oos}")

    if test_oos == 0:
        print(f"    * NOTE: Held-out set contains 0 out-of-scope examples; out-of-scope rejection CANNOT be evaluated on this split.")

    if split_reports:
        print("\n  Per-Category Split Diagnostics:")
        for r in split_reports:
            status = "TOO SMALL TO SPLIT" if r.is_too_small_to_split else "OK"
            print(f"    - [{status}] {r.category}: total={r.total_count} -> dev={r.dev_count}, test={r.test_count}")
            if r.warning:
                print(f"      NOTE: {r.warning}")

    collection = ensure_indexed()
    retriever = FAQRetriever(collection, get_config())

    # Phase 1: Threshold Sweep on Development / Tuning Data ONLY
    print(f"\n[2/3] Phase 1: Threshold Selection & Calibration (DEVELOPMENT SET ONLY)")
    print("  Sweeping similarity thresholds across tuning items to observe acceptance vs. retrieval trade-offs...")
    thresholds = [round(x, 2) for x in np.arange(0.30, 0.96, 0.05)]
    sweep_df = run_threshold_sweep(dev_items, retriever, thresholds, k=args.k)
    sweep_df.to_csv(args.output_csv, index=False)

    display_cols = [
        "threshold",
        "acceptance_precision",
        "answerable_coverage",
        "acceptance_f1",
        "unanswerable_rejection_rate",
        "hit1_among_accepted",
        "strict_hit1_accepted_rate",
        "false_refusal_count",
        "unanswerable_leakage_count",
    ]
    print(sweep_df[display_cols].to_string(index=False))

    # Pick recommended operating point based on development set
    best_f1_idx = sweep_df["acceptance_f1"].idxmax()
    best_f1_row = sweep_df.iloc[best_f1_idx]
    recommended_thresh = float(best_f1_row["threshold"])
    print(
        f"\n  Threshold selected from development set: {recommended_thresh:.2f} "
        f"(Dev Acceptance F1={best_f1_row['acceptance_f1']}, Coverage={best_f1_row['answerable_coverage']}, "
        f"Strict Hit@1={best_f1_row['strict_hit1_accepted_rate']})"
    )

    # Phase 2: Final Single-Point Evaluation on Held-Out Test Set (NO THRESHOLD TUNING)
    print(f"\n[3/3] Phase 2: Final Single-Point Evaluation (HELD-OUT TEST SET)")
    print(f"  Evaluating selected threshold ({recommended_thresh:.2f}) on held-out test set in isolation (NO TUNING)...")
    test_metrics = evaluate_single_threshold(test_items, retriever, threshold=recommended_thresh, k=args.k)

    print("\n  A. Retrieval Quality on Answerable Queries (Unfiltered Top-k Candidates):")
    print(f"    - Baseline Hit@1:                {test_metrics['baseline_hit_at_1']}")
    print(f"    - Baseline Hit@{args.k}:                {test_metrics['baseline_hit_at_k']}")
    print(f"    - Baseline MRR:                  {test_metrics['baseline_mrr']}")

    print("\n  B. Acceptance Gating Metrics (Threshold Filter):")
    cov_str = (
        f"{test_metrics['answerable_coverage']} ({test_metrics['accepted_answerable_count']}/{test_metrics['total_answerable']})"
        if test_metrics['answerable_coverage'] is not None else "N/A (0 answerable examples in split)"
    )
    prec_denom = test_metrics['accepted_answerable_count'] + test_metrics['unanswerable_leakage_count']
    prec_str = (
        f"{test_metrics['acceptance_precision']} ({test_metrics['accepted_answerable_count']}/{prec_denom})"
        if prec_denom > 0 else "N/A (0 queries accepted)"
    )
    print(f"    - Answerable Coverage (Recall):                 {cov_str}")
    print(f"    - Acceptance Precision:                         {prec_str}")
    print(f"    - Acceptance F1:                                {test_metrics['acceptance_f1']}")

    if test_metrics['unanswerable_combined_rejection_rate'] is not None:
        comb_str = f"{test_metrics['unanswerable_combined_rejection_rate']} ({test_metrics['unanswerable_rejection_count']}/{test_metrics['total_unanswerable']})"
    else:
        comb_str = "N/A (0 unanswerable/OOD examples in split)"
    print(f"    - Combined Unanswerable/OOD Rejection Rate:     {comb_str}  [Combines in-domain unanswerable + out-of-scope]")

    if test_metrics['unanswerable_in_domain_rejection_rate'] is not None:
        in_str = f"{test_metrics['unanswerable_in_domain_rejection_rate']} ({test_metrics['unanswerable_in_domain_rejected_count']}/{test_metrics['unanswerable_in_domain_count']} rejected)"
    else:
        in_str = "N/A (0 in-domain unanswerable examples in split)"
    print(f"      * In-Domain Unanswerable Rejection:           {in_str}")

    if test_metrics['out_of_scope_rejection_rate'] is not None:
        oos_str = f"{test_metrics['out_of_scope_rejection_rate']} ({test_metrics['out_of_scope_rejected_count']}/{test_metrics['out_of_scope_count']} rejected)"
    else:
        oos_str = "N/A (0 out-of-scope examples in split)"
    print(f"      * Out-of-Scope Rejection:                     {oos_str}")

    print(f"    - False Refusals (FN):                          {test_metrics['false_refusal_count']}")
    print(
        f"    - Unanswerable Leakage (FP):                    {test_metrics['unanswerable_leakage_count']} "
        f"({test_metrics['unanswerable_in_domain_leakage_count']} in-domain, {test_metrics['out_of_scope_leakage_count']} out-of-scope)"
    )

    print("\n  C. Retrieval Quality Among Accepted Answerable Queries:")
    hit1_str = f"{test_metrics['hit1_among_accepted']}" if test_metrics['hit1_among_accepted'] is not None else "N/A (0 accepted answerable queries)"
    hitk_str = f"{test_metrics['hitk_among_accepted']}" if test_metrics['hitk_among_accepted'] is not None else "N/A (0 accepted answerable queries)"
    mrr_str = f"{test_metrics['mrr_among_accepted']}" if test_metrics['mrr_among_accepted'] is not None else "N/A (0 accepted answerable queries)"
    print(f"    - Hit@1 Among Accepted:                         {hit1_str}")
    print(f"    - Hit@{args.k} Among Accepted:                         {hitk_str}")
    print(f"    - MRR Among Accepted:                           {mrr_str}")

    if test_metrics['strict_hit1_accepted_rate'] is not None:
        strict_str = f"{test_metrics['strict_hit1_accepted_rate']} ({test_metrics['hit1_accepted_count']}/{test_metrics['total_answerable']} total answerable)"
    else:
        strict_str = "N/A (0 answerable examples in split)"
    print(f"    - Strict Hit@1 Accepted Rate:                   {strict_str}")

    print("\n=== Proportionate Evaluation & Statistical Scope ===")
    print(
        "  1. The evaluation set comprises 45 manually authored/synthetic queries.\n"
        "  2. Metrics on this small set are preliminary indicators and do NOT represent\n"
        "     the full variance, typos, or distribution shift of real customer traffic.\n"
        "  3. Acceptance evaluates the threshold gating mechanism; retrieval evaluates ranking accuracy.\n"
        "     Neither metric guarantees that generated answers are factually grounded.\n"
        "  4. The held-out set evaluates the chosen threshold in isolation from tuning,\n"
        "     but cannot be claimed as an unbiased representation of production users without\n"
        "     continuous telemetry and logged real-traffic evaluation."
    )


if __name__ == "__main__":
    main()
