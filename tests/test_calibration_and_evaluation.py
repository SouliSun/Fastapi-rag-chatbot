"""Unit tests for dataset partitioning, threshold calibration, and evaluation split integrity."""

import pytest
from calibrate_threshold import partition_dev_and_test_sets, CategorySplitReport


@pytest.fixture
def sample_eval_items():
    """Synthetic dataset with mixed category sizes (large, medium, and single-item)."""
    items = []
    # 10 single-hop answerable items
    for i in range(10):
        items.append({
            "id": f"sh_{i}",
            "query_type": "single_hop",
            "answerable": True,
            "user_input": f"Single hop question {i}",
        })
    # 4 multi-hop answerable items
    for i in range(4):
        items.append({
            "id": f"mh_{i}",
            "query_type": "multi_hop",
            "answerable": True,
            "user_input": f"Multi hop question {i}",
        })
    # 3 unanswerable items
    for i in range(3):
        items.append({
            "id": f"un_{i}",
            "query_type": "unanswerable",
            "answerable": False,
            "user_input": f"Unanswerable question {i}",
        })
    # 1 out-of-scope item (single-item category)
    items.append({
        "id": "oos_0",
        "query_type": "out_of_scope",
        "answerable": False,
        "user_input": "Out of scope movie question",
    })
    return items


def test_partition_determinism(sample_eval_items):
    """Verify that identical random seed produces bitwise identical dev and test splits."""
    dev1, test1, rep1 = partition_dev_and_test_sets(sample_eval_items, dev_ratio=0.7, seed=42)
    dev2, test2, rep2 = partition_dev_and_test_sets(sample_eval_items, dev_ratio=0.7, seed=42)

    assert [x["id"] for x in dev1] == [x["id"] for x in dev2]
    assert [x["id"] for x in test1] == [x["id"] for x in test2]
    assert len(rep1) == len(rep2)


def test_partition_ratio_and_no_data_leakage_or_loss(sample_eval_items):
    """Verify total items are conserved, no overlap between dev and test, and ratio is respected."""
    total_expected = len(sample_eval_items)  # 18 items
    dev, test, reports = partition_dev_and_test_sets(sample_eval_items, dev_ratio=0.7, seed=42)

    dev_ids = {x["id"] for x in dev}
    test_ids = {x["id"] for x in test}

    # No data loss
    assert len(dev) + len(test) == total_expected
    assert dev_ids | test_ids == {x["id"] for x in sample_eval_items}

    # Strict separation: zero overlap
    assert len(dev_ids & test_ids) == 0

    # Approx 70% in dev
    assert len(dev) >= int(total_expected * 0.65)


def test_partition_handles_single_item_category_without_crash(sample_eval_items):
    """Verify single-item category (out_of_scope) is flagged, placed in dev, and test gets 0."""
    dev, test, reports = partition_dev_and_test_sets(sample_eval_items, dev_ratio=0.7, seed=42)

    oos_reports = [r for r in reports if r.category == "out_of_scope_una"]
    assert len(oos_reports) == 1
    rep = oos_reports[0]

    assert rep.is_too_small_to_split is True
    assert rep.total_count == 1
    assert rep.dev_count == 1
    assert rep.test_count == 0
    assert "only 1 example" in rep.warning

    # Check actual items
    dev_oos = [x for x in dev if x["query_type"] == "out_of_scope"]
    test_oos = [x for x in test if x["query_type"] == "out_of_scope"]
    assert len(dev_oos) == 1
    assert len(test_oos) == 0


def test_partition_category_balance_for_multi_item_strata(sample_eval_items):
    """Verify every stratum with >= 2 items gets at least 1 example in both dev and test."""
    dev, test, reports = partition_dev_and_test_sets(sample_eval_items, dev_ratio=0.7, seed=42)

    for rep in reports:
        if rep.total_count >= 2:
            assert rep.dev_count >= 1, f"Category {rep.category} had 0 dev items"
            assert rep.test_count >= 1, f"Category {rep.category} had 0 test items"
            assert rep.is_too_small_to_split is False


def test_partition_invalid_ratio_raises_value_error(sample_eval_items):
    """Verify invalid ratios <= 0 or >= 1 raise ValueError."""
    with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
        partition_dev_and_test_sets(sample_eval_items, dev_ratio=0.0)

    with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
        partition_dev_and_test_sets(sample_eval_items, dev_ratio=1.0)


def test_accepted_wrong_retrieval_counted_for_coverage_not_hit():
    """Verify that an answerable query clearing threshold with wrong FAQ counts for coverage, but NOT as a retrieval hit."""
    from calibrate_threshold import compute_metrics_from_cache

    # Mock eval cache entry: answerable query, high similarity score 0.88, but matched_rank is None (wrong FAQ retrieved)
    eval_cache = [
        {
            "id": "q1",
            "query": "How do I return?",
            "query_type": "single_hop",
            "answerable": True,
            "top_score": 0.88,
            "matched_rank": None,  # Wrong FAQ retrieved!
            "hit_at_1": False,
            "hit_at_k": False,
            "reciprocal_rank": 0.0,
        }
    ]

    metrics = compute_metrics_from_cache(eval_cache, threshold=0.75)

    # Acceptance gating: score 0.88 >= 0.75 -> accepted!
    assert metrics["accepted_answerable_count"] == 1
    assert metrics["answerable_coverage"] == 1.0
    assert metrics["false_refusal_count"] == 0

    # Retrieval correctness: wrong FAQ retrieved -> NOT a hit!
    assert metrics["hit1_among_accepted"] == 0.0
    assert metrics["hitk_among_accepted"] == 0.0
    assert metrics["mrr_among_accepted"] == 0.0
    assert metrics["strict_hit1_accepted_rate"] == 0.0


def test_hit_at_k_and_mrr_calculation_against_source_questions():
    """Verify Hit@1, Hit@k, and MRR calculations against ground-truth source FAQs."""
    from calibrate_threshold import compute_metrics_from_cache

    eval_cache = [
        # Query 1: matched at rank 1 (RR = 1.0)
        {
            "id": "q1",
            "query": "Q1",
            "query_type": "single_hop",
            "answerable": True,
            "top_score": 0.90,
            "matched_rank": 1,
            "hit_at_1": True,
            "hit_at_k": True,
            "reciprocal_rank": 1.0,
        },
        # Query 2: matched at rank 2 (RR = 0.5)
        {
            "id": "q2",
            "query": "Q2",
            "query_type": "multi_hop",
            "answerable": True,
            "top_score": 0.85,
            "matched_rank": 2,
            "hit_at_1": False,
            "hit_at_k": True,
            "reciprocal_rank": 0.5,
        },
        # Query 3: not in top-k (RR = 0.0)
        {
            "id": "q3",
            "query": "Q3",
            "query_type": "single_hop",
            "answerable": True,
            "top_score": 0.80,
            "matched_rank": None,
            "hit_at_1": False,
            "hit_at_k": False,
            "reciprocal_rank": 0.0,
        },
    ]

    metrics = compute_metrics_from_cache(eval_cache, threshold=0.70)

    # Baseline retrieval quality across all 3 answerable queries
    assert metrics["baseline_hit_at_1"] == pytest.approx(1 / 3, rel=1e-2)
    assert metrics["baseline_hit_at_k"] == pytest.approx(2 / 3, rel=1e-2)
    assert metrics["baseline_mrr"] == pytest.approx((1.0 + 0.5 + 0.0) / 3, rel=1e-2)

    # All 3 cleared threshold 0.70:
    assert metrics["hit1_among_accepted"] == pytest.approx(1 / 3, rel=1e-2)
    assert metrics["hitk_among_accepted"] == pytest.approx(2 / 3, rel=1e-2)
    assert metrics["mrr_among_accepted"] == pytest.approx(0.5, rel=1e-2)
    assert metrics["strict_hit1_accepted_rate"] == pytest.approx(1 / 3, rel=1e-2)


def test_absent_categories_avoid_misleading_divide_by_zero():
    """Verify that splits with 0 unanswerable or 0 answerable queries avoid divide-by-zero or misleading 1.0 rates."""
    from calibrate_threshold import compute_metrics_from_cache

    # Case A: Only answerable queries (0 unanswerable queries)
    cache_ans_only = [
        {
            "id": "q1",
            "query": "Q1",
            "query_type": "single_hop",
            "answerable": True,
            "top_score": 0.80,
            "matched_rank": 1,
            "hit_at_1": True,
            "hit_at_k": True,
            "reciprocal_rank": 1.0,
        }
    ]
    m_ans = compute_metrics_from_cache(cache_ans_only, threshold=0.75)
    # Must NOT report 1.0 (which would falsely imply 100% rejection) or raise error
    assert m_ans["unanswerable_rejection_rate"] is None
    assert m_ans["unanswerable_combined_rejection_rate"] is None
    assert m_ans["total_unanswerable"] == 0
    assert m_ans["unanswerable_leakage_count"] == 0

    # Case B: Only unanswerable queries (0 answerable queries)
    cache_unans_only = [
        {
            "id": "q2",
            "query": "Q2",
            "query_type": "out_of_scope",
            "answerable": False,
            "top_score": 0.60,
            "matched_rank": None,
            "hit_at_1": False,
            "hit_at_k": False,
            "reciprocal_rank": 0.0,
        }
    ]
    m_unans = compute_metrics_from_cache(cache_unans_only, threshold=0.75)
    assert m_unans["answerable_coverage"] is None
    assert m_unans["total_answerable"] == 0
    assert m_unans["unanswerable_rejection_rate"] == 1.0  # 0.60 < 0.75 -> safely rejected
    # 0 answerable queries accepted -> denominator is 0 -> None (N/A)
    assert m_unans["hit1_among_accepted"] is None
    assert m_unans["hitk_among_accepted"] is None
    assert m_unans["mrr_among_accepted"] is None


def test_accepted_only_metrics_are_none_when_no_answerable_pass_threshold():
    """Verify accepted retrieval metrics return None (N/A) when threshold rejects all answerable queries."""
    from calibrate_threshold import compute_metrics_from_cache

    eval_cache = [
        {
            "id": "q1",
            "query": "Q1",
            "query_type": "single_hop",
            "answerable": True,
            "top_score": 0.70,  # Below threshold
            "matched_rank": 1,
            "hit_at_1": True,
            "hit_at_k": True,
            "reciprocal_rank": 1.0,
        }
    ]

    metrics = compute_metrics_from_cache(eval_cache, threshold=0.85)
    assert metrics["accepted_answerable_count"] == 0
    assert metrics["false_refusal_count"] == 1
    assert metrics["hit1_among_accepted"] is None
    assert metrics["hitk_among_accepted"] is None
    assert metrics["mrr_among_accepted"] is None
    assert metrics["strict_hit1_accepted_rate"] == 0.0


def test_accepted_only_metrics_are_zero_when_accepted_exist_with_no_hits():
    """Verify accepted retrieval metrics return 0.0 (true zero) when queries clear threshold but none match."""
    from calibrate_threshold import compute_metrics_from_cache

    eval_cache = [
        {
            "id": "q1",
            "query": "Q1",
            "query_type": "single_hop",
            "answerable": True,
            "top_score": 0.85,  # Above threshold
            "matched_rank": None,  # No match in top-k
            "hit_at_1": False,
            "hit_at_k": False,
            "reciprocal_rank": 0.0,
        }
    ]

    metrics = compute_metrics_from_cache(eval_cache, threshold=0.80)
    assert metrics["accepted_answerable_count"] == 1
    assert metrics["hit1_among_accepted"] == 0.0
    assert metrics["hitk_among_accepted"] == 0.0
    assert metrics["mrr_among_accepted"] == 0.0
    assert metrics["strict_hit1_accepted_rate"] == 0.0


def test_reporting_when_out_of_scope_or_unanswerable_categories_absent():
    """Verify category absence reports None for absent categories and exact rates for present ones."""
    from calibrate_threshold import compute_metrics_from_cache

    # Split with 0 out_of_scope and 2 in-domain unanswerable
    eval_cache_no_oos = [
        {
            "id": "un1",
            "query": "What is CEO name?",
            "query_type": "unanswerable",
            "answerable": False,
            "top_score": 0.50,
            "matched_rank": None,
            "hit_at_1": False,
            "hit_at_k": False,
            "reciprocal_rank": 0.0,
        },
        {
            "id": "un2",
            "query": "What is phone number?",
            "query_type": "unanswerable",
            "answerable": False,
            "top_score": 0.80,  # leaks through at 0.70
            "matched_rank": None,
            "hit_at_1": False,
            "hit_at_k": False,
            "reciprocal_rank": 0.0,
        },
    ]

    m_no_oos = compute_metrics_from_cache(eval_cache_no_oos, threshold=0.70)
    assert m_no_oos["out_of_scope_count"] == 0
    assert m_no_oos["out_of_scope_rejection_rate"] is None  # N/A
    assert m_no_oos["unanswerable_in_domain_count"] == 2
    assert m_no_oos["unanswerable_in_domain_rejection_rate"] == 0.5  # 1/2 rejected
    assert m_no_oos["unanswerable_combined_rejection_rate"] == 0.5

    # Split with 2 out_of_scope and 0 in-domain unanswerable
    eval_cache_no_unans = [
        {
            "id": "oos1",
            "query": "Movie recommendation?",
            "query_type": "out_of_scope",
            "answerable": False,
            "top_score": 0.40,
            "matched_rank": None,
            "hit_at_1": False,
            "hit_at_k": False,
            "reciprocal_rank": 0.0,
        }
    ]

    m_no_unans = compute_metrics_from_cache(eval_cache_no_unans, threshold=0.70)
    assert m_no_unans["unanswerable_in_domain_count"] == 0
    assert m_no_unans["unanswerable_in_domain_rejection_rate"] is None  # N/A
    assert m_no_unans["out_of_scope_count"] == 1
    assert m_no_unans["out_of_scope_rejection_rate"] == 1.0  # 1/1 rejected
    assert m_no_unans["unanswerable_combined_rejection_rate"] == 1.0

