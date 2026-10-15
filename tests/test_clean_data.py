"""Unit tests for data cleaning, normalization, and deduplication."""

import pytest
import pandas as pd
from clean_data import clean_faq_dataframe, ConflictResolution, compute_faq_id


def test_missing_required_columns():
    """Verify ValueError is raised when required columns are absent."""
    df_missing_answer = pd.DataFrame({"question": ["Where is my order?"]})
    with pytest.raises(ValueError, match="missing required column"):
        clean_faq_dataframe(df_missing_answer)

    df_missing_all = pd.DataFrame({"colA": [1], "colB": [2]})
    with pytest.raises(ValueError, match="missing required column"):
        clean_faq_dataframe(df_missing_all)


def test_null_and_blank_questions():
    """Verify null, empty, and whitespace-only questions are cleanly filtered."""
    df = pd.DataFrame({
        "question": [None, "", "   ", "\t\n", "Valid question?"],
        "answer": ["Answer 1", "Answer 2", "Answer 3", "Answer 4", "Valid answer."],
    })
    cleaned, report = clean_faq_dataframe(df)
    assert len(cleaned) == 1
    assert report.missing_values_removed == 4
    assert cleaned.iloc[0]["question"] == "Valid question?"


def test_null_and_blank_answers():
    """Verify null, empty, and whitespace-only answers are cleanly filtered."""
    df = pd.DataFrame({
        "question": ["Question 1?", "Question 2?", "Question 3?"],
        "answer": [None, "  ", "Valid answer."],
    })
    cleaned, report = clean_faq_dataframe(df)
    assert len(cleaned) == 1
    assert report.missing_values_removed == 2
    assert cleaned.iloc[0]["question"] == "Question 3?"


def test_duplicate_questions_with_different_whitespace():
    """Verify questions with different spacing, tabs, and newlines are normalized and deduplicated."""
    df = pd.DataFrame({
        "question": [
            "How do I track my order?",
            "  How  do   I track   my order? ",
            "How\tdo\nI track my order?\n",
        ],
        "answer": [
            "Use the tracking link.",
            "Use the tracking link.",
            "Use the tracking link.",
        ],
    })
    cleaned, report = clean_faq_dataframe(df)
    assert len(cleaned) == 1
    assert report.exact_duplicates_removed == 2
    assert cleaned.iloc[0]["question"] == "How do I track my order?"


def test_conflicting_answers_keep_first_policy():
    """Verify KEEP_FIRST policy logs conflict and retains first record."""
    df = pd.DataFrame({
        "question": [
            "What is your return window?",
            "What is your return window?",
        ],
        "answer": [
            "30 days from delivery.",
            "60 days from delivery.",
        ],
    })
    cleaned, report = clean_faq_dataframe(df, conflict_resolution=ConflictResolution.KEEP_FIRST)
    assert len(cleaned) == 1
    assert report.conflicts_detected == 1
    assert cleaned.iloc[0]["answer"] == "30 days from delivery."


def test_conflicting_answers_keep_last_policy():
    """Verify KEEP_LAST policy retains last record."""
    df = pd.DataFrame({
        "question": [
            "What is your return window?",
            "What is your return window?",
        ],
        "answer": [
            "30 days from delivery.",
            "60 days from delivery.",
        ],
    })
    cleaned, report = clean_faq_dataframe(df, conflict_resolution=ConflictResolution.KEEP_LAST)
    assert len(cleaned) == 1
    assert report.conflicts_detected == 1
    assert cleaned.iloc[0]["answer"] == "60 days from delivery."


def test_conflicting_answers_drop_policy():
    """Verify DROP policy drops all rows associated with conflicting answers."""
    df = pd.DataFrame({
        "question": [
            "Conflicting question?",
            "Conflicting question?",
            "Safe question?",
        ],
        "answer": [
            "Answer A",
            "Answer B",
            "Safe answer.",
        ],
    })
    cleaned, report = clean_faq_dataframe(df, conflict_resolution=ConflictResolution.DROP)
    assert len(cleaned) == 1
    assert report.conflicts_detected == 1
    assert cleaned.iloc[0]["question"] == "Safe question?"


def test_conflicting_answers_error_policy():
    """Verify ERROR policy raises an explicit ValueError when conflicts exist."""
    df = pd.DataFrame({
        "question": ["Conflicting question?", "Conflicting question?"],
        "answer": ["Answer A", "Answer B"],
    })
    with pytest.raises(ValueError, match="conflicting FAQ questions"):
        clean_faq_dataframe(df, conflict_resolution=ConflictResolution.ERROR)


def test_deterministic_output_and_ids():
    """Verify identical input in different row order yields identical IDs and sorted order."""
    df1 = pd.DataFrame({
        "question": ["Can I pay with PayPal?", "How do I return an item?"],
        "answer": ["Yes, PayPal is accepted.", "Contact support for returns."],
    })
    df2 = pd.DataFrame({
        "question": ["How do I return an item?", "Can I pay with PayPal?"],
        "answer": ["Contact support for returns.", "Yes, PayPal is accepted."],
    })

    cleaned1, _ = clean_faq_dataframe(df1)
    cleaned2, _ = clean_faq_dataframe(df2)

    pd.testing.assert_frame_equal(cleaned1, cleaned2)
    assert cleaned1["faq_id"].tolist() == cleaned2["faq_id"].tolist()
    assert compute_faq_id("How do I return an item?") == cleaned1.loc[cleaned1["question"] == "How do I return an item?", "faq_id"].iloc[0]


def test_exact_duplicates_and_conflicts_simultaneous_accounting():
    """Verify exact duplicates and conflicting answers are counted accurately in the same dataset.

    Mathematical guarantee:
    total_loaded == missing_values_removed + exact_duplicates_removed + conflict_rows_pruned + total_retained
    """
    df = pd.DataFrame({
        "question": [
            "",                       # 1. Missing
            "Valid Q1",              # 2. Original
            "Valid Q1",              # 3. Exact dup of row 2
            "Valid Q1",              # 4. Exact dup of row 2
            "Conflict Q2",           # 5. Conflict pair A
            "Conflict Q2",           # 6. Conflict pair B
            "Conflict Q2",           # 7. Exact dup of row 6
        ],
        "answer": [
            "Missing answer",
            "Answer 1",
            "Answer 1",
            "Answer 1",
            "Answer A",
            "Answer B",
            "Answer B",
        ],
    })

    # Test KEEP_FIRST policy
    cleaned_keep_first, rep_kf = clean_faq_dataframe(df, conflict_resolution=ConflictResolution.KEEP_FIRST)
    assert rep_kf.total_loaded == 7
    assert rep_kf.missing_values_removed == 1
    assert rep_kf.exact_duplicates_removed == 3  # 2 for Q1, 1 for (Q2, Answer B)
    assert rep_kf.conflicts_detected == 1
    assert rep_kf.conflict_rows_pruned == 1      # 1 distinct conflicting row pruned
    assert rep_kf.total_retained == 2
    assert rep_kf.total_loaded == (
        rep_kf.missing_values_removed
        + rep_kf.exact_duplicates_removed
        + rep_kf.conflict_rows_pruned
        + rep_kf.total_retained
    )

    # Test DROP policy
    cleaned_drop, rep_drop = clean_faq_dataframe(df, conflict_resolution=ConflictResolution.DROP)
    assert rep_drop.total_loaded == 7
    assert rep_drop.missing_values_removed == 1
    assert rep_drop.exact_duplicates_removed == 3
    assert rep_drop.conflicts_detected == 1
    assert rep_drop.conflict_rows_pruned == 2    # both distinct conflict rows dropped
    assert rep_drop.total_retained == 1
    assert rep_drop.total_loaded == (
        rep_drop.missing_values_removed
        + rep_drop.exact_duplicates_removed
        + rep_drop.conflict_rows_pruned
        + rep_drop.total_retained
    )

