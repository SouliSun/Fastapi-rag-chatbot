"""Unit tests for index synchronization, fingerprinting, and ChromaDB lifecycle management."""

import pytest
import pandas as pd
from unittest.mock import patch

from clean_data import clean_faq_dataframe
from ingest import (
    compute_dataset_fingerprint,
    ensure_indexed,
    read_index_manifest,
    rebuild_index,
)


@pytest.fixture
def mock_embedding_env(monkeypatch, mock_embed_fn):
    """Patch get_embedding_function to return fast mock embedder."""
    monkeypatch.setattr("ingest.get_embedding_function", lambda model_name=None: mock_embed_fn)


def test_first_time_ingestion(temp_chroma_dir, sample_faq_df, mock_embedding_env):
    """Verify first-time ingestion creates collection and writes manifest."""
    col = ensure_indexed(faq_df=sample_faq_df, db_path=temp_chroma_dir, collection_name="test_col")
    assert col.count() == len(sample_faq_df)

    manifest = read_index_manifest(temp_chroma_dir)
    assert manifest is not None
    assert manifest["record_count"] == len(sample_faq_df)
    assert manifest["collection_name"] == "test_col"


def test_repeated_ingestion_idempotence(temp_chroma_dir, sample_faq_df, mock_embedding_env):
    """Verify repeated ingestion with unchanged data is a no-op."""
    col1 = ensure_indexed(faq_df=sample_faq_df, db_path=temp_chroma_dir, collection_name="test_col")
    fp1 = read_index_manifest(temp_chroma_dir)["fingerprint"]

    # Re-running ensure_indexed should not rebuild
    with patch("ingest.rebuild_index") as mock_rebuild:
        col2 = ensure_indexed(faq_df=sample_faq_df, db_path=temp_chroma_dir, collection_name="test_col")
        mock_rebuild.assert_not_called()

    assert col2.count() == col1.count()
    fp2 = read_index_manifest(temp_chroma_dir)["fingerprint"]
    assert fp1 == fp2


def test_updated_faq_answer_triggers_reindex(temp_chroma_dir, sample_faq_df, mock_embedding_env):
    """Verify that changing an answer alters fingerprint and triggers index synchronization."""
    ensure_indexed(faq_df=sample_faq_df, db_path=temp_chroma_dir, collection_name="test_col")
    fp_before = read_index_manifest(temp_chroma_dir)["fingerprint"]

    # Modify an answer
    modified_df = sample_faq_df.copy()
    modified_df.loc[0, "answer"] = "MODIFIED ANSWER: We only ship on weekdays."

    col_after = ensure_indexed(faq_df=modified_df, db_path=temp_chroma_dir, collection_name="test_col")
    fp_after = read_index_manifest(temp_chroma_dir)["fingerprint"]

    assert fp_before != fp_after
    assert col_after.count() == len(modified_df)


def test_added_faq_triggers_reindex(temp_chroma_dir, sample_faq_df, mock_embedding_env):
    """Verify adding a new FAQ triggers index synchronization."""
    ensure_indexed(faq_df=sample_faq_df, db_path=temp_chroma_dir, collection_name="test_col")

    new_row = pd.DataFrame([{"question": "Do you sell gift cards?", "answer": "Yes, digital cards.", "faq_id": "faq_giftcard"}])
    augmented_df = pd.concat([sample_faq_df, new_row], ignore_index=True)

    col = ensure_indexed(faq_df=augmented_df, db_path=temp_chroma_dir, collection_name="test_col")
    assert col.count() == len(augmented_df)


def test_deleted_faq_removes_stale_data(temp_chroma_dir, sample_faq_df, mock_embedding_env):
    """Verify deleting a record does not leave orphaned documents in the collection."""
    ensure_indexed(faq_df=sample_faq_df, db_path=temp_chroma_dir, collection_name="test_col")
    assert sample_faq_df.shape[0] == 3

    # Delete 1 record
    reduced_df = sample_faq_df.iloc[:2].copy()
    col = ensure_indexed(faq_df=reduced_df, db_path=temp_chroma_dir, collection_name="test_col")

    assert col.count() == 2
    # Verify the deleted question is completely gone from Chroma
    deleted_id = sample_faq_df.iloc[2]["faq_id"]
    existing_ids = col.get()["ids"]
    assert deleted_id not in existing_ids


def test_reordered_rows_preserves_fingerprint(sample_faq_df):
    """Verify fingerprint is invariant to DataFrame row permutations."""
    reordered_df = sample_faq_df.sample(frac=1.0, random_state=123).reset_index(drop=True)
    fp1 = compute_dataset_fingerprint(sample_faq_df, "test-model")
    fp2 = compute_dataset_fingerprint(reordered_df, "test-model")
    assert fp1 == fp2


def test_model_change_triggers_reindex(temp_chroma_dir, sample_faq_df, mock_embedding_env):
    """Verify switching the embedding model triggers reindexing."""
    ensure_indexed(faq_df=sample_faq_df, db_path=temp_chroma_dir, collection_name="test_col", embed_model="model_v1")
    fp1 = read_index_manifest(temp_chroma_dir)["fingerprint"]

    ensure_indexed(faq_df=sample_faq_df, db_path=temp_chroma_dir, collection_name="test_col", embed_model="model_v2")
    fp2 = read_index_manifest(temp_chroma_dir)["fingerprint"]

    assert fp1 != fp2

