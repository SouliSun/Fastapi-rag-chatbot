"""Unit tests for retrieval pipeline, input validation, thresholding, and error handling."""

import pytest
from unittest.mock import MagicMock
from retrieval import FAQRetriever, SearchResult, RetrievalDiagnostics
from config import AppConfig


def test_empty_and_invalid_query_validation(mock_chroma_collection):
    retriever = FAQRetriever(mock_chroma_collection)

    # Empty string
    results, diag = retriever.search("")
    assert results == []
    assert "Query cannot be empty" in diag.error
    assert not diag.is_accepted

    # Whitespace only
    results, diag = retriever.search("   \n\t  ")
    assert results == []
    assert "Query cannot be empty" in diag.error

    # Non-string input
    results, diag = retriever.search(12345)
    assert results == []
    assert "Query must be a string" in diag.error


def test_unreasonable_query_length_validation(mock_chroma_collection):
    cfg = AppConfig(max_query_length=50)
    retriever = FAQRetriever(mock_chroma_collection, config=cfg)

    long_query = "word " * 20  # 100 chars
    results, diag = retriever.search(long_query)
    assert results == []
    assert "exceeds maximum allowed" in diag.error


def test_invalid_k_and_clamping(mock_chroma_collection):
    retriever = FAQRetriever(mock_chroma_collection)

    with pytest.raises(ValueError, match="k must be a positive integer"):
        retriever.validate_k(0, collection_count=10)

    with pytest.raises(ValueError, match="k must be a positive integer"):
        retriever.validate_k(-5, collection_count=10)

    # Clamping: requesting k=10 when collection has 3 should clamp to 3
    assert retriever.validate_k(10, collection_count=3) == 3


def test_empty_collection_handling(mock_chroma_collection):
    retriever = FAQRetriever(mock_chroma_collection)
    assert mock_chroma_collection.count() == 0

    results, diag = retriever.search("Valid question?")
    assert results == []
    assert diag.error == "Collection is empty"
    assert not diag.is_accepted


def test_cosine_distance_to_similarity_conversion(mock_chroma_collection):
    """Verify that Chroma distance d is converted via similarity = 1 - d."""
    mock_chroma_collection.add(
        ids=["faq_1"],
        documents=["How do I track my order?"],
        metadatas=[{"answer": "Use the tracking link."}],
    )

    # Mock query returning a known distance
    mock_chroma_collection.query = MagicMock(return_value={
        "ids": [["faq_1"]],
        "documents": [["How do I track my order?"]],
        "metadatas": [[{"answer": "Use the tracking link."}]],
        "distances": [[0.20]],  # cosine distance = 0.20 -> similarity = 0.80
    })

    retriever = FAQRetriever(mock_chroma_collection)
    results, diag = retriever.search("tracking query", k=1, threshold=0.75)

    assert len(results) == 1
    assert results[0].distance == 0.20
    assert pytest.approx(results[0].score, 0.001) == 0.80
    assert diag.is_accepted is True


def test_similarity_threshold_acceptance_and_rejection(mock_chroma_collection):
    mock_chroma_collection.add(
        ids=["faq_1"],
        documents=["What is your return policy?"],
        metadatas=[{"answer": "30 days."}],
    )
    mock_chroma_collection.query = MagicMock(return_value={
        "ids": [["faq_1"]],
        "documents": [["What is your return policy?"]],
        "metadatas": [[{"answer": "30 days."}]],
        "distances": [[0.40]],  # similarity = 0.60
    })

    retriever = FAQRetriever(mock_chroma_collection)

    # At threshold 0.50 -> accepted
    res_acc, diag_acc = retriever.search("returns", threshold=0.50)
    assert diag_acc.is_accepted is True
    assert len(retriever.filter_relevant(res_acc, threshold=0.50)) == 1

    # At threshold 0.75 -> rejected
    res_rej, diag_rej = retriever.search("returns", threshold=0.75)
    assert diag_rej.is_accepted is False
    assert len(retriever.filter_relevant(res_rej, threshold=0.75)) == 0


def test_malformed_chroma_results_handling(mock_chroma_collection):
    """Verify retriever handles missing metadata or None distances gracefully."""
    mock_chroma_collection.count = MagicMock(return_value=1)
    mock_chroma_collection.query = MagicMock(return_value={
        "ids": [["faq_x"]],
        "documents": [[None]],
        "metadatas": [[None]],  # missing metadata dict
        "distances": [[None]],  # None distance
    })

    retriever = FAQRetriever(mock_chroma_collection)
    results, diag = retriever.search("query", k=1)

    assert len(results) == 1
    assert results[0].id == "faq_x"
    assert results[0].question == ""
    assert results[0].answer == ""
    assert results[0].distance == 1.0
    assert results[0].score == 0.0

