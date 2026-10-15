"""Shared pytest fixtures and mock utilities for the RAG test suite."""

import os
import shutil
import tempfile
import pytest
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

from config import AppConfig
from clean_data import clean_faq_dataframe


class MockEmbeddingFunction:
    """Fast, deterministic dummy embedding function for unit tests."""
    def name(self) -> str:
        return "mock_embedding_function"

    def __call__(self, input: list[str]) -> list[list[float]]:
        # Generate predictable 4-dimensional normalized vectors
        results = []
        for text in input:
            val = float(len(text) % 10 + 1)
            results.append([val / 10.0, 0.5, 0.2, 0.1])
        return results


@pytest.fixture
def mock_embed_fn():
    return MockEmbeddingFunction()


@pytest.fixture
def sample_faq_df():
    """Deterministic sample dataframe."""
    data = {
        "question": [
            "How do I track my order?",
            "What is your return policy?",
            "Do you offer international shipping?",
        ],
        "answer": [
            "Log in and go to Order History to track your package.",
            "Items can be returned within 30 days for a full refund.",
            "Yes, we ship to select international destinations.",
        ],
    }
    df, _ = clean_faq_dataframe(pd.DataFrame(data))
    return df


@pytest.fixture
def temp_chroma_dir():
    """Create a temporary directory for ChromaDB and clean it up after tests."""
    temp_dir = tempfile.mkdtemp(prefix="test_chroma_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_chroma_collection(temp_chroma_dir, mock_embed_fn):
    client = chromadb.PersistentClient(path=temp_chroma_dir)
    return client.create_collection(
        name="test_faqs",
        embedding_function=mock_embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

