"""Regression, security, secret protection, and interface stability tests."""

from unittest.mock import MagicMock, patch
import pytest

from config import AppConfig, get_config
import ingest
from retrieval import FAQRetriever, SearchResult


def test_api_keys_are_masked_in_repr_and_logs():
    """Verify raw API keys are never exposed in string representations."""
    cfg = AppConfig(groq_api_key="gsk_1234567890abcdef1234567890abcdef")
    cfg_str = str(cfg)
    cfg_repr = repr(cfg)

    assert "gsk_1234567890abcdef1234567890abcdef" not in cfg_str
    assert "gsk_1234567890abcdef1234567890abcdef" not in cfg_repr
    assert "gsk_...cdef" in cfg_repr


def test_public_search_interface_backward_compatibility():
    """Verify the signature and return types of ingest.search remain stable."""
    # Mock default retriever
    mock_retriever = MagicMock()
    mock_retriever.search.return_value = (
        [
            SearchResult(
                id="faq_test",
                question="Can I order by phone?",
                answer="No, please use the website.",
                score=0.88,
                distance=0.12,
            )
        ],
        MagicMock(),
    )

    with patch("ingest.get_default_retriever", return_value=mock_retriever):
        top_ans, results = ingest.search("Can I order by phone?", k=1, threshold=0.75)

    assert top_ans == "No, please use the website."
    assert isinstance(results, list)
    assert results[0]["id"] == "faq_test"
    assert results[0]["score"] == 0.88
    assert results[0]["answer"] == "No, please use the website."


def test_public_generate_answer_interface_backward_compatibility():
    """Verify ingest.generate_answer returns expected dict keys."""
    mock_retriever = MagicMock()
    mock_retriever.search.return_value = (
        [
            SearchResult(
                id="faq_test",
                question="Can I order by phone?",
                answer="No, please use the website.",
                score=0.88,
                distance=0.12,
            )
        ],
        MagicMock(),
    )
    mock_retriever.filter_relevant.return_value = [
        SearchResult(
            id="faq_test",
            question="Can I order by phone?",
            answer="No, please use the website.",
            score=0.88,
            distance=0.12,
        )
    ]

    mock_generator = MagicMock()
    mock_generator.generate.return_value = MagicMock(
        answer="No, phone orders are not supported.",
        fallback=False,
        latency_ms=120.0,
        prompt_tokens=45,
        completion_tokens=15,
    )

    with patch("ingest.get_default_retriever", return_value=mock_retriever), patch(
        "ingest.get_default_generator", return_value=mock_generator
    ):
        out = ingest.generate_answer("Can I order by phone?", k=3, threshold=0.75)

    assert "answer" in out
    assert "results" in out
    assert "contexts" in out
    assert "fallback" in out
    assert out["answer"] == "No, phone orders are not supported."
    assert out["fallback"] is False


def test_health_check_function():
    """Verify health_check() function returns healthy status structure."""
    with patch("ingest.ensure_indexed") as mock_ensure, patch(
        "ingest.FAQRetriever"
    ) as mock_retriever_cls:
        mock_coll = MagicMock()
        mock_coll.count.return_value = 89
        mock_ensure.return_value = mock_coll

        mock_inst = MagicMock()
        mock_inst.search.return_value = ([MagicMock()], MagicMock())
        mock_retriever_cls.return_value = mock_inst

        h = ingest.health_check()

    assert h["status"] == "healthy"
    assert h["indexed_faqs"] == 89
    assert h["test_retrieval_ok"] is True


def test_lazy_df_proxy_defers_execution_until_accessed():
    """Verify that _LazyDFProxy does not call load_and_clean_faqs on instantiation."""
    from clean_data import _LazyDFProxy
    import clean_data

    # Save previous cached state
    orig_cached = clean_data._CACHED_DF
    try:
        clean_data._CACHED_DF = None
        with patch("clean_data.load_and_clean_faqs") as mock_loader:
            mock_loader.return_value = (MagicMock(), MagicMock())
            proxy = _LazyDFProxy()
            # Creating proxy must not trigger data loading
            mock_loader.assert_not_called()

            # Accessing an attribute on the proxy triggers data loading
            _ = len(proxy)
            mock_loader.assert_called_once()
    finally:
        clean_data._CACHED_DF = orig_cached

