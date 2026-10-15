"""Unit tests for conversation memory and bounded LRU store."""

from __future__ import annotations

import concurrent.futures
import time
import pytest

from memory import (
    ConversationSession,
    ConversationStore,
    ConversationTurn,
    generate_conversation_id,
    validate_conversation_id,
)


def test_conversation_id_validation():
    """Test valid and invalid conversation IDs."""
    assert validate_conversation_id("conv-123_abc") is True
    assert validate_conversation_id("session1") is True
    assert validate_conversation_id("a" * 64) is True

    # Invalid: empty, whitespace, special chars, too long
    assert validate_conversation_id("") is False
    assert validate_conversation_id("   ") is False
    assert validate_conversation_id("conv/123") is False
    assert validate_conversation_id("conv!@#") is False
    assert validate_conversation_id("conv 123") is False
    assert validate_conversation_id("a" * 65) is False
    assert validate_conversation_id(None) is False  # type: ignore


def test_conversation_id_generation():
    """Test generating conversation IDs."""
    ids = [generate_conversation_id() for _ in range(100)]
    assert len(set(ids)) == 100
    for cid in ids:
        assert cid.startswith("conv_")
        assert validate_conversation_id(cid) is True
        assert len(cid) <= 64


def test_conversation_session_turn_bounding():
    """Test that a conversation session bounds turns in FIFO order."""
    session = ConversationSession(conversation_id="conv_test_1", max_turns=3)
    assert len(session.get_history()) == 0

    session.add_turn("q1", "a1", fallback=False)
    session.add_turn("q2", "a2", fallback=False)
    session.add_turn("q3", "a3", fallback=False)
    assert len(session.get_history()) == 3
    assert session.get_history()[0].user_query == "q1"
    assert session.get_history()[2].user_query == "q3"

    # Adding a 4th turn should evict q1
    session.add_turn("q4", "a4", fallback=False)
    history = session.get_history()
    assert len(history) == 3
    assert history[0].user_query == "q2"
    assert history[1].user_query == "q3"
    assert history[2].user_query == "q4"


def test_conversation_session_ttl_expiration():
    """Test that session expires after TTL."""
    session = ConversationSession(conversation_id="conv_test_2", max_turns=5)
    # Manually backdate last_accessed_at
    session.last_accessed_at = time.time() - 100.0

    assert session.is_expired(ttl_seconds=50.0) is True
    assert session.is_expired(ttl_seconds=200.0) is False


def test_conversation_store_get_or_create_behavior():
    """Test store get_or_create with various input conditions."""
    store = ConversationStore(max_conversations=5, max_turns_per_conversation=3, ttl_seconds=60.0)

    # 1. No ID supplied -> generates a new one
    session1, cid1 = store.get_or_create(None)
    assert cid1.startswith("conv_")
    assert session1.conversation_id == cid1

    # 2. Existing ID supplied -> returns existing session
    session1_again, cid1_again = store.get_or_create(cid1)
    assert cid1_again == cid1
    assert session1_again is session1

    # 3. Valid but unknown ID supplied -> creates fresh session with that ID
    session2, cid2 = store.get_or_create("custom-session-99")
    assert cid2 == "custom-session-99"
    assert session2.conversation_id == "custom-session-99"

    # 4. Invalid ID supplied -> raises ValueError
    with pytest.raises(ValueError, match="Invalid conversation_id"):
        store.get_or_create("invalid/id!@#")

    with pytest.raises(ValueError, match="exceeds maximum allowed length"):
        store.get_or_create("x" * 65)


def test_conversation_store_lru_eviction():
    """Test that the store evicts the least recently accessed conversation when capacity is reached."""
    store = ConversationStore(max_conversations=3, max_turns_per_conversation=2, ttl_seconds=300.0)

    # Fill store with 3 sessions
    _, c1 = store.get_or_create("c1")
    _, c2 = store.get_or_create("c2")
    _, c3 = store.get_or_create("c3")
    assert store.active_count() == 3

    # Access c1 so c2 becomes the least recently used
    store.get_or_create("c1")

    # Add a 4th session -> c2 should be evicted
    _, c4 = store.get_or_create("c4")
    assert store.active_count() == 3

    # c1, c3, c4 should exist; c2 should have been evicted
    # Calling get_or_create on c2 now should create a brand new empty session
    s2_new, _ = store.get_or_create("c2")
    assert len(s2_new.get_history()) == 0


def test_conversation_store_ttl_eviction_on_access():
    """Test that expired sessions are evicted when accessed."""
    store = ConversationStore(max_conversations=10, max_turns_per_conversation=5, ttl_seconds=0.1)

    session, cid = store.get_or_create("temp-conv")
    store.record_turn(cid, "Hello", "Hi there", fallback=False)
    assert len(session.get_history()) == 1

    # Wait for TTL to pass
    time.sleep(0.15)

    # Retrieving temp-conv should detect expiration, evict the old session, and return a fresh one
    fresh_session, fresh_cid = store.get_or_create("temp-conv")
    assert fresh_cid == "temp-conv"
    assert len(fresh_session.get_history()) == 0


def test_conversation_store_thread_safety():
    """Test that concurrent access to ConversationStore does not corrupt state."""
    store = ConversationStore(max_conversations=50, max_turns_per_conversation=5, ttl_seconds=60.0)

    def worker(worker_id: int):
        cid = f"worker_{worker_id % 10}"
        session, _ = store.get_or_create(cid)
        store.record_turn(cid, f"query from {worker_id}", f"answer for {worker_id}", fallback=False)
        return len(session.get_history())

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(worker, i) for i in range(100)]
        results = [f.result() for f in futures]

    assert len(results) == 100
    assert store.active_count() <= 10
