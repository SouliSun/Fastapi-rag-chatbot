"""In-memory, bounded short-term conversation store for multi-turn RAG chat.

Provides thread-safe session tracking, LRU eviction, and turn-level limits.
Conversation state is ephemeral and kept strictly in-memory (no disk/database persistence).
"""

from __future__ import annotations

import logging
import re
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# Validates conversation IDs: 1 to 64 alphanumeric characters, underscores, or hyphens.
CONVERSATION_ID_REGEX = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


def validate_conversation_id(conversation_id: str) -> bool:
    """Check if conversation_id is valid without raising exceptions."""
    if not isinstance(conversation_id, str):
        return False
    cleaned = conversation_id.strip()
    return bool(cleaned and len(cleaned) <= 64 and CONVERSATION_ID_REGEX.match(cleaned))


def generate_conversation_id() -> str:
    """Generate a random, URL-safe conversation ID."""
    return f"conv_{uuid.uuid4().hex[:16]}"


@dataclass(frozen=True)
class ConversationTurn:
    """A single turn in an ongoing customer inquiry."""

    user_query: str
    assistant_answer: str
    timestamp: float = field(default_factory=time.time)
    fallback: bool = False


class ConversationSession:
    """Bounded, chronological history container for a single conversation."""

    def __init__(self, conversation_id: str, max_turns: int = 5):
        self.conversation_id = conversation_id
        self.max_turns = max_turns
        self.turns: List[ConversationTurn] = []
        self.last_accessed_at: float = time.time()

    def add_turn(
        self,
        user_query: str,
        assistant_answer: str,
        fallback: bool = False,
    ) -> None:
        """Append a new interaction turn and evict oldest turns beyond capacity."""
        self.last_accessed_at = time.time()
        turn = ConversationTurn(
            user_query=user_query,
            assistant_answer=assistant_answer,
            timestamp=self.last_accessed_at,
            fallback=fallback,
        )
        self.turns.append(turn)
        if len(self.turns) > self.max_turns:
            # Drop oldest turns to keep strictly bounded
            self.turns = self.turns[-self.max_turns :]

    def get_history(self) -> List[ConversationTurn]:
        """Return a copy of recent conversation turns and update access time."""
        self.last_accessed_at = time.time()
        return list(self.turns)

    def is_expired(self, ttl_seconds: float) -> bool:
        """Check whether the session has exceeded inactivity TTL."""
        return (time.time() - self.last_accessed_at) > ttl_seconds


class ConversationStore:
    """Thread-safe, bounded LRU conversation memory manager."""

    def __init__(
        self,
        max_conversations: int = 1000,
        max_turns_per_conversation: int = 5,
        ttl_seconds: float = 3600.0,
    ):
        self.max_conversations = max_conversations
        self.max_turns_per_conversation = max_turns_per_conversation
        self.ttl_seconds = ttl_seconds
        self._store: OrderedDict[str, ConversationSession] = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def validate_conversation_id(conversation_id: str) -> str:
        """Validate format, allowed characters, and length bounds.

        Raises:
            ValueError: If conversation_id is non-string, empty, too long, or contains illegal characters.
        """
        if not isinstance(conversation_id, str):
            raise ValueError(
                f"Conversation ID must be a string, got {type(conversation_id).__name__}."
            )
        cleaned = conversation_id.strip()
        if not cleaned:
            raise ValueError("Conversation ID cannot be empty or whitespace-only.")
        if len(cleaned) > 64:
            raise ValueError("Conversation ID exceeds maximum allowed length of 64 characters.")
        if not CONVERSATION_ID_REGEX.match(cleaned):
            raise ValueError(
                "Invalid conversation_id format. Must contain only letters, numbers, "
                "hyphens, and underscores."
            )
        return cleaned

    @staticmethod
    def generate_conversation_id() -> str:
        """Generate a random, URL-safe conversation ID."""
        return f"conv_{uuid.uuid4().hex[:16]}"

    def _purge_expired_locked(self, now: float) -> None:
        """Purge expired sessions while holding lock."""
        expired_keys = [
            k for k, sess in self._store.items()
            if (now - sess.last_accessed_at) > self.ttl_seconds
        ]
        for k in expired_keys:
            del self._store[k]

    def get_or_create(
        self, conversation_id: Optional[str] = None
    ) -> Tuple[ConversationSession, str]:
        """Retrieve an existing session, or create a fresh one if absent or expired.

        Returns:
            Tuple of (session, validated_or_generated_conversation_id)
        """
        with self._lock:
            now = time.time()
            self._purge_expired_locked(now)

            if conversation_id:
                cid = self.validate_conversation_id(conversation_id)
                if cid in self._store:
                    session = self._store[cid]
                    self._store.move_to_end(cid)
                    session.last_accessed_at = now
                    return session, cid
            else:
                cid = self.generate_conversation_id()

            # Enforce global capacity via LRU eviction of oldest session
            while len(self._store) >= self.max_conversations:
                evicted_id, _ = self._store.popitem(last=False)
                logger.debug("Evicted oldest conversation session: %s", evicted_id)

            session = ConversationSession(
                conversation_id=cid,
                max_turns=self.max_turns_per_conversation,
            )
            self._store[cid] = session
            return session, cid

    def get_history(self, conversation_id: str) -> List[ConversationTurn]:
        """Fetch chronological turns for an active conversation."""
        with self._lock:
            session = self._store.get(conversation_id)
            if session and not session.is_expired(self.ttl_seconds):
                self._store.move_to_end(conversation_id)
                return session.get_history()
            return []

    def record_turn(
        self,
        conversation_id: str,
        user_query: str,
        assistant_answer: str,
        fallback: bool = False,
    ) -> None:
        """Record a completed interaction turn into the session."""
        with self._lock:
            session = self._store.get(conversation_id)
            if session is None:
                # Session expired or was never created: create now
                while len(self._store) >= self.max_conversations:
                    self._store.popitem(last=False)
                session = ConversationSession(
                    conversation_id=conversation_id,
                    max_turns=self.max_turns_per_conversation,
                )
                self._store[conversation_id] = session

            self._store.move_to_end(conversation_id)
            session.add_turn(
                user_query=user_query,
                assistant_answer=assistant_answer,
                fallback=fallback,
            )

    def clear(self) -> None:
        """Purge all sessions from memory (used for tests)."""
        with self._lock:
            self._store.clear()

    def active_count(self) -> int:
        """Return the number of active sessions in memory."""
        with self._lock:
            return len(self._store)

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)
