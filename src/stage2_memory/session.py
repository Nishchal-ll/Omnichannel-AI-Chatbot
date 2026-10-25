"""
src/stage2_memory/session.py
────────────────────────────
SessionBuffer: Short-term dialogue context window with SQLite persistence.
Maintains in-memory FIFO queue for fast turn access and backs up to data/chat.db.
"""

import os
import sqlite3
import time
from threading import Lock
from typing import List, Dict, Optional
from .models import ChatMessage

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "chat.db")


def _get_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp REAL NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_conv_user ON conversations(user_id)")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS processed_messages (
            message_id TEXT PRIMARY KEY,
            channel TEXT NOT NULL,
            processed_at REAL NOT NULL
        )
    """)
    conn.commit()
    return conn


class SessionBuffer:
    """Multi-turn short-term conversation context manager."""

    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self._cache: Dict[str, List[ChatMessage]] = {}
        self._lock = Lock()
        self._db = _get_db()

    def _load_user_history(self, user_id: str) -> List[ChatMessage]:
        cursor = self._db.cursor()
        cursor.execute(
            "SELECT role, content, timestamp FROM conversations WHERE user_id = ? ORDER BY id ASC LIMIT 50",
            (user_id,)
        )
        rows = cursor.fetchall()
        return [ChatMessage(role=r[0], content=r[1], timestamp=r[2]) for r in rows]

    def get_history(self, user_id: str, limit: Optional[int] = None) -> List[ChatMessage]:
        """Returns recent conversation turns for context window assembly."""
        with self._lock:
            if user_id not in self._cache:
                self._cache[user_id] = self._load_user_history(user_id)
            history = self._cache[user_id]

        lim = limit or self.max_turns
        return history[-lim:]

    def add_turn(self, user_id: str, user_text: str, assistant_text: str):
        """Appends user query and bot reply to memory and commits to SQLite."""
        now = time.time()
        u_msg = ChatMessage(role="user", content=user_text, timestamp=now)
        a_msg = ChatMessage(role="assistant", content=assistant_text, timestamp=now + 0.001)

        with self._lock:
            if user_id not in self._cache:
                self._cache[user_id] = self._load_user_history(user_id)
            self._cache[user_id].extend([u_msg, a_msg])
            # Keep cache bounded
            if len(self._cache[user_id]) > self.max_turns * 3:
                self._cache[user_id] = self._cache[user_id][-self.max_turns * 2:]

            # Persist to SQLite
            cursor = self._db.cursor()
            cursor.execute(
                "INSERT INTO conversations (user_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
                (user_id, "user", user_text, now)
            )
            cursor.execute(
                "INSERT INTO conversations (user_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
                (user_id, "assistant", assistant_text, now + 0.001)
            )
            self._db.commit()

    def clear_history(self, user_id: str):
        """Clears conversational history for the given user."""
        with self._lock:
            self._cache[user_id] = []
            cursor = self._db.cursor()
            cursor.execute("DELETE FROM conversations WHERE user_id = ?", (user_id,))
            self._db.commit()

    def is_message_processed(self, message_id: str) -> bool:
        """Checks if a WhatsApp/channel message ID has already been processed."""
        if not message_id:
            return False
        cursor = self._db.cursor()
        cursor.execute("SELECT 1 FROM processed_messages WHERE message_id = ?", (message_id,))
        return cursor.fetchone() is not None

    def mark_message_processed(self, message_id: str, channel: str = "whatsapp"):
        """Records processed message ID to prevent duplicate handling on network retries."""
        if not message_id:
            return
        with self._lock:
            try:
                cursor = self._db.cursor()
                cursor.execute(
                    "INSERT OR IGNORE INTO processed_messages (message_id, channel, processed_at) VALUES (?, ?, ?)",
                    (message_id, channel, time.time())
                )
                self._db.commit()
            except Exception:
                pass

    def delete_all_user_data(self, user_id: str):
        """Completely purges all data (dialogue history and brain memories) for a user."""
        with self._lock:
            self._cache[user_id] = []
            cursor = self._db.cursor()
            cursor.execute("DELETE FROM conversations WHERE user_id = ?", (user_id,))
            cursor.execute("DELETE FROM user_memories WHERE user_id = ?", (user_id,))
            self._db.commit()

    def get_all_users(self) -> List[str]:
        """Returns list of all distinct users with conversation history."""
        cursor = self._db.cursor()
        cursor.execute("SELECT DISTINCT user_id FROM conversations ORDER BY timestamp DESC")
        rows = cursor.fetchall()
        users = [r[0] for r in rows]
        if "default" not in users:
            users.insert(0, "default")
        return users


session_buffer = SessionBuffer(max_turns=8)

