"""
src/agent/session.py
────────────────────
Multi-turn conversation session buffer and user contact memory.
Preserves the last N conversation turns per sender and stores client contact/profile details.
"""

from threading import Lock
from typing import List, Dict, Optional, Any
from datetime import datetime


class SessionBuffer:
    """
    In-memory multi-turn dialogue buffer keyed by sender_id.
    Maintains user and assistant turns for contextual pronoun resolution and history injection.
    """
    def __init__(self, max_turns: int = 6):
        self.max_turns = max_turns
        self._lock = Lock()
        self._sessions: Dict[str, List[Dict[str, str]]] = {}
        self._user_contacts: Dict[str, Dict[str, Any]] = {}

    def add_turn(self, sender: str, user_msg: str, assistant_reply: str):
        """Append a completed exchange (user + assistant) to the sender's history."""
        with self._lock:
            if sender not in self._sessions:
                self._sessions[sender] = []
            
            self._sessions[sender].append({"role": "user", "content": user_msg})
            self._sessions[sender].append({"role": "assistant", "content": assistant_reply})
            
            # Keep only the last max_turns exchanges (each exchange is 2 entries)
            max_entries = self.max_turns * 2
            if len(self._sessions[sender]) > max_entries:
                self._sessions[sender] = self._sessions[sender][-max_entries:]

    def get_history(self, sender: str, max_turns: Optional[int] = None) -> List[Dict[str, str]]:
        """Retrieve recent conversation history for a sender."""
        with self._lock:
            history = self._sessions.get(sender, [])
            limit = (max_turns or self.max_turns) * 2
            return list(history[-limit:])

    def update_user_contact(self, sender: str, **kwargs):
        """Track customer profile information (e.g., client code, name, email, phone)."""
        with self._lock:
            if sender not in self._user_contacts:
                self._user_contacts[sender] = {
                    "sender_id": sender,
                    "first_seen": datetime.utcnow().isoformat(),
                }
            self._user_contacts[sender].update(kwargs)
            self._user_contacts[sender]["last_updated"] = datetime.utcnow().isoformat()

    def get_user_contact(self, sender: str) -> Dict[str, Any]:
        """Get tracked profile and contact information for a sender."""
        with self._lock:
            return dict(self._user_contacts.get(sender, {"sender_id": sender}))

    def clear(self, sender: Optional[str] = None):
        """Clear history for a specific sender or all sessions."""
        with self._lock:
            if sender:
                self._sessions.pop(sender, None)
                self._user_contacts.pop(sender, None)
            else:
                self._sessions.clear()
                self._user_contacts.clear()


# Global shared instance
session_buffer = SessionBuffer(max_turns=6)
