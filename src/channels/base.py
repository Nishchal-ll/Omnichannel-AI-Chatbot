"""
Abstract base class for all messaging channels.

To add a new channel (Messenger, Threads, Telegram, etc.):
1. Create src/channels/messenger.py
2. Subclass BaseChannel
3. Implement send(), channel_name, and optionally fetch_unread()
4. Register it in src/channels/registry.py
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Message:
    """Unified message object across all channels."""
    channel: str                    # "whatsapp" | "email" | "messenger" | ...
    sender: str                     # phone number, email address, user id
    body: str                       # plain text content
    subject: Optional[str] = None   # email subject (None for chat channels)
    message_id: Optional[str] = None  # original message ID for threading
    uid: Optional[str] = None       # channel-specific UID (e.g. IMAP uid)
    ts: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: dict = field(default_factory=dict)  # channel-specific extras

    def to_dict(self) -> dict:
        return {
            "channel": self.channel,
            "sender": self.sender,
            "body": self.body,
            "subject": self.subject,
            "message_id": self.message_id,
            "uid": self.uid,
            "ts": self.ts,
            "metadata": self.metadata,
        }


class BaseChannel(ABC):
    """
    Every channel must implement:
      - channel_name: str property
      - send(message, reply_text): sends reply back to sender
      - fetch_unread(): returns list[Message] (for polling channels like email)
    """

    @property
    @abstractmethod
    def channel_name(self) -> str:
        """Unique identifier string e.g. 'whatsapp', 'email', 'messenger'"""
        ...

    @abstractmethod
    def send(self, message: Message, reply_text: str) -> bool:
        """
        Send reply_text back to the sender of message.
        Returns True on success, False on failure.
        """
        ...

    def fetch_unread(self) -> list[Message]:
        """
        For polling channels (email): return list of unread messages.
        Webhook channels (WhatsApp, Messenger) can leave this as-is.
        """
        return []

    def __repr__(self):
        return f"<Channel: {self.channel_name}>"
