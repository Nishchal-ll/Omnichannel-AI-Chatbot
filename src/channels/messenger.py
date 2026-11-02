"""
src/channels/messenger.py
─────────────────────────
Facebook Messenger channel — sends via Meta Send API v21.0.
Receives via messenger webhook in webhook_server.py.
"""

import os
import requests
from .base import BaseChannel, Message


class MessengerChannel(BaseChannel):

    def __init__(self):
        self._page_token = os.getenv("MESSENGER_PAGE_TOKEN") or os.getenv("MESSENGER_ACCESS_TOKEN") or ""

    @property
    def channel_name(self) -> str:
        return "messenger"

    def send(self, message: Message, reply_text: str) -> bool:
        """Send reply via Meta Messenger Send API v21.0."""
        token = (os.getenv("MESSENGER_PAGE_TOKEN") or os.getenv("MESSENGER_ACCESS_TOKEN") or self._page_token).strip()
        if not token:
            print("[Messenger] ℹ️ Missing MESSENGER_PAGE_TOKEN in .env (simulation mode)")
            return False

        payload = {
            "recipient": {"id": message.sender},
            "message": {"text": reply_text},
        }
        try:
            r = requests.post(
                "https://graph.facebook.com/v21.0/me/messages",
                params={"access_token": token},
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10,
            )
            if r.status_code == 200:
                print(f"[Messenger] ✓ Replied to {message.sender}: {r.text}")
                return True
            print(f"[Messenger] ✗ {r.status_code}: {r.text}")
            return False
        except Exception as e:
            print(f"[Messenger] ✗ Exception: {e}")
            return False
