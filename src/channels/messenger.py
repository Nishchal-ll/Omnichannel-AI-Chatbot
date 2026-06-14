"""
Messenger channel — STUB for future implementation.

To activate:
  1. Fill in the send() method with Meta Messenger Send API calls
  2. Uncomment the import in src/channels/registry.py
  3. Add MessengerChannel() to _CHANNEL_LIST in registry.py
  4. Add a /messenger-webhook POST route in webhook_server.py

No changes needed in pipeline.py, store.py, or streamlit_app.py —
they all work with any channel automatically.
"""
import os
import requests
from .base import BaseChannel, Message


class MessengerChannel(BaseChannel):

    def __init__(self):
        self._page_token = os.getenv("MESSENGER_ACCESS_TOKEN") or os.getenv("MESSENGER_PAGE_TOKEN") or ""

    @property
    def channel_name(self) -> str:
        return "messenger"

    def send(self, message: Message, reply_text: str) -> bool:
        """Send reply via Messenger Send API."""
        if not self._page_token:
            print("[Messenger] ✗ Missing MESSENGER_PAGE_TOKEN")
            return False

        payload = {
            "recipient": {"id": message.sender},
            "message": {"text": reply_text},
        }
        try:
            r = requests.post(
                "https://graph.facebook.com/v20.0/me/messages",
                params={"access_token": self._page_token},
                json=payload,
                timeout=10,
            )
            if r.status_code == 200:
                print(f"[Messenger] ✓ Replied to {message.sender}")
                return True
            print(f"[Messenger] ✗ {r.status_code}: {r.text}")
            return False
        except Exception as e:
            print(f"[Messenger] ✗ {e}")
            return False
