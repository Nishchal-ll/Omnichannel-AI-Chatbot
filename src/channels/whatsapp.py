"""
WhatsApp channel — sends via Meta Cloud API v20.
Receives via webhook (see webhook_server.py).
"""
import os
import requests
from .base import BaseChannel, Message


class WhatsAppChannel(BaseChannel):

    def __init__(self):
        self._access_token = os.getenv("WHATSAPP_ACCESS_TOKEN", "")
        self._phone_id = os.getenv("WHATSAPP_PHONE_ID", "")
        self._api_url = f"https://graph.facebook.com/v20.0/{self._phone_id}/messages"

    @property
    def channel_name(self) -> str:
        return "whatsapp"

    def send(self, message: Message, reply_text: str) -> bool:
        """Send a WhatsApp text reply to message.sender."""
        token = os.getenv("WHATSAPP_ACCESS_TOKEN", self._access_token)
        phone_id = os.getenv("WHATSAPP_PHONE_ID", self._phone_id)

        if not token or not phone_id:
            print("[WhatsApp] ✗ Missing ACCESS_TOKEN or PHONE_ID in .env")
            return False

        url = f"https://graph.facebook.com/v20.0/{phone_id}/messages"
        payload = {
            "messaging_product": "whatsapp",
            "to": message.sender,
            "type": "text",
            "text": {"body": reply_text},
        }

        # Thread context if message_id exists
        if message.message_id:
            payload["context"] = {"message_id": message.message_id}

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=10)
            if r.status_code == 200:
                print(f"[WhatsApp] ✓ Replied to {message.sender}: {r.text}")
                return True
            else:
                print(f"[WhatsApp] ✗ API {r.status_code}: {r.text}")
                return False
        except Exception as e:
            print(f"[WhatsApp] ✗ Exception: {e}")
            return False

    # fetch_unread() not needed — WhatsApp uses webhooks
