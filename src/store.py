"""Shared file-backed message store. All processes read/write the same JSON file."""
import json, os
from datetime import datetime
from threading import Lock

DATA_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "messages.json")
_lock = Lock()
_messages: list[dict] = []


def _load():
    global _messages
    try:
        os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
        if os.path.exists(DATA_FILE):
            with open(DATA_FILE) as f:
                _messages = json.load(f)
    except Exception:
        _messages = []


def _save():
    try:
        with open(DATA_FILE, "w") as f:
            json.dump(_messages[-500:], f, indent=2)
    except Exception:
        pass


_load()


def add_message(channel: str, sender: str, body: str, subject: str = None) -> dict:
    with _lock:
        msg = {
            "id": f"{channel}_{int(datetime.utcnow().timestamp()*1000)}",
            "channel": channel,
            "sender": sender,
            "body": body,
            "subject": subject,
            "reply": None,
            "ts": datetime.utcnow().isoformat(),
            "status": "pending",
        }
        _messages.append(msg)
        _save()
        return msg


def set_reply(msg_id: str, reply: str, status: str = "replied"):
    with _lock:
        for m in _messages:
            if m["id"] == msg_id:
                m["reply"] = reply
                m["status"] = status
                break
        _save()


def get_messages(channel: str = None) -> list[dict]:
    _load()  # always reload so Streamlit sees live updates
    with _lock:
        if channel:
            return [m for m in _messages if m["channel"] == channel]
        return list(_messages)
