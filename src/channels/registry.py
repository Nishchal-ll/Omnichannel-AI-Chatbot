"""
Channel registry.

To add Messenger, Threads, Telegram etc:
  1. Create src/channels/messenger.py with class MessengerChannel(BaseChannel)
  2. Import it here and add to CHANNELS list — that's it.
     The pipeline, webhook server, and dashboard all auto-discover channels.
"""
from .base import BaseChannel
from .whatsapp import WhatsAppChannel
from .email import EmailChannel
from .messenger import MessengerChannel

# ── Register all active channels here ────────────────────────────────────────
# Order doesn't matter — lookup is by channel_name string.
_CHANNEL_LIST: list[BaseChannel] = [
    WhatsAppChannel(),
    EmailChannel(),
    MessengerChannel(),
    # ThreadsChannel(),
    # TelegramChannel(),
]

# Build lookup dict: { "whatsapp": <WhatsAppChannel>, "email": <EmailChannel>, ... }
_REGISTRY: dict[str, BaseChannel] = {ch.channel_name: ch for ch in _CHANNEL_LIST}


def get_channel(name: str) -> BaseChannel:
    """
    Get a channel by name.
    Raises KeyError if channel not registered.
    Usage: get_channel("whatsapp").send(message, reply)
    """
    if name not in _REGISTRY:
        raise KeyError(
            f"Channel '{name}' not registered. "
            f"Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[name]


def all_channels() -> list[BaseChannel]:
    """Return all registered channel instances."""
    return list(_REGISTRY.values())


def polling_channels() -> list[BaseChannel]:
    """Return channels that support fetch_unread() — i.e. email-like channels."""
    return [ch for ch in _CHANNEL_LIST if ch.fetch_unread() != []]
