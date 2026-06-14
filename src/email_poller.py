"""
Email poller — polls Gmail every EMAIL_POLL_INTERVAL seconds.
Builds Message objects and hands them to Pipeline.
Pipeline replies via EmailChannel automatically.
"""
import os, sys, time
from dotenv import load_dotenv

load_dotenv(override=True)
sys.path.insert(0, os.path.dirname(__file__))

from channels.email import EmailChannel
from crew.pipeline import Pipeline

POLL_INTERVAL = int(os.getenv("EMAIL_POLL_INTERVAL", "60"))
email_channel = EmailChannel()
pipeline = Pipeline()


def poll_once():
    messages = email_channel.fetch_unread()
    if not messages:
        return

    print(f"[Email Poller] {len(messages)} unread email(s)")
    for message in messages:
        print(f"[Email Poller] ← From {message.sender}: {message.subject}")
        # Pipeline: classify → respond → send reply via EmailChannel
        pipeline.process(message)
        # Mark as read after processing
        email_channel.mark_read(message.uid)


def run():
    print(f"[Email Poller] Starting — polling every {POLL_INTERVAL}s")
    while True:
        try:
            poll_once()
        except Exception as e:
            print(f"[Email Poller] ✗ Error: {e}")
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    run()
