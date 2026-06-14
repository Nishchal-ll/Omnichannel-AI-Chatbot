"""
Email channel — fetches via Gmail IMAP, replies via SMTP.
"""
import os
import imaplib
import smtplib
import email as email_lib
from email.message import EmailMessage
from email.header import decode_header
from .base import BaseChannel, Message


def _decode_str(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    parts = decode_header(value)
    out = []
    for part, enc in parts:
        if isinstance(part, bytes):
            out.append(part.decode(enc or "utf-8", errors="replace"))
        else:
            out.append(str(part))
    return " ".join(out)


class EmailChannel(BaseChannel):

    IMAP_HOST = "imap.gmail.com"
    SMTP_HOST = "smtp.gmail.com"
    SMTP_PORT = 587

    def __init__(self):
        self._user = os.getenv("GMAIL_USER", "")
        self._password = os.getenv("GMAIL_APP_PASSWORD", "")

    @property
    def channel_name(self) -> str:
        return "email"

    def send(self, message: Message, reply_text: str) -> bool:
        """Send email reply back to message.sender."""
        if not self._user or not self._password:
            print("[Email] ✗ Missing GMAIL_USER or GMAIL_APP_PASSWORD")
            return False

        subject = message.subject or "(no subject)"
        reply_subject = f"Re: {subject}" if not subject.startswith("Re:") else subject

        msg = EmailMessage()
        msg["From"] = self._user
        msg["To"] = message.sender
        msg["Subject"] = reply_subject
        if message.message_id:
            msg["In-Reply-To"] = message.message_id
            msg["References"] = message.message_id
        msg.set_content(reply_text)

        try:
            with smtplib.SMTP(self.SMTP_HOST, self.SMTP_PORT) as smtp:
                smtp.starttls()
                smtp.login(self._user, self._password)
                smtp.send_message(msg)
            print(f"[Email] ✓ Replied to {message.sender}")
            return True
        except Exception as e:
            print(f"[Email] ✗ Send error: {e}")
            return False

    def fetch_unread(self) -> list[Message]:
        """Fetch unread Gmail messages, return as Message objects."""
        if not self._user or not self._password:
            print("[Email] ✗ Missing credentials")
            return []

        results = []
        try:
            mail = imaplib.IMAP4_SSL(self.IMAP_HOST)
            mail.login(self._user, self._password)
            mail.select("INBOX")

            _, data = mail.search(None, "UNSEEN")
            uids = data[0].split()

            for uid in uids[-20:]:
                _, msg_data = mail.fetch(uid, "(RFC822)")
                raw = msg_data[0][1]
                parsed = email_lib.message_from_bytes(raw)

                from_addr = _decode_str(parsed.get("From", ""))
                subject = _decode_str(parsed.get("Subject", "(no subject)"))
                message_id = parsed.get("Message-ID", "")

                body = ""
                if parsed.is_multipart():
                    for part in parsed.walk():
                        if part.get_content_type() == "text/plain" and \
                           not part.get("Content-Disposition"):
                            body = part.get_payload(decode=True).decode("utf-8", errors="replace")
                            break
                else:
                    body = parsed.get_payload(decode=True).decode("utf-8", errors="replace")

                results.append(Message(
                    channel="email",
                    sender=from_addr,
                    body=body.strip()[:2000],
                    subject=subject,
                    message_id=message_id,
                    uid=uid.decode(),
                ))

            mail.logout()
        except Exception as e:
            print(f"[Email] ✗ Fetch error: {e}")

        return results

    def mark_read(self, uid: str):
        """Mark an email as read by IMAP uid."""
        try:
            mail = imaplib.IMAP4_SSL(self.IMAP_HOST)
            mail.login(self._user, self._password)
            mail.select("INBOX")
            mail.store(uid, "+FLAGS", "\\Seen")
            mail.logout()
        except Exception as e:
            print(f"[Email] ✗ Mark-read error: {e}")
