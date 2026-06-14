"""
FastAPI webhook server — port 5001.
Receives WhatsApp events, creates Message objects, hands to Pipeline.
Pipeline figures out the reply and sends it back via the correct channel.
"""
import os, sys
import uvicorn
from fastapi import FastAPI, Request, Response, Query, BackgroundTasks
from dotenv import load_dotenv

load_dotenv(override=True)
sys.path.insert(0, os.path.dirname(__file__))

from channels.base import Message
from crew.pipeline import Pipeline

WEBHOOK_TOKEN = os.getenv("WHATSAPP_WEBHOOK_TOKEN", "mysecrettoken")
MESSENGER_WEBHOOK_TOKEN = os.getenv("MESSENGER_WEBHOOK_TOKEN", WEBHOOK_TOKEN)

app = FastAPI(title="Bot Webhook")
pipeline = Pipeline()   # single shared instance


# ── Webhook verification ─────────────────────────────────────────────────────
@app.get("/webhook")
async def verify(
    hub_mode: str = Query(None, alias="hub.mode"),
    hub_challenge: str = Query(None, alias="hub.challenge"),
    hub_verify_token: str = Query(None, alias="hub.verify_token"),
):
    if hub_mode == "subscribe" and hub_verify_token in (WEBHOOK_TOKEN, MESSENGER_WEBHOOK_TOKEN):
        print("[Webhook] ✓ Verified by Meta")
        return Response(content=hub_challenge, media_type="text/plain")
    print("[Webhook] ✗ Verification failed — wrong token")
    return Response(content="Forbidden", status_code=403)


@app.get("/messenger-webhook")
async def verify_messenger(
    hub_mode: str = Query(None, alias="hub.mode"),
    hub_challenge: str = Query(None, alias="hub.challenge"),
    hub_verify_token: str = Query(None, alias="hub.verify_token"),
):
    if hub_mode == "subscribe" and hub_verify_token == MESSENGER_WEBHOOK_TOKEN:
        print("[Messenger Webhook] ✓ Verified by Meta")
        return Response(content=hub_challenge, media_type="text/plain")
    print("[Messenger Webhook] ✗ Verification failed — wrong token")
    return Response(content="Forbidden", status_code=403)


# ── Payload Handler ──────────────────────────────────────────────────────────
def handle_incoming_payload(body: dict, background_tasks: BackgroundTasks):
    obj = body.get("object")
    
    # 1. WhatsApp Channel
    if obj == "whatsapp_business_account":
        for entry in body.get("entry", []):
            for change in entry.get("changes", []):
                value = change.get("value", {})
                for msg in value.get("messages", []):
                    if msg.get("type") != "text":
                        continue
                    message = Message(
                        channel="whatsapp",
                        sender=msg["from"],
                        body=msg["text"]["body"],
                    )
                    print(f"[Webhook] ← WhatsApp from {message.sender}: {message.body}")
                    background_tasks.add_task(pipeline.process, message)
                    
    # 2. Messenger Channel
    elif obj == "page":
        for entry in body.get("entry", []):
            for messaging in entry.get("messaging", []):
                sender_id = messaging.get("sender", {}).get("id")
                msg_data = messaging.get("message", {})
                msg_text = msg_data.get("text")
                # Ignore non-text messages and echoes
                if not msg_text or msg_data.get("is_echo"):
                    continue
                message = Message(
                    channel="messenger",
                    sender=sender_id,
                    body=msg_text,
                )
                print(f"[Webhook] ← Messenger from {message.sender}: {message.body}")
                background_tasks.add_task(pipeline.process, message)


# ── Incoming message routes ──────────────────────────────────────────────────
@app.post("/webhook")
async def receive(request: Request, background_tasks: BackgroundTasks):
    try:
        body = await request.json()
    except Exception:
        return {"status": "ok"}

    handle_incoming_payload(body, background_tasks)
    return {"status": "ok"}


@app.post("/messenger-webhook")
async def receive_messenger(request: Request, background_tasks: BackgroundTasks):
    try:
        body = await request.json()
    except Exception:
        return {"status": "ok"}

    handle_incoming_payload(body, background_tasks)
    return {"status": "ok"}


@app.get("/health")
async def health():
    return {"status": "running"}


if __name__ == "__main__":
    print("[Webhook] Starting on :5001")
    uvicorn.run(app, host="0.0.0.0", port=5001, log_level="info")
