"""
src/webhook_server.py
─────────────────────
FastAPI Server supporting:
1. Pure Web Chat API (Fast Q&A, Short-Term Context Window, SQLite Brain Memory).
2. Diagnostic telemetry and memory inspection endpoints.
3. Preserved WhatsApp Webhook endpoints (/webhook, /messenger-webhook) using Orchestrator.
"""

import os
import sys
import hmac
import hashlib
import json
import uvicorn
from fastapi import FastAPI, Request, Response, Query, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

load_dotenv(override=True)
sys.path.insert(0, os.path.dirname(__file__))

import store
from channels.base import Message
from stage3_brain.orchestrator import orchestrator
from stage2_memory.session import session_buffer
from stage2_memory.memory_service import memory_service
from stage2_memory.retriever import retriever

WEBHOOK_TOKEN = os.getenv("WHATSAPP_WEBHOOK_TOKEN", "mysecrettoken")
MESSENGER_WEBHOOK_TOKEN = os.getenv("MESSENGER_WEBHOOK_TOKEN", WEBHOOK_TOKEN)
APP_SECRET = os.getenv("WHATSAPP_APP_SECRET", os.getenv("APP_SECRET", ""))

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

app = FastAPI(title="Cognitive Brain Chat & Telemetry Console")

# Mount static assets
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ── Web UI Root ──────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return HTMLResponse("<h2>Cognitive Brain Chat</h2><p>Static files missing.</p>")


# ── Direct Chat Endpoints (Phase 2 & 3) ──────────────────────────────────────
class ChatRequest(BaseModel):
    user_id: str
    message: str


@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    """
    Direct Web Chat endpoint:
    Executes memory extraction, intent routing, RAG retrieval, and Gemma 4 generation.
    Returns response text, diagnostic metrics, and active brain memories.
    """
    user_id = req.user_id.strip() or "user_default"
    message = req.message.strip()
    if not message:
        return {"error": "Message is empty"}

    result = orchestrator.process(user_id=user_id, query=message, channel="chat")
    return result


@app.get("/api/users")
async def get_users():
    """Returns list of distinct users recorded in the database."""
    users = session_buffer.get_all_users()
    if not users:
        users = ["default_user"]
    return {"users": users}


@app.get("/api/history/{user_id}")
async def get_user_history(user_id: str):
    """Fetches full conversational context history for the user."""
    turns = session_buffer.get_history(user_id, limit=50)
    return {
        "user_id": user_id,
        "turns": [m.to_dict() for m in turns]
    }


@app.delete("/api/history/{user_id}")
async def clear_user_history(user_id: str):
    """Resets dialogue context for the user."""
    session_buffer.clear_history(user_id)
    return {"status": "cleared", "user_id": user_id}


@app.get("/api/memories/{user_id}")
async def get_user_memories(user_id: str):
    """Fetches persistent long-term brain memories saved in SQLite."""
    memories = memory_service.get_memories(user_id)
    return {
        "user_id": user_id,
        "memories": [m.to_dict() for m in memories]
    }


@app.delete("/api/memories/{user_id}")
async def clear_user_memories(user_id: str):
    """Clears all long-term brain memories for the user."""
    memory_service.clear_memories(user_id)
    return {"status": "cleared", "user_id": user_id}


@app.delete("/api/memories/{user_id}/{memory_id}")
async def delete_single_memory(user_id: str, memory_id: str):
    """Deletes a specific memory fact from SQLite."""
    deleted = memory_service.delete_memory(user_id, memory_id)
    return {"status": "deleted" if deleted else "not_found", "memory_id": memory_id}


# ── Health & Diagnostics ─────────────────────────────────────────────────────
@app.get("/api/health")
async def get_system_health():
    """System, model, and database readiness status."""
    is_ollama_ok, ollama_msg = orchestrator.brain.is_available()
    return {
        "status": "running",
        "ollama": {
            "online": is_ollama_ok,
            "model": orchestrator.brain.model,
            "details": ollama_msg,
        },
        "rag": {
            "collection": retriever.collection_name,
            "total_chunks": retriever.collection.count(),
        },
        "whatsapp": {
            "configured": bool(os.getenv("WHATSAPP_ACCESS_TOKEN")),
            "phone_id": bool(os.getenv("WHATSAPP_PHONE_ID")),
        },
    }


# ── WhatsApp Support Legacy & Simulation Endpoints ───────────────────────────
@app.get("/api/whatsapp/messages")
async def get_whatsapp_messages():
    all_msgs = store.get_messages()
    wa_msgs = [m for m in all_msgs if m.get("channel") == "whatsapp"]
    senders = sorted(list(set(m.get("sender") for m in wa_msgs if m.get("sender"))))
    return {
        "messages": wa_msgs,
        "senders": senders,
        "total": len(wa_msgs),
    }


@app.post("/api/whatsapp/simulate")
async def simulate_whatsapp_message(data: dict):
    sender = str(data.get("sender", "9779765341686")).strip()
    body = str(data.get("body", "")).strip()
    if not body:
        return {"error": "Message body is empty"}

    message = Message(channel="whatsapp", sender=sender, body=body)
    record = store.add_message(channel="whatsapp", sender=sender, body=body)

    # Process via orchestrator
    result = orchestrator.process(user_id=sender, query=body, channel="whatsapp", message_obj=message)
    store.set_reply(record["id"], result["reply"], status="replied", metrics=result["metrics"])

    updated = next((m for m in store.get_messages() if m["id"] == record["id"]), record)
    return {"message": updated, "orchestrator_result": result}


# ── Meta Webhook Verification (Meta GET) ─────────────────────────────────────
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


# ── Incoming Webhook Payload Handler (Meta POST) ─────────────────────────────
def _process_webhook_turn(message: Message):
    is_first_contact = len(session_buffer.get_history(message.sender)) == 0
    record = store.add_message(channel=message.channel, sender=message.sender, body=message.body)
    res = orchestrator.process(
        user_id=message.sender,
        query=message.body,
        channel=message.channel,
        message_obj=message
    )
    reply = res["reply"]
    # Item 4: Privacy notice on first WhatsApp contact
    if is_first_contact and message.channel == "whatsapp":
        reply = (
            "👋 *Welcome to the Lionel Messi AI Knowledge Assistant!*\n"
            "🔒 _Privacy Notice: We store your dialogue history and learned facts locally. "
            "You can type 'forget me' anytime to erase all your data._\n\n"
            + reply
        )
    store.set_reply(record["id"], reply, status="replied", metrics=res["metrics"])


def handle_incoming_payload(body: dict, background_tasks: BackgroundTasks):
    obj = body.get("object")
    if obj == "whatsapp_business_account":
        for entry in body.get("entry", []):
            for change in entry.get("changes", []):
                value = change.get("value", {})
                for msg in value.get("messages", []):
                    if msg.get("type") != "text":
                        continue
                    msg_id = msg.get("id", "")
                    # Item 5: WhatsApp deduplication check against Meta retry loops
                    if msg_id and session_buffer.is_message_processed(msg_id):
                        print(f"[Webhook] ↷ Duplicate message {msg_id} ignored")
                        continue
                    if msg_id:
                        session_buffer.mark_message_processed(msg_id, channel="whatsapp")

                    message = Message(
                        channel="whatsapp",
                        sender=msg["from"],
                        body=msg["text"]["body"],
                        message_id=msg_id,
                    )
                    print(f"[Webhook] ← WhatsApp from {message.sender}: {message.body}")
                    background_tasks.add_task(_process_webhook_turn, message)

    elif obj == "page":
        for entry in body.get("entry", []):
            for messaging in entry.get("messaging", []):
                sender_id = messaging.get("sender", {}).get("id")
                msg_data = messaging.get("message", {})
                msg_text = msg_data.get("text")
                msg_mid = msg_data.get("mid", "")
                if not msg_text or msg_data.get("is_echo"):
                    continue
                if msg_mid and session_buffer.is_message_processed(msg_mid):
                    print(f"[Webhook] ↷ Duplicate messenger message {msg_mid} ignored")
                    continue
                if msg_mid:
                    session_buffer.mark_message_processed(msg_mid, channel="messenger")

                message = Message(
                    channel="messenger",
                    sender=sender_id,
                    body=msg_text,
                    message_id=msg_mid,
                )
                print(f"[Webhook] ← Messenger from {message.sender}: {message.body}")
                background_tasks.add_task(_process_webhook_turn, message)


@app.post("/webhook")
async def receive(request: Request, background_tasks: BackgroundTasks):
    raw_body = await request.body()

    # Item 5: HMAC-SHA256 signature verification if APP_SECRET is configured
    if APP_SECRET:
        sig = request.headers.get("X-Hub-Signature-256", "")
        expected = "sha256=" + hmac.new(APP_SECRET.encode(), raw_body, hashlib.sha256).hexdigest()
        if not sig or not hmac.compare_digest(sig, expected):
            print("[Webhook] ✗ Invalid X-Hub-Signature-256")
            return Response(content="Forbidden: Invalid signature", status_code=403)

    try:
        body = json.loads(raw_body.decode("utf-8"))
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
    port = int(os.getenv("PORT", "8080"))
    print(f"[Server] Cognitive Brain Chat starting on http://0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
