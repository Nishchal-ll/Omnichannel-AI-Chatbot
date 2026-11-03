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
WHATSAPP_APP_SECRET = os.getenv("WHATSAPP_APP_SECRET", os.getenv("APP_SECRET", ""))
MESSENGER_APP_SECRET = os.getenv("MESSENGER_APP_SECRET", os.getenv("APP_SECRET", ""))

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

app = FastAPI(title="Apex Auto Recondition Nepal — Support Analytics & Inquiries Console")


@app.middleware("http")
async def add_no_cache_headers(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/static") or request.url.path == "/":
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


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
    Executes customer preference extraction, intent routing, vehicle inventory RAG retrieval, and Llama 3.2 generation.
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


# ── Dealership Reports & Analytics Endpoints ─────────────────────────────────
@app.get("/api/reports/overview")
async def get_reports_overview():
    """Returns analytics overview across WhatsApp, Messenger, and Web channels."""
    import re
    messages = store.get_messages()
    all_users = session_buffer.get_all_users()
    
    senders = set(m.get("sender") for m in messages if m.get("sender"))
    for u in all_users:
        senders.add(u)
    
    wa_count = sum(1 for m in messages if m.get("channel") == "whatsapp")
    fb_count = sum(1 for m in messages if m.get("channel") == "messenger")
    chat_count = sum(1 for m in messages if m.get("channel") == "chat")
    if chat_count == 0 and len(all_users) > 0:
        chat_count = sum(len(session_buffer.get_history(u)) for u in all_users) // 2

    TOPIC_PATTERNS = {
        "Hyundai Santro": re.compile(r"\bsantro\b", re.IGNORECASE),
        "Grand i10": re.compile(r"\bi10\b|\bnios\b", re.IGNORECASE),
        "Nissan Magnite": re.compile(r"\bmagnite\b", re.IGNORECASE),
        "Maruti Alto": re.compile(r"\balto\b", re.IGNORECASE),
        "Maruti Swift": re.compile(r"\bswift\b", re.IGNORECASE),
        "Maruti Brezza": re.compile(r"\bbrezza\b", re.IGNORECASE),
        "Hyundai Creta": re.compile(r"\bcreta\b", re.IGNORECASE),
        "Tata Nexon/Tiago": re.compile(r"\bnexon\b|\btiago\b", re.IGNORECASE),
        "Exchange (साटासाट)": re.compile(r"\bexchange\b|\bsatasat\b|\bswap\b", re.IGNORECASE),
        "Bluebook (नामसारी)": re.compile(r"\bbluebook\b|\bnamsari\b|\byatayat\b", re.IGNORECASE),
        "Bank Financing / EMI": re.compile(r"\bfinance\b|\bemi\b|\bloan\b", re.IGNORECASE),
    }
    topic_counts = {t: 0 for t in TOPIC_PATTERNS}
    latencies = []

    for m in messages:
        text = (m.get("body") or "") + " " + (m.get("reply") or "")
        for t, rx in TOPIC_PATTERNS.items():
            if rx.search(text):
                topic_counts[t] += 1
        metric = m.get("metrics") or {}
        if metric.get("total_ms"):
            latencies.append(metric["total_ms"])

    # Also inspect history from session buffer
    for u in all_users:
        for turn in session_buffer.get_history(u):
            for t, rx in TOPIC_PATTERNS.items():
                if rx.search(turn.content):
                    topic_counts[t] += 1

    avg_latency = round(sum(latencies) / len(latencies), 1) if latencies else 2100.0

    return {
        "total_queries": len(messages) if messages else sum(len(session_buffer.get_history(u)) for u in all_users) // 2,
        "unique_leads": len(senders),
        "channels": {
            "whatsapp": wa_count,
            "messenger": fb_count,
            "web_chat": chat_count,
        },
        "avg_latency_ms": avg_latency,
        "top_topics": sorted(
            [{"name": k, "count": v} for k, v in topic_counts.items() if v > 0],
            key=lambda x: x["count"],
            reverse=True
        )[:8],
        "channel_status": {
            "whatsapp_configured": bool(os.getenv("WHATSAPP_ACCESS_TOKEN")),
            "whatsapp_phone_id": bool(os.getenv("WHATSAPP_PHONE_ID")),
            "messenger_configured": bool(os.getenv("MESSENGER_PAGE_TOKEN") or os.getenv("MESSENGER_ACCESS_TOKEN")),
        },
        "inventory_chunks": retriever.collection.count(),
        "model": orchestrator.brain.model,
    }


@app.get("/api/reports/customers")
async def get_reports_customers():
    """Returns list of customer leads with metadata, last query, and status."""
    messages = store.get_messages()
    all_users = session_buffer.get_all_users()
    customers_map = {}

    for m in messages:
        sender = m.get("sender")
        if not sender:
            continue
        channel = m.get("channel") or "whatsapp"
        if sender not in customers_map:
            customers_map[sender] = {
                "sender_id": sender,
                "channel": channel,
                "first_seen": m.get("ts"),
                "last_active": m.get("ts"),
                "total_inquiries": 0,
                "last_message": m.get("body", ""),
                "last_reply": m.get("reply", ""),
            }
        customers_map[sender]["total_inquiries"] += 1
        customers_map[sender]["last_active"] = m.get("ts")
        customers_map[sender]["last_message"] = m.get("body", "")
        if m.get("reply"):
            customers_map[sender]["last_reply"] = m.get("reply")

    for u in all_users:
        if u not in customers_map:
            turns = session_buffer.get_history(u, limit=10)
            last_text = turns[-1].content if turns else ""
            customers_map[u] = {
                "sender_id": u,
                "channel": "whatsapp" if (u.startswith("+") or u.replace("+", "").isdigit()) else ("messenger" if "fb" in u.lower() or "messenger" in u.lower() else "chat"),
                "first_seen": "Recent",
                "last_active": "Recent",
                "total_inquiries": max(1, len(turns) // 2),
                "last_message": last_text,
                "last_reply": "",
            }

    cust_list = list(customers_map.values())
    cust_list.sort(key=lambda c: str(c["last_active"]), reverse=True)
    return {"customers": cust_list, "total": len(cust_list)}


@app.get("/api/reports/customer/{sender_id}")
async def get_customer_report_detail(sender_id: str):
    """Returns complete turn transcript and SQLite brain preferences for customer."""
    turns = session_buffer.get_history(sender_id, limit=50)
    memories = memory_service.get_memories(sender_id)
    raw_msgs = [m for m in store.get_messages() if m.get("sender") == sender_id]

    return {
        "sender_id": sender_id,
        "turns": [t.to_dict() for t in turns],
        "memories": [m.to_dict() for m in memories],
        "raw_messages": raw_msgs,
    }


class ChannelSimulateRequest(BaseModel):
    channel: str = "whatsapp"
    sender: str = "+977-9801234567"
    message: str


@app.post("/api/channels/simulate")
async def simulate_channel_message(req: ChannelSimulateRequest):
    """Simulates an inbound WhatsApp or Messenger inquiry."""
    import time
    channel = req.channel.lower().strip() or "whatsapp"
    sender = req.sender.strip() or ("+977-9801234567" if channel == "whatsapp" else "fb_customer_101")
    body = req.message.strip()
    if not body:
        return {"error": "Message body is empty"}

    msg_obj = Message(
        channel=channel,
        sender=sender,
        body=body,
        message_id=f"sim_{channel}_{int(time.time()*1000)}"
    )
    record = store.add_message(channel=channel, sender=sender, body=body)

    res = orchestrator.process(
        user_id=sender,
        query=body,
        channel=channel,
        message_obj=msg_obj
    )
    store.set_reply(record["id"], res["reply"], status="replied", metrics=res["metrics"])
    return {
        "channel": channel,
        "sender": sender,
        "query": body,
        "reply": res["reply"],
        "metrics": res["metrics"],
        "user_memories": res.get("user_memories", [])
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
            "🚗 *Welcome to Apex Auto Recondition House!*\n"
            "🔒 _Privacy Notice: We store your dialogue history and vehicle preferences locally to personalize recommendations. "
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

    # HMAC-SHA256 signature verification if WHATSAPP_APP_SECRET is configured
    if WHATSAPP_APP_SECRET:
        sig = request.headers.get("X-Hub-Signature-256", "")
        expected = "sha256=" + hmac.new(WHATSAPP_APP_SECRET.encode(), raw_body, hashlib.sha256).hexdigest()
        if not sig or not hmac.compare_digest(sig, expected):
            print("[Webhook] ✗ Invalid WhatsApp X-Hub-Signature-256")
            return Response(content="Forbidden: Invalid signature", status_code=403)

    try:
        body = json.loads(raw_body.decode("utf-8"))
    except Exception:
        return {"status": "ok"}
    handle_incoming_payload(body, background_tasks)
    return {"status": "ok"}


@app.post("/messenger-webhook")
async def receive_messenger(request: Request, background_tasks: BackgroundTasks):
    raw_body = await request.body()

    # HMAC-SHA256 signature verification if MESSENGER_APP_SECRET is configured
    if MESSENGER_APP_SECRET:
        sig = request.headers.get("X-Hub-Signature-256", "")
        expected = "sha256=" + hmac.new(MESSENGER_APP_SECRET.encode(), raw_body, hashlib.sha256).hexdigest()
        if not sig or not hmac.compare_digest(sig, expected):
            print("[Messenger Webhook] ✗ Invalid Messenger X-Hub-Signature-256")
            return Response(content="Forbidden: Invalid signature", status_code=403)

    try:
        body = json.loads(raw_body.decode("utf-8"))
    except Exception:
        return {"status": "ok"}
    handle_incoming_payload(body, background_tasks)
    return {"status": "ok"}
    return {"status": "ok"}


@app.get("/health")
async def health():
    return {"status": "running"}


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    print(f"[Server] Cognitive Brain Chat starting on http://0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
