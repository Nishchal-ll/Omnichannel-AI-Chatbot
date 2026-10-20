"""
src/agent/pipeline.py
─────────────────────
Main Pipeline Orchestrator for Naasa Securities (Broker 58) Customer Support Agent.
Replaces the legacy CrewAI pipeline with a fast, zero-overhead architecture:
1. Message Persistence (data/messages.json)
2. User Contact & Profile Extraction
3. Intent Routing & 0ms Fast Anchors
4. Out-of-Domain Guard Rejection
5. Multi-Turn Session Context & Query Rewriting
6. Section-Aware ChromaDB RAG Retrieval
7. Grounded Ollama Gemma 4 Inference
8. Channel Dispatch & Status Updating
"""

import os
import sys
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from channels.base import Message
from channels.registry import get_channel
from knowledge import get_relevant_knowledge
import store

from .session import session_buffer
from .router import IntentRouter
from .prompt import build_messages, OUT_OF_CONTEXT_DEFLECTION
from .ollama_client import OllamaClient

log = logging.getLogger(__name__)


class Pipeline:
    """
    Stateless, multi-thread safe pipeline.
    Invoked by Webhook server, Email poller, Streamlit dashboard, or CLI tests.
    """

    def __init__(self):
        self.client = OllamaClient()

    def process(self, message: Message, record_id: str = None) -> str:
        """Process an incoming message and return the generated answer."""
        sender = message.sender
        query = message.body.strip()

        # 1. Audit / Store Message
        if record_id is None:
            record = store.add_message(
                channel=message.channel,
                sender=sender,
                body=query,
                subject=message.subject,
            )
            record_id = record["id"]

        print(f"[Pipeline] Processing [{message.channel}] from {sender}: {query[:60]}...")

        # 2. Extract and update contact / profile info (client code, phone, email)
        contact_info = IntentRouter.extract_contact_info(query)
        if contact_info:
            session_buffer.update_user_contact(sender, **contact_info)

        user_contact = session_buffer.get_user_contact(sender)

        # 3. Check 0ms Fast Paths (Greetings, Signoffs)
        fast_reply = IntentRouter.check_fast_path(query)
        if fast_reply:
            print("[Pipeline] ✓ Fast-path anchor triggered")
            session_buffer.add_turn(sender, query, fast_reply)
            self._dispatch_and_persist(message, record_id, fast_reply)
            return fast_reply

        # 4. Out-of-Domain Rejection Guard
        if IntentRouter.is_out_of_domain(query):
            print("[Pipeline] ✗ Out-of-domain query detected — deflecting")
            deflection = OUT_OF_CONTEXT_DEFLECTION
            session_buffer.add_turn(sender, query, deflection)
            self._dispatch_and_persist(message, record_id, deflection)
            return deflection

        # 5. Multi-Turn Session History & Query Rewriting
        chat_history = session_buffer.get_history(sender)
        standalone_query = IntentRouter.rewrite_query(query, chat_history)
        if standalone_query != query:
            print(f"[Pipeline] Query rewritten: \"{query}\" → \"{standalone_query}\"")

        # 6. Retrieve Domain Knowledge from ChromaDB
        kb_context = get_relevant_knowledge(standalone_query, top_k=3, max_distance=0.85)

        # 7. Build Grounded Prompt
        messages = build_messages(
            user_query=query,
            knowledge_context=kb_context,
            chat_history=chat_history,
            user_contact=user_contact
        )

        # 8. Ollama Gemma 4 Inference
        reply_text = self.client.chat_complete(messages)
        print(f"[Pipeline] Ollama Reply: {reply_text[:80]}...")

        # 9. Update Session Memory
        session_buffer.add_turn(sender, query, reply_text)

        # 10. Dispatch to Channel & Persist
        self._dispatch_and_persist(message, record_id, reply_text)
        return reply_text

    def _dispatch_and_persist(self, message: Message, record_id: str, reply_text: str):
        """Sends reply back via channel (if configured) and updates audit store."""
        status = "replied"
        try:
            channel = get_channel(message.channel)
            ok = channel.send(message, reply_text)
            status = "replied" if ok else "saved_local"
        except KeyError:
            # Local test or unregistered channel
            status = "saved_local"
        except Exception as e:
            print(f"[Pipeline] Channel send warning: {e}")
            status = "saved_local"

        store.set_reply(record_id, reply_text, status)
